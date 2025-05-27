import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import click

logger = logging.getLogger("Main process")


def create_executor(engine_dir, trtllm_config=None, gpu_ids=None):
    """
    Creates and returns a TensorRT-LLM executor instance.

    Args:
        engine_dir (str): Path to the TensorRT-LLM engine directory
        trtllm_config (dict): Configuration for the TensorRT-LLM executor

    Returns:
        trtllm.Executor: The initialized executor object
    """
    from tensorrt_llm.bindings import executor as trtllm
    trtllm_config = trtllm.ExecutorConfig(
        1,
        enable_chunked_context=True,
        # kv_cache_config=kv_cache_config,
        # parallel_config=parallel_config,
        # scheduler_config=scheduler_config,
    )

    executor = trtllm.Executor(Path(engine_dir), trtllm.ModelType.DECODER_ONLY,
                               trtllm_config)
    return executor


def batch(request_list: list[tuple[list[int], int]],
          engine_path: str,
          n_workers: int,
          batch_send_queues: list[mp.Queue],
          batch_recv_queue: mp.Queue,
          trtllm_config=None):

    total_nr_requests = len(request_list)
    recv_results = []

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(engine_path)

    logger.info(f"Batch mode based on {n_workers} workers")

    sum_tokens = 0
    nr_reqs = 0

    # Measure the time it takes to send requests to workers
    send_start_time = time.time()

    for idx, kv in enumerate(request_list):
        sum_tokens += len(kv['input_ids'])
        nr_reqs += 1

        batch_send_queues[nr_reqs % n_workers].put(
            (kv['input_ids'], kv['output_tokens'], idx))

    for queue in batch_send_queues:
        queue.put((None, None, None))

    send_end_time = time.time()
    send_time = send_end_time - send_start_time

    # Measure the time it takes to receive and process results
    receive_start_time = time.time()

    results_count = 0
    last_log_time = receive_start_time

    while len(recv_results) < total_nr_requests:
        recv_tokens, idx = batch_recv_queue.get()
        recv_results.append((tokenizer.decode(recv_tokens), idx))

        results_count += 1
        current_time = time.time()

        # Log progress every 5 seconds
        if current_time - last_log_time > 5:
            elapsed = current_time - receive_start_time
            if elapsed > 0:
                logger.info(
                    f"Received {results_count}/{total_nr_requests} results ({results_count/elapsed:.2f} results/sec)"
                )
            last_log_time = current_time

    receive_end_time = time.time()
    receive_time = receive_end_time - receive_start_time

    # Measure sorting time
    sort_start_time = time.time()
    sorted_recv = sorted(recv_results, key=lambda x: x[1])
    sort_end_time = time.time()
    sort_time = sort_end_time - sort_start_time

    # Print detailed timing information
    logger.info(f"Batch processing details:")
    logger.info(
        f"  Sending {total_nr_requests} requests: {send_time:.2f}s ({total_nr_requests/send_time:.2f} req/s)"
    )
    logger.info(
        f"  Receiving {total_nr_requests} results: {receive_time:.2f}s ({total_nr_requests/receive_time:.2f} req/s)"
    )
    logger.info(f"  Sorting results: {sort_time:.2f}s")
    logger.info(f"  Total tokens: {sum_tokens} input tokens")

    return [elem[0] for elem in sorted_recv]


def create_tllm_request(*, token_ids, params, tokenizer, streaming,
                        return_all_generated_tokens):
    global llm_api, trtllm, SamplingParams
    llm_api_sampling_params = SamplingParams(temperature=1,
                                             max_tokens=params,
                                             exclude_input_from_output=True)
    llm_api_sampling_params._setup(tokenizer)

    request_obj = llm_api.GenerationRequest(
        prompt_token_ids=token_ids,
        sampling_params=llm_api_sampling_params,
        lora_request=None,
        streaming=streaming)

    executor_request = trtllm.Request(
        input_token_ids=request_obj.prompt_token_ids,
        max_tokens=request_obj.sampling_params.max_tokens,
        streaming=request_obj.streaming,
        sampling_config=request_obj.sampling_params._get_sampling_config(),
        end_id=request_obj.sampling_params.end_id,
        pad_id=request_obj.sampling_params.pad_id,
        output_config=request_obj.sampling_params._get_output_config(),
        bad_words=request_obj.sampling_params._get_bad_words(),
        stop_words=request_obj.sampling_params._get_stop_words(),
        embedding_bias=request_obj.sampling_params.embedding_bias,
        lora_config=None,
        return_all_generated_tokens=return_all_generated_tokens,
    )

    return executor_request


def batch_worker(
    wid,
    modelPath,
    supplier_queue: mp.Queue,
    resultQueue: mp.Queue,
    barrier,
    n_gpus,
):

    worker_logger = logging.getLogger(f'bWorker {wid:3d}:')
    worker_logger.warning(f"{wid} printing here")

    gpu_indices = list(range(wid * n_gpus, wid * n_gpus + n_gpus))
    gpu_string = ",".join(map(str, gpu_indices))

    worker_logger.warning(f"setting CUDA_VISIBLE_DEVICES to {gpu_string}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_string

    worker_logger.info(
        f'PID = {os.getpid()}, host={"localhost"}, port={8081}, model_path={modelPath}'
    )

    print("child process is at first barrier", flush=True)
    barrier.wait()

    # ================================================================================================================
    from transformers import AutoTokenizer
    global llm_api, trtllm, SamplingParams

    import tensorrt_llm.executor as llm_api
    from tensorrt_llm import SamplingParams
    from tensorrt_llm.bindings import executor as trtllm

    worker_logger.info("Loading model...")
    executor = create_executor(modelPath)
    worker_logger.info("Loaded model")

    tokenizer = AutoTokenizer.from_pretrained(modelPath)

    def collect_responses():
        resp_list = executor.await_responses()

        resp_token_idx_pairs = [(resp.result.output_token_ids[0],
                                 tllm_id_to_queue_id[resp.request_id])
                                for resp in resp_list]
        [tllm_id_to_queue_id.pop(resp.request_id) for resp in resp_list]

        for resp_pair in resp_token_idx_pairs:
            resultQueue.put(resp_pair)

    # ================================================================================================================
    print("child process is at second barrier ", flush=True)
    barrier.wait()  # Wait for LLM creation (to get accurate tput.)

    sample = 0
    tllm_id_to_queue_id = dict()
    while True:
        inp_tokens, params, rid = supplier_queue.get()
        if inp_tokens is None:
            break

        executor_request = create_tllm_request(
            token_ids=inp_tokens,
            params=params,
            tokenizer=tokenizer,
            streaming=False,
            return_all_generated_tokens=False,
        )

        # API call here
        tmp_req_id = executor.enqueue_request(executor_request)
        tllm_id_to_queue_id[tmp_req_id] = rid
        sample += 1

        if tmp_req_id % 100 == 0:
            collect_responses()
        print(f"worker {wid} has enqueued {tmp_req_id} requests", flush=True)

    worker_logger.info(f'Just waiting for results')

    while len(tllm_id_to_queue_id) > 0:
        collect_responses()

    # stats = holder.executor.get_latest_iteration_stats()
    # create_usage_graphs(stats, holder=holder)

    worker_logger.info(f'Exiting')


@click.command()
@click.option('--n_workers', type=int, default=2)
@click.option('--engine_path',
              type=str,
              default="./tmp/meta-llama/Llama-3.1-8B/tp_1_pp_1/")
@click.option('--trtllm_config', type=str, default=None)
@click.option('--n_gpus', type=int, default=1)
@click.option('--request_path',
              type=str,
              default="./tmp/synthetic_2048_128.txt")
def main(n_workers, engine_path, trtllm_config, n_gpus, request_path):
    with open(request_path, "r") as f:
        request_list = [json.loads(line.strip()) for line in f.readlines()]

    batch_send_queues = [mp.Queue() for i in range(n_workers)]
    batch_recv_queue = mp.Queue()
    nQueues = len(batch_send_queues)

    if nQueues > n_workers:
        raise NotImplementedError(
            f'There are more Supply Queues ({nQueues}) ' +
            f'than batch workers ({n_workers}). Not supported.')

    # mp.set_start_method('spawn', force=True)
    barrier = mp.Barrier(n_workers + 1)
    mp.Lock()

    mp.set_start_method('fork', force=True)
    workers = []
    print(f"Batch mode has {n_workers} workers")

    for wid in range(n_workers):
        kwargs = {
            'wid': wid,
            'modelPath': engine_path,
            'supplier_queue': batch_send_queues[wid % nQueues],
            'resultQueue': batch_recv_queue,
            'barrier': barrier,
            'n_gpus': n_gpus,
        }

        w = mp.Process(target=batch_worker, kwargs=kwargs)
        w.start()
        workers.append(w)

    logger.info('Waiting for bWorkers to start')
    print("parent process is at first barrier waiting for workers to start",
          flush=True)
    barrier.wait()
    print("parent process passed first barrier waiting for model load",
          flush=True)
    barrier.wait()

    logger.info('The bWorkers have all started, sending requests')

    # Start timing for measuring requests per second
    start_time = time.time()
    total_requests = len(request_list)
    print(f"Starting batch processing of {total_requests} requests...")

    batch(request_list=request_list,
          engine_path=engine_path,
          n_workers=n_workers,
          batch_send_queues=batch_send_queues,
          batch_recv_queue=batch_recv_queue,
          trtllm_config=trtllm_config)

    # End timing and calculate requests per second
    end_time = time.time()
    processing_time = end_time - start_time
    requests_per_second = total_requests / processing_time

    print(f"Batch processing completed:")
    print(f"Total requests: {total_requests}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Throughput: {requests_per_second:.2f} requests/second")


if __name__ == "__main__":
    main()
