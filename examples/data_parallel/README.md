# Data Parallel Example

This example demonstrates how to use data parallelism in TensorRT-LLM to process multiple requests in parallel across multiple GPUs. It implements a batch worker approach where each worker handles a subset of requests on dedicated GPUs.

## Overview

The example implements:
- A multi-process architecture with worker processes each assigned to specific GPUs
- Request batching and load distribution across workers
- Result collection and reassembly
- Performance metrics for throughput measurement

## Requirements

- TensorRT-LLM installed
- A TensorRT-LLM engine (built with `trtllm-build`)
- A tokenizer compatible with the model
- Input requests in the expected JSON format

## Usage

### Preparing Test Data and Model

First, generate synthetic test data and build the TensorRT-LLM engine:

```bash
# Generate synthetic test data
python benchmarks/cpp/prepare_dataset.py --stdout --tokenizer=meta-llama/Llama-3.1-8B token-norm-dist \
  --num-requests=3000 --input-mean=2048 --output-mean=128 --input-stdev=0 --output-stdev=0 > ./tmp/synthetic_2048_128.txt

# Build the TensorRT-LLM engine
trtllm-bench --workspace=./tmp --model meta-llama/Llama-3.1-8B build --dataset ./tmp/synthetic_2048_128.txt
```

### Running the Example

Run the data parallel example:

```bash
# Basic usage with default parameters
python examples/data_parallel/example.py

# Specifying custom parameters
python examples/data_parallel/example.py \
  --n_workers 4 \
  --engine_path ./tmp/meta-llama/Llama-3.1-8B/tp_1_pp_1/ \
  --n_gpus 2 \
  --request_path ./tmp/synthetic_2048_128.txt
```

### Options

- `n_workers`: Number of worker processes to create (default: 2)
- `engine_path`: Path to the TensorRT-LLM engine directory (default: "./tmp/meta-llama/Llama-3.1-8B/tp_1_pp_1/")
- `n_gpus`: Number of GPUs to use per worker (default: 1)
- `request_path`: Path to the file containing requests in JSON format (default: "./tmp/synthetic_2048_128.txt")

### Request Format

The request file should contain JSON objects, one per line, with the following structure:
```json
{
  "input_ids": [123, 456, 789, ...],  // Input token IDs
  "output_tokens": 128  // Number of tokens to generate
}
```

## Example Workflow

1. The main process creates worker processes, each assigned to specific GPUs
2. Each worker loads the model on its assigned GPUs
3. Requests are distributed among workers through queues
4. Workers process requests in parallel
5. Results are collected and assembled in the original request order
6. Performance metrics are reported

## Performance Considerations

This example reports:
- Request throughput (requests per second)
- Processing time breakdown (sending, receiving, sorting)
- Total number of tokens processed

## Limitations

- Each worker can only use its assigned GPUs
- The implementation uses process-based parallelism which may have higher overhead than thread-based approaches for some workloads
