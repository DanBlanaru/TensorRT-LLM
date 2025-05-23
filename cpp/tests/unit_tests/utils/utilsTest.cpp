
#include "common.h"
#include "tensorrt_llm/runtime/common.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>

struct RandomLogitsTestParameters
{
    using TupleT = std::tuple<int32_t, tensorrt_llm::runtime::SizeType32>;

    int32_t randomSeed;
    tensorrt_llm::runtime::SizeType32 vocabSize;

    // Constructor that takes a tuple
    RandomLogitsTestParameters( // NOLINT: implicit to allow gtest to convert from tuple generated by
                                // 'combine'
        TupleT t)
        : randomSeed(std::get<0>(t))
        , vocabSize(std::get<1>(t))
    {
    }
};

class RandomLogits : public ::testing::Test, public ::testing::WithParamInterface<RandomLogitsTestParameters>
{
protected:
    static constexpr int randomSeed = 2345;
};

namespace
{
constexpr int32_t kRandomSeed1 = 45;
constexpr int32_t kRandomSeed2 = 567;
auto const randomSeeds = ::testing::Values(kRandomSeed1, kRandomSeed2);

constexpr tensorrt_llm::runtime::SizeType32 kMinVocabSize = 16;
constexpr tensorrt_llm::runtime::SizeType32 kMaxVocabSize = 100000;
auto const vocabSizes = ::testing::Values(kMinVocabSize, kMaxVocabSize);

auto const paramGenerator
    = ::testing::ConvertGenerator<RandomLogitsTestParameters::TupleT>(::testing::Combine(randomSeeds, vocabSizes));
} // namespace

TEST_P(RandomLogits, FloatSumToOne)
{
    auto rng = std::mt19937(randomSeed);
    auto const randomLogits = tensorrt_llm::testing::randomLogits<std::mt19937, float>(456, &rng);
    auto const sum = std::reduce(randomLogits.begin(), randomLogits.end());
    ASSERT_DOUBLE_EQ(sum, 1.0);
}

INSTANTIATE_TEST_SUITE_P(Float, RandomLogits, paramGenerator);
