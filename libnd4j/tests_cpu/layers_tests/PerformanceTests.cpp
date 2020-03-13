/*******************************************************************************
 * Copyright (c) 2019 Konduit
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <graph/Graph.h>
#include <chrono>
#include <graph/Node.h>
#include <ops/declarable/CustomOperations.h>
#include <graph/profiling/GraphProfilingHelper.h>
#include <loops/type_conversions.h>
#include <helpers/threshold.h>
#include <helpers/MmulHelper.h>
#include <ops/ops.h>
#include <helpers/OmpLaunchHelper.h>
#include <helpers/GradCheck.h>
#include <ops/declarable/helpers/im2col.h>
#include <helpers/Loops.h>
#include <helpers/RandomLauncher.h>

#include <helpers/BenchmarkHelper.h>
#include <ops/declarable/helpers/scatter.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <array>
#include <performance/benchmarking/FullBenchmarkSuit.h>
#include <performance/benchmarking/LightBenchmarkSuit.h>

#include <ops/declarable/helpers/legacy_helpers.h>
#include <execution/ThreadPool.h>

#include <ops/declarable/CustomOperations.h>
#include "performance/benchmarking/LightBenchmarkSuit.h"


using namespace sd;
using namespace sd::graph;

#define WARMUP 5
#define NUM_ITER 100


template <typename T>
static std::string lstmBenchmark() {
    std::string output;
    output += "lstm " + DataTypeUtils::asString(DataTypeUtils::fromT<T>());
    BenchmarkHelper helper(WARMUP, NUM_ITER);

    BoolParameters format("format");    //0=TNS=[seqLen,mb,size]; 1=NST=[mb,size,seqLen]
    PredefinedParameters mb("mb", {1, 8});
    int n = 128;

    ParametersBatch batch({&format, &mb});
    sd::ops::lstmBlock lstmBlock;
    DeclarableBenchmark benchmark(lstmBlock, "lstm");

    int seqLength = 8;

    auto generator = PARAMETRIC_D() {
        auto ctx = new Context(1);
        int f = p.getIntParam("format");
        int m = p.getIntParam("mb");

        Nd4jLong l = 0;
        ctx->setInputArray(0, NDArrayFactory::create_<Nd4jLong>(l), true);  //Max TS length (unused)


        if (f == 0) {
            //TNS format
            ctx->setInputArray(1, NDArrayFactory::create_<T>('c', {seqLength, m, n}), true);     //x
            ctx->setOutputArray(0, NDArrayFactory::create_<T>('c', {seqLength, m, n}), true);    //i
            ctx->setOutputArray(1, NDArrayFactory::create_<T>('c', {seqLength, m, n}), true);    //c
            ctx->setOutputArray(2, NDArrayFactory::create_<T>('c', {seqLength, m, n}), true);    //f
            ctx->setOutputArray(3, NDArrayFactory::create_<T>('c', {seqLength, m, n}), true);    //o
            ctx->setOutputArray(4, NDArrayFactory::create_<T>('c', {seqLength, m, n}), true);    //z
            ctx->setOutputArray(5, NDArrayFactory::create_<T>('c', {seqLength, m, n}), true);    //h
            ctx->setOutputArray(6, NDArrayFactory::create_<T>('c', {seqLength, m, n}), true);    //y
        } else {
            //NST format
            ctx->setInputArray(1, NDArrayFactory::create_<T>('f', {m, n, seqLength}), true);     //x
            ctx->setOutputArray(0, NDArrayFactory::create_<T>('f', {m, n, seqLength}), true);    //i
            ctx->setOutputArray(1, NDArrayFactory::create_<T>('f', {m, n, seqLength}), true);    //c
            ctx->setOutputArray(2, NDArrayFactory::create_<T>('f', {m, n, seqLength}), true);    //f
            ctx->setOutputArray(3, NDArrayFactory::create_<T>('f', {m, n, seqLength}), true);    //o
            ctx->setOutputArray(4, NDArrayFactory::create_<T>('f', {m, n, seqLength}), true);    //z
            ctx->setOutputArray(5, NDArrayFactory::create_<T>('f', {m, n, seqLength}), true);    //h
            ctx->setOutputArray(6, NDArrayFactory::create_<T>('f', {m, n, seqLength}), true);    //y
        }

        auto cLast = NDArrayFactory::create_<T>('c', {m, n});
        auto yLast = NDArrayFactory::create_<T>('c', {m, n});
        auto W = NDArrayFactory::create_<T>('c', {2 * n, 4 * n});
        auto Wci = NDArrayFactory::create_<T>('c', {n});
        auto Wcf = NDArrayFactory::create_<T>('c', {n});
        auto Wco = NDArrayFactory::create_<T>('c', {n});
        auto b = NDArrayFactory::create_<T>('c', {4 * n});

        ctx->setInputArray(2, cLast, true);
        ctx->setInputArray(3, yLast, true);
        ctx->setInputArray(4, W, true);
        ctx->setInputArray(5, Wci, true);
        ctx->setInputArray(6, Wcf, true);
        ctx->setInputArray(7, Wco, true);
        ctx->setInputArray(8, b, true);

        auto iargs = new Nd4jLong[2];
        iargs[0] = 0;   //No peephole
        iargs[1] = f;
        ctx->setIArguments(iargs, 2);
        delete[] iargs;

        auto targs = new double[2];
        targs[0] = 1.0; //forget bias
        targs[1] = 0.0; //cell clipping value
        ctx->setTArguments(targs, 2);
        delete[] targs;
        return ctx;
    };

    output += helper.runOperationSuit(&benchmark, generator, batch, "LSTMBlock");
    return output;
}


class PerformanceTests : public testing::Test {
public:
    int numIterations = 100;

    PerformanceTests() {
        samediff::ThreadPool::getInstance();
    }
};

TEST_F(PerformanceTests, benchmarksLight) {
    LightBenchmarkSuit bench;
    printf("%s\n", bench.runSuit().c_str());
}

TEST_F(PerformanceTests, benchmarksFull) {
    FullBenchmarkSuit bench;
    printf("%s\n", bench.runSuit().c_str());
}

TEST_F(PerformanceTests, benchmarksLSTM) {
    std::vector<sd::DataType> dtypes({sd::DataType::FLOAT32});

    std::string result;

    for (auto t:dtypes) {
        nd4j_printf("Running LightBenchmarkSuite.lstmBenchmark [%s]\n", DataTypeUtils::asString(t).c_str());
        BUILD_SINGLE_SELECTOR(t, result += lstmBenchmark, (), LIBND4J_TYPES);
    }
    printf("%s\n", result.c_str());
}



#ifdef RELEASE_BUILD

TEST_F(PerformanceTests, test_maxpooling2d_1) {
    std::vector<Nd4jLong> valuesX;
    // auto x = NDArrayFactory::create<float>('c', {32, 3, 224, 224});
    // auto z = NDArrayFactory::create<float>('c', {32, 3, 224, 224});
    auto x = NDArrayFactory::create<float>('c', {8, 3, 64, 64});
    auto z = NDArrayFactory::create<float>('c', {8, 3, 64, 64});
    x.linspace(1.0f);
    Nd4jLong k = 5;


    Nd4jLong iArgs[] {k,k, 1,1, 0,0, 1,1, 1};
    Context ctx(1);
    ctx.setInputArray(0, &x);
    ctx.setOutputArray(0, &z);
    ctx.setIArguments(iArgs, 9);

    sd::ops::maxpool2d op;

    for (int i = 0; i < numIterations; i++) {
        auto timeStart = std::chrono::system_clock::now();

        op.execute(&ctx);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
        valuesX.emplace_back(outerTime);

        if ((i + 1) % 1000 == 0)
            nd4j_printf("Iteration %i finished...\n", i + 1);
    }

    std::sort(valuesX.begin(), valuesX.end());
    nd4j_printf("Execution time: %lld; Min: %lld; Max: %lld;\n", valuesX[valuesX.size() / 2], valuesX[0], valuesX[valuesX.size() - 1]);
}

#endif
