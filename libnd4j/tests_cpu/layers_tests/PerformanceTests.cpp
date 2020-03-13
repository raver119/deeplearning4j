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


TEST_F(PerformanceTests, subarray_perfs) {

    NDArray x('c', {2,3,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, DataType::FLOAT32);
    NDArray y('f', {2,3,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, DataType::FLOAT32);

    Nd4jLong shapeExpX0[] = {1, 2, 12, 8192, 12, 99};
    float    buffExpX0[]  = {1.000000, 13.000000};
    float    buffExpX1[]  = {2.000000, 14.000000};
    Nd4jLong shapeExpX2[] = {3, 2, 1, 1, 12, 4, 1, 8192, 12, 99};
    float    buffExpX2[]  = {1.000000, 13.000000};
    Nd4jLong shapeExpX3[] = {2, 2, 4, 12, 1, 8192, 0, 99};
    float    buffExpX3[]  = {9.000000, 10.000000, 11.000000, 12.000000, 21.000000, 22.000000, 23.000000, 24.000000};
    Nd4jLong shapeExpX4[] = {3, 2, 1, 4, 12, 4, 1, 8192, 0, 99};
    float    buffExpX4[]  = {9.000000, 10.000000, 11.000000, 12.000000, 21.000000, 22.000000, 23.000000, 24.000000};
    Nd4jLong shapeExpX5[] = {2, 2, 3, 12, 4, 8192, 4, 99};
    float    buffExpX5[]  = {4.000000, 8.000000, 12.000000, 16.000000, 20.000000, 24.000000};

    Nd4jLong shapeExpY0[] = {1, 2, 1, 8192, 1, 102};
    float    buffExpY0[]  = {1.000000, 2.000000};
    float    buffExpY1[]  = {7.000000, 8.000000};
    Nd4jLong shapeExpY2[] = {3, 2, 1, 1, 1, 2, 6, 8192, 1, 102};
    float    buffExpY2[]  = {1.000000, 2.000000};
    Nd4jLong shapeExpY3[] = {2, 2, 4, 1, 6, 8192, 0, 102};
    float    buffExpY3[]  = {5.000000, 11.000000, 17.000000, 23.000000, 6.000000, 12.000000, 18.000000, 24.000000};
    Nd4jLong shapeExpY4[] = {3, 2, 1, 4, 1, 2, 6, 8192, 0, 102};
    float    buffExpY4[]  = {5.000000, 11.000000, 17.000000, 23.000000, 6.000000, 12.000000, 18.000000, 24.000000};
    Nd4jLong shapeExpY5[] = {2, 2, 3, 1, 2, 8192, 1, 102};
    float    buffExpY5[]  = {19.000000, 21.000000, 23.000000, 20.000000, 22.000000, 24.000000};


    NDArray x0 = x(0, {1,2});
    for(int i = 0; i < shape::shapeInfoLength(x0.rankOf()); ++i)
        ASSERT_TRUE(x0.getShapeInfo()[i] == shapeExpX0[i]);
    for(int i = 0; i < x0.lengthOf(); ++i)
        ASSERT_TRUE(x0.e<float>(i) == buffExpX0[i]);

    NDArray x1 = x(1, {1,2});
    for(int i = 0; i < shape::shapeInfoLength(x1.rankOf()); ++i)
        ASSERT_TRUE(x1.getShapeInfo()[i] == shapeExpX0[i]);
    for(int i = 0; i < x1.lengthOf(); ++i)
        ASSERT_TRUE(x1.e<float>(i) == buffExpX1[i]);

    NDArray x2 = x(0, {1,2}, true);
    for(int i = 0; i < shape::shapeInfoLength(x2.rankOf()); ++i)
        ASSERT_TRUE(x2.getShapeInfo()[i] == shapeExpX2[i]);
    for(int i = 0; i < x2.lengthOf(); ++i)
        ASSERT_TRUE(x2.e<float>(i) == buffExpX2[i]);

    NDArray x3 = x(2, {1});
    for(int i = 0; i < shape::shapeInfoLength(x3.rankOf()); ++i)
        ASSERT_TRUE(x3.getShapeInfo()[i] == shapeExpX3[i]);
    for(int i = 0; i < x3.lengthOf(); ++i)
        ASSERT_TRUE(x3.e<float>(i) == buffExpX3[i]);

    NDArray x4 = x(2, {1}, true);
    for(int i = 0; i < shape::shapeInfoLength(x4.rankOf()); ++i)
        ASSERT_TRUE(x4.getShapeInfo()[i] == shapeExpX4[i]);
    for(int i = 0; i < x4.lengthOf(); ++i)
        ASSERT_TRUE(x4.e<float>(i) == buffExpX4[i]);

    NDArray x5 = x(3, {2});
    for(int i = 0; i < shape::shapeInfoLength(x5.rankOf()); ++i)
    {
        ASSERT_TRUE(x5.getShapeInfo()[i] == shapeExpX5[i]);
    }
    for(int i = 0; i < x5.lengthOf(); ++i)
        ASSERT_TRUE(x5.e<float>(i) == buffExpX5[i]);

    // ******************* //
    NDArray y0 = y(0, {1,2});
    for(int i = 0; i < shape::shapeInfoLength(y0.rankOf()); ++i)
        ASSERT_TRUE(y0.getShapeInfo()[i] == shapeExpY0[i]);
    for(int i = 0; i < y0.lengthOf(); ++i)
        ASSERT_TRUE(y0.e<float>(i) == buffExpY0[i]);

    NDArray y1 = y(1, {1,2});
    for(int i = 0; i < shape::shapeInfoLength(y1.rankOf()); ++i)
        ASSERT_TRUE(y1.getShapeInfo()[i] == shapeExpY0[i]);
    for(int i = 0; i < y1.lengthOf(); ++i)
        ASSERT_TRUE(y1.e<float>(i) == buffExpY1[i]);

    NDArray y2 = y(0, {1,2}, true);
    for(int i = 0; i < shape::shapeInfoLength(y2.rankOf()); ++i)
        ASSERT_TRUE(y2.getShapeInfo()[i] == shapeExpY2[i]);
    for(int i = 0; i < y2.lengthOf(); ++i)
        ASSERT_TRUE(y2.e<float>(i) == buffExpY2[i]);

    NDArray y3 = y(2, {1});
    for(int i = 0; i < shape::shapeInfoLength(y3.rankOf()); ++i)
        ASSERT_TRUE(y3.getShapeInfo()[i] == shapeExpY3[i]);
    for(int i = 0; i < y3.lengthOf(); ++i)
        ASSERT_TRUE(y3.e<float>(i) == buffExpY3[i]);

    NDArray y4 = y(2, {1}, true);
    for(int i = 0; i < shape::shapeInfoLength(y4.rankOf()); ++i)
        ASSERT_TRUE(y4.getShapeInfo()[i] == shapeExpY4[i]);
    for(int i = 0; i < y4.lengthOf(); ++i)
        ASSERT_TRUE(y4.e<float>(i) == buffExpY4[i]);

    NDArray y5 = y(3, {2});
    for(int i = 0; i < shape::shapeInfoLength(y5.rankOf()); ++i)
        ASSERT_TRUE(y5.getShapeInfo()[i] == shapeExpY5[i]);
    for(int i = 0; i < y5.lengthOf(); ++i)
        ASSERT_TRUE(y5.e<float>(i) == buffExpY5[i]);
}


TEST_F(PerformanceTests, concat_perfs) {
    auto x0 = NDArrayFactory::create<double>('c', {1,28});
    auto x1 = NDArrayFactory::create<double>('c', {1,128});

    x0.linspace(1);
    x1.linspace(1);

    sd::ops::concat op;

    auto result = op.evaluate({&x0, &x1}, {}, {1});
}


TEST_F(PerformanceTests, split_perfs) {
    auto input = NDArrayFactory::create<double>('c', {10},{1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f,9.f,10.f});
    auto axis = NDArrayFactory::create<double>(-1);
    auto exp1 = NDArrayFactory::create<double>('c', {5}, {1.f,2.f,3.f,4.f,5.f});
    auto exp2 = NDArrayFactory::create<double>('c', {5}, {6.f,7.f,8.f,9.f,10.f});

    sd::ops::split op;
    auto results = op.evaluate({&input, &axis}, {}, {2}, {});

}


TEST_F(PerformanceTests, stack_1d_perfs) {
    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1,2,3,4, 13, 14, 16, 16, 5,6,7,8, 17, 18, 19, 20, 9, 10, 11, 12, 21, 22, 23, 24};
    Nd4jLong shape1[]    = {2, 3, 4, 4, 1, 0, 1, 99};
    Nd4jLong shape2[]    = {2, 3, 4, 4, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 3, 2, 4, 8, 4, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, DataType::FLOAT32);
    ArrayOptions::setDataType(shape2, DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray input2(buff2, shape2);
    NDArray expected(expBuff, expShape);

    sd::ops::stack op;
    auto results = op.evaluate({&input1, &input2}, {}, {1});
}



TEST_F(PerformanceTests, stack_2d_perfs) {
    auto t = NDArrayFactory::create<float>('c', {1, 1}, {1.0f});
    auto u = NDArrayFactory::create<float>('c', {1, 1}, {2.0f});
    auto v = NDArrayFactory::create<float>('c', {1, 1}, {3.0f});
    auto w = NDArrayFactory::create<float>('c', {1, 1}, {4.0f});
    auto exp = NDArrayFactory::create<float>('c', {4, 1, 1}, {1, 2, 3, 4});

    sd::ops::stack op;
    auto result = op.evaluate({&t, &u, &v, &w}, {}, {0});
}


TEST_F(PerformanceTests, test_matmul_512_various) {

    for(int dim=512;dim<516;dim++){
        auto x1 = NDArrayFactory::create<float>('c', {dim, 512});
        auto y1 = NDArrayFactory::create<float>('c', {512, dim});
        auto z1 = NDArrayFactory::create<float>('c', {dim, dim});
        sd::ops::matmul op;
        auto timeStart = std::chrono::system_clock::now();
        op.execute({&x1, &y1}, {&z1}, {0, 0});
        auto timeEnd = std::chrono::system_clock::now();
        auto t1 = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
        nd4j_printf("Mat mul [%d,512]x[512,%d] = [%d,%d]: %lld ms\n", dim,dim,dim,dim,t1);
    }

}
