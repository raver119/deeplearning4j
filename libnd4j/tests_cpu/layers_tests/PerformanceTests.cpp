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
#include <Graph.h>
#include <chrono>
#include <Node.h>
#include <ops/declarable/CustomOperations.h>
#include <graph/profiling/GraphProfilingHelper.h>
#include <type_conversions.h>
#include <helpers/threshold.h>
#include <helpers/MmulHelper.h>
#include <ops/ops.h>
#include <OmpLaunchHelper.h>
#include <GradCheck.h>
#include <ops/declarable/helpers/im2col.h>
#include <Loops.h>
#include <RandomLauncher.h>

#include <helpers/BenchmarkHelper.h>
#include <ops/declarable/helpers/scatter.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <array>
#include <performance/benchmarking/FullBenchmarkSuit.h>
#include <performance/benchmarking/LightBenchmarkSuit.h>

#include <ops/declarable/helpers/legacy_helpers.h>
#include <execution/ThreadPool.h>
#include <performance/benchmarking/global_timers.h>

using namespace nd4j;
using namespace nd4j::graph;

class PerformanceTests : public testing::Test {
public:
    int numIterations = 100;

    PerformanceTests() {
        samediff::ThreadPool::getInstance();
    }
};

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

    nd4j::ops::maxpool2d op;

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

    NDArray x('c', {2,3,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, nd4j::DataType::FLOAT32);
    NDArray y('f', {2,3,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, nd4j::DataType::FLOAT32);

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

    GlobalTimers::getInstance()->displayTimers();
}


TEST_F(PerformanceTests, concat_perfs) {
    auto timers = nd4j::GlobalTimers::getInstance();
    auto x0 = NDArrayFactory::create<double>('c', {1,28});
    auto x1 = NDArrayFactory::create<double>('c', {1,128});

    x0.linspace(1);
    x1.linspace(1);

    nd4j::ops::concat op;

    auto result = op.evaluate({&x0, &x1}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);
    delete result;

    //timers->displayTimers();
}


TEST_F(PerformanceTests, split_perfs) {
    auto timers = nd4j::GlobalTimers::getInstance();
    auto input = NDArrayFactory::create<double>('c', {10},{1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f,9.f,10.f});
    auto axis = NDArrayFactory::create<double>(-1);
    auto exp1 = NDArrayFactory::create<double>('c', {5}, {1.f,2.f,3.f,4.f,5.f});
    auto exp2 = NDArrayFactory::create<double>('c', {5}, {6.f,7.f,8.f,9.f,10.f});

    nd4j::ops::split op;
    auto results = op.evaluate({&input, &axis}, {}, {2}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto out1 = results->at(0);
    auto out2 = results->at(1);

    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp1.equalsTo(out1));
    ASSERT_TRUE(exp2.equalsTo(out2));

    delete results;

    //timers->displayTimers();
}


TEST_F(PerformanceTests, stack_1d_perfs) {
    auto timers = nd4j::GlobalTimers::getInstance();
    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1,2,3,4, 13, 14, 16, 16, 5,6,7,8, 17, 18, 19, 20, 9, 10, 11, 12, 21, 22, 23, 24};
    Nd4jLong shape1[]    = {2, 3, 4, 4, 1, 0, 1, 99};
    Nd4jLong shape2[]    = {2, 3, 4, 4, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 3, 2, 4, 8, 4, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(shape2, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, nd4j::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray input2(buff2, shape2);
    NDArray expected(expBuff, expShape);

    nd4j::ops::stack op;
    auto results = op.evaluate({&input1, &input2}, {}, {1});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
    //timers->displayTimers();
}



TEST_F(PerformanceTests, stack_2d_perfs) {
    auto timers = nd4j::GlobalTimers::getInstance();
    auto t = NDArrayFactory::create<float>('c', {1, 1}, {1.0f});
    auto u = NDArrayFactory::create<float>('c', {1, 1}, {2.0f});
    auto v = NDArrayFactory::create<float>('c', {1, 1}, {3.0f});
    auto w = NDArrayFactory::create<float>('c', {1, 1}, {4.0f});
    auto exp = NDArrayFactory::create<float>('c', {4, 1, 1}, {1, 2, 3, 4});

    nd4j::ops::stack op;
    auto result = op.evaluate({&t, &u, &v, &w}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    // z->printShapeInfo("z shape");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
    //timers->displayTimers();
}
