/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

#ifndef LIBND4J_VARIABLETESTS_H
#define LIBND4J_VARIABLETESTS_H

#include "testlayers.h"
#include <array/NDArray.h>
#include <graph/Variable.h>
#include <flatbuffers/flatbuffers.h>

using namespace sd;
using namespace sd::graph;

class VariableTests : public testing::Test {
public:

};

TEST_F(VariableTests, Test_FlatVariableDataType_1) {
    flatbuffers::FlatBufferBuilder builder(1024);
    auto original = NDArrayFactory::create<float>('c', {5, 10});
    original.linspace(1);

    auto vec = original.asByteVector();

    auto fShape = builder.CreateVector(original.getShapeInfoAsFlatVector());
    auto fBuffer = builder.CreateVector(vec);
    auto fVid = CreateIntPair(builder, 1, 12);

    auto fArray = CreateFlatArray(builder, fShape, fBuffer, sd::graph::DType::DType_FLOAT);

    auto flatVar = CreateFlatVariable(builder, fVid, 0, sd::graph::DType::DType_FLOAT, 0, fArray);

    builder.Finish(flatVar);

    auto ptr = builder.GetBufferPointer();

    auto restoredVar = GetFlatVariable(ptr);

    auto rv = new Variable(restoredVar);

    ASSERT_EQ(1, rv->id());
    ASSERT_EQ(12, rv->index());

    auto restoredArray = rv->getNDArray();

    ASSERT_TRUE(original.isSameShape(*restoredArray));
    ASSERT_TRUE(original.equalsTo(*restoredArray));

    delete rv;
}

TEST_F(VariableTests, Test_FlatVariableDataType_2) {
    flatbuffers::FlatBufferBuilder builder(1024);
    auto original = NDArrayFactory::create<double>('c', {5, 10});
    original.linspace(1);

    auto vec = original.asByteVector();

    auto fShape = builder.CreateVector(original.getShapeInfoAsFlatVector());
    auto fBuffer = builder.CreateVector(vec);
    auto fVid = CreateIntPair(builder, 1, 12);

    auto fArray = CreateFlatArray(builder, fShape, fBuffer, sd::graph::DType::DType_DOUBLE);

    auto flatVar = CreateFlatVariable(builder, fVid, 0, sd::graph::DType::DType_DOUBLE, 0, fArray);

    builder.Finish(flatVar);

    auto ptr = builder.GetBufferPointer();

    auto restoredVar = GetFlatVariable(ptr);

    auto rv = new Variable(restoredVar);

    ASSERT_EQ(1, rv->id());
    ASSERT_EQ(12, rv->index());

    auto restoredArray = rv->getNDArray();

    ASSERT_TRUE(original.isSameShape(*restoredArray));
    ASSERT_TRUE(original.equalsTo(*restoredArray));

    delete rv;
}


TEST_F(VariableTests, Test_FlatVariableDataType_3) {
    flatbuffers::FlatBufferBuilder builder(1024);
    auto original = NDArrayFactory::create<double>('c', {5, 10});
    auto floating = NDArrayFactory::create<float>('c', {5, 10});
    original.linspace(1);
    floating.linspace(1);

    auto vec = original.asByteVector();

    auto fShape = builder.CreateVector(original.getShapeInfoAsFlatVector());
    auto fBuffer = builder.CreateVector(vec);
    auto fVid = CreateIntPair(builder, 1, 12);

    auto fArray = CreateFlatArray(builder, fShape, fBuffer, sd::graph::DType::DType_DOUBLE);

    auto flatVar = CreateFlatVariable(builder, fVid, 0, sd::graph::DType::DType_DOUBLE, 0, fArray);

    builder.Finish(flatVar);

    auto ptr = builder.GetBufferPointer();

    auto restoredVar = GetFlatVariable(ptr);

    auto rv = new Variable(restoredVar);

    ASSERT_EQ(1, rv->id());
    ASSERT_EQ(12, rv->index());

    auto restoredArray = rv->getNDArray();
    auto conv = restoredArray->asT<float>();

    ASSERT_TRUE(floating.isSameShape(*restoredArray));
    ASSERT_TRUE(floating.equalsTo(conv));

    delete rv;
}

/*
TEST_F(VariableTests, Test_FlatVariableDataType_4) {
    flatbuffers::FlatBufferBuilder builder(1024);
    auto original = NDArrayFactory::create<float>('c', {5, 10});
    std::vector<Nd4jLong> exp({5, 10});

    auto vec = original.asByteVector();

    auto fShape = builder.CreateVector(original.getShapeAsFlatVector());
    auto fVid = CreateIntPair(builder, 37, 12);

    auto flatVar = CreateFlatVariable(builder, fVid, 0, sd::graph::DType::DType_FLOAT, fShape, 0, 0, VarType_PLACEHOLDER);

    builder.Finish(flatVar);

    auto ptr = builder.GetBufferPointer();

    auto restoredVar = GetFlatVariable(ptr);

    auto rv = new Variable(restoredVar);

    ASSERT_EQ(37, rv->id());
    ASSERT_EQ(12, rv->index());

    //auto restoredArray = rv->getNDArray();
    ASSERT_EQ(PLACEHOLDER, rv->variableType());
    ASSERT_EQ(exp, rv->shape());

    //ASSERT_TRUE(original.isSameShape(restoredArray));
    //ASSERT_TRUE(original.equalsTo(restoredArray));

    delete rv;
}
*/
#endif //LIBND4J_VARIABLETESTS_H
