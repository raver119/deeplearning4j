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

#include "testlayers.h"
#include <array/NDArray.h>
#include <ops/declarable/CustomOperations.h>

using namespace sd;
using namespace sd::ops;

class ListOperationsTests : public testing::Test {

};

TEST_F(ListOperationsTests, BasicTest_Write_1) {
    NDArrayList list(5);
    auto x = NDArrayFactory::create<double>('c', {128});
    x.linspace(1);

    sd::ops::write_list op;

    auto result = op.execute(&list, {&x}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    ASSERT_EQ(1, list.elements());

    auto result2 = op.execute(&list, {&x}, {}, {2});

    ASSERT_EQ(2, list.elements());

    
    
}

TEST_F(ListOperationsTests, BasicTest_Stack_1) {
    NDArrayList list(10);
    auto exp = NDArrayFactory::create<double>('c', {10, 100});
    auto tads = exp.allTensorsAlongDimension({1});
    for (int e = 0; e < 10; e++) {
        auto row = NDArrayFactory::create_<double>('c', {100});
        row->assign((double) e);
        list.write(e, row);
        tads.at(e).assign(row);
    }

    sd::ops::stack_list op;

    auto result = op.execute(&list, {}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);
    // z->printShapeInfo();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    
}

TEST_F(ListOperationsTests, BasicTest_UnStackList_1) {
    NDArrayList list(0, true);
    auto x = NDArrayFactory::create<double>('c', {10, 100});
    auto tads = x.allTensorsAlongDimension({1});
    for (int e = 0; e < 10; e++) {
        auto row = NDArrayFactory::create<double>('c', {100});
        row.assign((double) e);
        //list.write(e, row);
        tads.at(e).assign(row);
    }

    sd::ops::unstack_list op;

    auto result = op.execute(&list, {&x}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result.status());
    ASSERT_EQ(list.elements(), 10);

//    auto z = result.at(0);
//    z->printShapeInfo("The first of");
//    ASSERT_TRUE(exp.isSameShape(z));
//    ASSERT_TRUE(exp.equalsTo(z));
    for (int e = 0; e < 10; e++) {
        auto row = list.read(e);
        ASSERT_TRUE(row->equalsTo(tads.at(e)));
        //list.write(e, row);
        delete row;
    }

    
}

//TEST_F(ListOperationsTests, BasicTest_UnStackList_2) {
////    NDArrayList list(0, true);
//    auto x = NDArrayFactory::create<double>('c', {10, 100});
//    auto tads = x.allTensorsAlongDimension({1});
//    for (int e = 0; e < 10; e++) {
//        auto row = NDArrayFactory::create_<double>('c', {100});
//        row->assign((double) e);
//        //list.write(e, row);
//        tads->at(e)->assign(row);
//        delete row;
//    }
//
//    sd::ops::unstack_list op;
//
//    auto result = op.execute(nullptr, {&x}, {}, {0});
//
//    ASSERT_EQ(ND4J_STATUS_OK, result.status());
//    ASSERT_EQ(result->size(), 10);
//
//    //    auto z = result.at(0);
////    z->printShapeInfo("The first of");
////    ASSERT_TRUE(exp.isSameShape(z));
////    ASSERT_TRUE(exp.equalsTo(z));
//    for (int e = 0; e < 10; e++) {
//        auto row = result.at(e);
//        ASSERT_TRUE(row->equalsTo(tads->at(e)));
//        //list.write(e, row);
//    }
//
//    
//    delete tads;
//}

TEST_F(ListOperationsTests, BasicTest_Read_1) {
    NDArrayList list(10);
    auto exp = NDArrayFactory::create<double>('c', {1, 100});
    exp.assign(4.0f);

    for (int e = 0; e < 10; e++) {
        auto row = NDArrayFactory::create_<double>('c', {1, 100});
        row->assign((double) e);
        list.write(e, new NDArray(row->dup()));

        delete row;
    }

    sd::ops::read_list op;

    auto result = op.execute(&list, {}, {}, {4});

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    
}

TEST_F(ListOperationsTests, BasicTest_Pick_1) {
    NDArrayList list(10);
    auto exp = NDArrayFactory::create<double>('c', {4, 100});

    for (int e = 0; e < 10; e++) {
        auto row = NDArrayFactory::create_<double>('c', {100});
        row->assign((double) e);
        list.write(e, new NDArray(row->dup()));

        delete row;
    }

    auto tads = exp.allTensorsAlongDimension({1});
    tads.at(0).assign(1.0f);
    tads.at(1).assign(1.0f);
    tads.at(2).assign(3.0f);
    tads.at(3).assign(3.0f);


    sd::ops::pick_list op;
    auto result = op.execute(&list, {}, {}, {1, 1, 3, 3});

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    
}

TEST_F(ListOperationsTests, BasicTest_Size_1) {
    NDArrayList list(10);
    auto exp = NDArrayFactory::create<int>(10);
    for (int e = 0; e < 10; e++) {
        auto row = NDArrayFactory::create_<double>('c', {100});
        row->assign((double) e);
        list.write(e, new NDArray(row->dup()));

        delete row;
    }

    sd::ops::size_list op;

    auto result = op.execute(&list, {}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    
}

TEST_F(ListOperationsTests, BasicTest_Create_1) {
    auto matrix = NDArrayFactory::create<double>('c', {3, 2});
    matrix.linspace(1);

    sd::ops::create_list op;

    auto result = op.execute(nullptr, {&matrix}, {}, {1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    // we return flow as well
    ASSERT_EQ(1, result.size());

    
}

TEST_F(ListOperationsTests, BasicTest_Split_1) {
    NDArrayList list(0, true);

    auto exp0 = NDArrayFactory::create<double>('c', {2, 5});
    auto exp1 = NDArrayFactory::create<double>('c', {3, 5});
    auto exp2 = NDArrayFactory::create<double>('c', {5, 5});

    auto matrix = NDArrayFactory::create<double>('c', {10, 5});

    auto lengths = NDArrayFactory::create<int>('c', {3});
    lengths.p(0, 2);
    lengths.p(1, 3);
    lengths.p(2, 5);

    auto tads = matrix.allTensorsAlongDimension({1});

    auto tads0 = exp0.allTensorsAlongDimension({1});
    auto tads1 = exp1.allTensorsAlongDimension({1});
    auto tads2 = exp2.allTensorsAlongDimension({1});

    int cnt0 = 0;
    int cnt1 = 0;
    int cnt2 = 0;
    for (int e = 0; e < 10; e++) {
        auto row = NDArrayFactory::create_<double>('c', {5});
        row->assign((double) e);
        tads.at(e).assign(row);

        if (e < 2)
            tads0.at(cnt0++).assign(row);
        else if (e < 5)
            tads1.at(cnt1++).assign(row);
        else
            tads2.at(cnt2++).assign(row);

        delete row;
    }

    sd::ops::split_list op;
    auto result = op.execute(&list, {&matrix, &lengths}, {}, {});
    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_EQ(3, list.height());

    ASSERT_TRUE(exp0.isSameShape(list.readRaw(0)));
    ASSERT_TRUE(exp0.equalsTo(list.readRaw(0)));

    ASSERT_TRUE(exp1.isSameShape(list.readRaw(1)));
    ASSERT_TRUE(exp1.equalsTo(list.readRaw(1)));

    ASSERT_TRUE(exp2.isSameShape(list.readRaw(2)));
    ASSERT_TRUE(exp2.equalsTo(list.readRaw(2)));

    
}

TEST_F(ListOperationsTests, BasicTest_Scatter_1) {
    NDArrayList list(0, true);
    auto s = NDArrayFactory::create<double>(0.0);

    auto matrix = NDArrayFactory::create<double>('c', {10, 5});
    auto tads = matrix.allTensorsAlongDimension({1});
    for (int e = 0; e < 10; e++) {
        auto row = NDArrayFactory::create_<double>('c', {1, 5});
        row->assign((double) e);
        tads.at(e).assign(row);

        delete row;
    }
    auto indices = NDArrayFactory::create<double>('c', {1, 10});
    for (int e = 0; e < matrix.rows(); e++)
        indices.p(e, 9 - e);

    sd::ops::scatter_list op;
    auto result = op.execute(&list, {&indices, &matrix, &s}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    for (int e = 0; e < 10; e++) {
        auto row = tads.at(9 - e);
        auto chunk = list.readRaw(e);

        ASSERT_TRUE(chunk->isSameShape(row));

        ASSERT_TRUE(chunk->equalsTo(row));
    }
    
}

TEST_F(ListOperationsTests, BasicTest_Clone_1) {
    auto list = new NDArrayList(0, true);

    VariableSpace variableSpace;
    auto var = new Variable();
    //var->setNDArrayList(list);

    //variableSpace.putVariable(-1, var);
    //variableSpace.trackList(list);

    Context block(1, &variableSpace);
    block.pickInput(-1);

    sd::ops::clone_list op;

    ASSERT_TRUE(list == block.variable(0)->getNDArrayList().get());

    auto result = op.execute(&block);

    ASSERT_EQ(ND4J_STATUS_OK, result);

    auto resVar = variableSpace.getVariable(1);

    auto resList = resVar->getNDArrayList();

    ASSERT_TRUE( resList != nullptr);

    ASSERT_TRUE(list->equals(*resList));
}

TEST_F(ListOperationsTests, BasicTest_Gather_1) {
    NDArrayList list(0, true);
    for (int e = 0; e < 10; e++) {
        auto row = NDArrayFactory::create_<double>('c', {3});
        row->assign((double) e);
        list.write(e, new NDArray(row->dup()));

        delete row;
    }

    auto exp = NDArrayFactory::create<double>('c', {10, 3});
    auto tads = exp.allTensorsAlongDimension({1});
    for (int e = 0; e < 10; e++) {
        auto tad = tads.at(9 - e);
        tad.assign(e);
    }

    auto indices = NDArrayFactory::create<double>('c', {1, 10});
    indices.linspace(9, -1);

    sd::ops::gather_list op;
    auto result = op.execute(&list, {&indices}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result.status());
    ASSERT_EQ(1, result.size());

    auto z = result.at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    //exp.printIndexedBuffer("e");
    //z->printIndexedBuffer("z");

    ASSERT_TRUE(exp.equalsTo(z));

    
}