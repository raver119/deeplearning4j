/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2020 Konduit K.K.
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
// Created by raver119 on 29.11.17.
//


#include "testlayers.h"
#include <flatbuffers/flatbuffers.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/Node.h>
#include <graph/Graph.h>
#include <array/NDArray.h>
#include <ops/declarable/DeclarableOp.h>
#include <graph/exceptions/unresolved_output_exception.h>
#include <graph/exceptions/unresolved_input_exception.h>
#include <ops/declarable/headers/broadcastable.h>

using namespace sd;
using namespace sd::graph;

class GraphExecutorTests : public testing::Test {
public:

};

TEST_F(GraphExecutorTests, test_basic_exec_1) {
    GraphMemoryManager memoryManager;
    Graph graph;

    OptimizedGraph optimizedGraph;
    OpSequence sequence;

    optimizedGraph.append(sequence);

    GraphExecutor executor;
    executor.execute(optimizedGraph);
}

TEST_F(GraphExecutorTests, test_basic_exec_2) {
    GraphMemoryManager mgr;
    Graph graph(nullptr, mgr);

    auto A = NDArrayFactory::create<int>('c', {3}, {1, 1, 1});
    auto B = NDArrayFactory::create<int>('c', {3}, {2, 2, 2});
    auto C = NDArrayFactory::create<int>('c', {3}, {3, 3, 3});

    auto exp = NDArrayFactory::create<int>('c', {3}, {5, 5, 5});

    graph.addVariable("A", A);
    graph.addVariable("B", B);
    graph.addVariable("C", C);

    Node m(sd::ops::multiply(), "mul");
    Node a(sd::ops::add(), "add");

    graph.addNode(m, {"A", "B"});
    graph.addNode(a, {"mul", "C"});

    OptimizedGraph optimizedGraph;
    OpSequence sequence;

    ASSERT_EQ(2, m.protoContext().inputs().size());
    ASSERT_EQ(2, a.protoContext().inputs().size());

    sequence.append(m.customOp(), m.protoContext());
    sequence.append(a.customOp(), a.protoContext());

    optimizedGraph.append(sequence);

    ASSERT_EQ(2, sequence.length());
    ASSERT_EQ(1, optimizedGraph.layers());

    GraphExecutor executor;
    executor.execute(optimizedGraph);

    // checking results by ID
    ASSERT_TRUE(graph.variableSpace().hasVariable(m.id()));
    ASSERT_TRUE(graph.variableSpace().hasVariable(a.id()));

    // checking results by name
    ASSERT_TRUE(graph.variableSpace().hasVariable("mul"));
    ASSERT_TRUE(graph.variableSpace().hasVariable("add"));

    // checking if result is valid
    auto result = graph.variableSpace().getVariable(a.id())->getNDArray();
    ASSERT_EQ(exp, *result);
}