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

using namespace sd;
using namespace sd::graph;

class GraphExecutorTests : public testing::Test {
public:

};

TEST_F(GraphExecutorTests, test_execution_1) {
    Graph graph;

    // A
    graph.getVariableSpace()->putVariable(-1, 0, NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // B
    graph.getVariableSpace()->putVariable(-2, 0, NDArrayFactory::create<int>('c', {3}, {2, 2, 2}));

    // C
    graph.getVariableSpace()->putVariable(-3, 0, NDArrayFactory::create<int>('c', {3}, {3, 3, 3}));

    Node a("multiply", "multiply_node", 10, {{-1, 0}, {-2, 0}});
    Node b("add", "add_node", 20, {{10, 0}, {-3, 0}});

    graph.addNode(b);
    graph.addNode(a);

    auto result = graph.execute({}, {"add_node"});
    ASSERT_EQ(1, result.size());
    ASSERT_EQ(1, result.count("add_node"));
}

TEST_F(GraphExecutorTests, test_placeholder_resolution_1) {
    Graph graph;

    graph.addPlaceholder("input", 0, DataType::FLOAT32);

    graph.addNode(Node ("tanh", "tanh_node", 10, {{"input"}}));

    // this test must throw an exception, because input isn't resolved yet
    ASSERT_ANY_THROW(graph.execute());
}

TEST_F(GraphExecutorTests, test_placeholder_resolution_2) {
    Graph graph;

    graph.addPlaceholder("input", 0, DataType::FLOAT32);

    graph.addNode(Node ("tanh", "tanh_node", 10, {{"input"}}));

    auto result = graph.execute({{"input", NDArrayFactory::create(0.5f)}}, {"tanh_node"});

}

TEST_F(GraphExecutorTests, test_output_resolution_1) {
    Graph graph;

    graph.addPlaceholder("input", 0, DataType::FLOAT32);

    graph.addNode(Node ("tanh", "tanh_node", 10, {{"input"}}));

    // since we're requesting output of non-existent node - we expect exception
    ASSERT_THROW(graph.execute({{"input", NDArrayFactory::create(0.5f)}}, {"pow_node"}), graph::unresolved_output_exception);
}

TEST_F(GraphExecutorTests, test_input_resolution_1) {
    Graph graph;

    graph.addPlaceholder("input", 0, DataType::FLOAT32);

    graph.addNode(Node ("tanh", "tanh_node", 10, {{"input"}}));

    // since we're trying to resolve non-existent placeholder - we expect exception
    ASSERT_THROW(graph.execute({{"array", NDArrayFactory::create(0.5f)}}, {"tanh_node"}), graph::unresolved_input_exception);
}