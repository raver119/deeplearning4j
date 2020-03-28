/*******************************************************************************
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
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <array/NDArray.h>
#include <legacy/NativeOps.h>
#include <fstream>
#include <graph/Graph.h>

using namespace sd;
using namespace sd::graph;

class GraphAnalysisTests : public testing::Test {
public:
    GraphAnalysisTests() {
        ///
    }
};

TEST_F(GraphAnalysisTests, basic_toposort_test_1) {
    Graph graph;

    // A
    graph.getVariableSpace()->putVariable(-1, 0, NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // B
    graph.getVariableSpace()->putVariable(-2, 0, NDArrayFactory::create<int>('c', {3}, {2, 2, 2}));

    // C
    graph.getVariableSpace()->putVariable(-3, 0, NDArrayFactory::create<int>('c', {3}, {3, 3, 3}));

    Node a("multiply", 10, {{-1, 0}, {-2, 0}});
    Node b("add", 20, {{10, 0}, {-3, 0}});

    graph.addNode(b);
    graph.addNode(a);

    // we just check that nodes were really added
    ASSERT_EQ(2, graph.totalNodes());

    auto optimized = graph.optimizedGraph();

    // we expect that OptimizedGraph has exactly 1 layer
    ASSERT_EQ(1, optimized.layers());

    auto layer = optimized.layer(0);

    // we expect layer has exactly 1 OpSequence
    ASSERT_EQ(1, layer.size());
    auto sequence = layer[0];

    // we expect that OpSequence has exactly 2 ops
    ASSERT_EQ(2, sequence.length());

    ASSERT_EQ(10, sequence.at(0).second->nodeId());
    ASSERT_EQ(20, sequence.at(1).second->nodeId());
}

TEST_F(GraphAnalysisTests, basic_toposort_test_2) {
    Graph graph;

    // A
    graph.getVariableSpace()->putVariable(-1, 0, NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // B
    graph.getVariableSpace()->putVariable(-2, 0, NDArrayFactory::create<int>('c', {3}, {2, 2, 2}));

    // C
    graph.getVariableSpace()->putVariable(-3, 0, NDArrayFactory::create<int>('c', {3}, {3, 3, 3}));

    // D
    graph.getVariableSpace()->putVariable(-4, 0, NDArrayFactory::create<int>('c', {3}, {4, 4, 4}));

    Node a("multiply", 10, {{-1, 0}, {-2, 0}});
    Node b("add", 20, {{10, 0}, {-3, 0}});
    Node c("subtract", 30, {{10, 0}, {-4, 0}});


    graph.addNode(b);
    graph.addNode(c);
    graph.addNode(a);

    // we just check that nodes were really added
    ASSERT_EQ(3, graph.totalNodes());

    auto optimized = graph.optimizedGraph();

    // we expect that OptimizedGraph has exactly 1 layer
    ASSERT_EQ(2, optimized.layers());

    // checking first layer first
    auto layer = optimized.layer(0);

    // we expect layer has exactly 1 OpSequence
    ASSERT_EQ(1, layer.size());
    auto sequence = layer[0];

    // we expect that OpSequence has exactly 2 ops
    ASSERT_EQ(1, sequence.length());

    ASSERT_EQ(10, sequence.at(0).second->nodeId());

}