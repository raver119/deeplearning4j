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
    graph.addVariable("A", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // B
    graph.addVariable("B", NDArrayFactory::create<int>('c', {3}, {2, 2, 2}));

    // C
    graph.addVariable("C", NDArrayFactory::create<int>('c', {3}, {3, 3, 3}));

    Node a(sd::ops::multiply(), "multiply");
    Node b(sd::ops::add(), "add");

    graph.addNode(a, {"A", "B"});
    graph.addNode(b, {"multiply", "C"});

    // we just check that nodes were really added
    ASSERT_EQ(2, graph.size());

    auto optimized = graph.optimizedGraph();

    // we expect that OptimizedGraph has exactly 1 layer
    ASSERT_EQ(1, optimized.layers());

    auto layer = optimized.layer(0);

    // we expect layer has exactly 1 OpSequence
    ASSERT_EQ(1, layer.width());
    auto sequence = layer[0];

    // we expect that OpSequence has exactly 2 ops
    ASSERT_EQ(2, sequence.length());

    ASSERT_EQ(4, sequence.at(0).protoContext().nodeId());
    ASSERT_EQ(5, sequence.at(1).protoContext().nodeId());
}

TEST_F(GraphAnalysisTests, basic_toposort_test_2) {
    Graph graph;

    // A
    graph.addVariable("A", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // B
    graph.addVariable("B", NDArrayFactory::create<int>('c', {3}, {2, 2, 2}));

    // C
    graph.addVariable("C", NDArrayFactory::create<int>('c', {3}, {3, 3, 3}));

    // D
    graph.addVariable("D", NDArrayFactory::create<int>('c', {3}, {4, 4, 4}));

    Node a(sd::ops::multiply(), "multiply");
    Node b(sd::ops::add(), "add");
    Node c(sd::ops::subtract(), "subtract");


    graph.addNode(a, {"A", "B"});
    graph.addNode(b, {"multiply", "C"});
    graph.addNode(c, {"multiply", "D"});

    // we just check that nodes were really added
    ASSERT_EQ(3, graph.size());

    auto optimized = graph.optimizedGraph();

    // we expect that OptimizedGraph has exactly 2 layers
    ASSERT_EQ(2, optimized.layers());

    // checking first layer first
    auto layer0 = optimized.layer(0);

    // we expect layer has exactly 1 OpSequence
    ASSERT_EQ(1, layer0.width());
    auto sequence = layer0[0];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());

    ASSERT_EQ(5, sequence.at(0).protoContext().nodeId());

    // checking second layer now
    auto layer1 = optimized.layer(1);

    // we expect layer has exactly 2 OpSequences
    ASSERT_EQ(2, layer1.width());

    sequence = layer1[0];

    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(6, sequence.at(0).protoContext().nodeId());

    sequence = layer1[1];

    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(7, sequence.at(0).protoContext().nodeId());
}

TEST_F(GraphAnalysisTests, basic_toposort_test_3) {
    Graph graph;

    // A
    graph.addVariable("A", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // B
    graph.addVariable("B", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // C
    graph.addVariable("C", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // D
    graph.addVariable("D", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    Node a(sd::ops::multiply(), "a");
    Node b(sd::ops::add(), "b");
    Node c(sd::ops::subtract(), "c");
    Node d(sd::ops::add(), "d");
    Node e(sd::ops::multiply(), "e");

    graph.addNode(a, {"A", "B"});
    graph.addNode(b, {"a", "C"});

    graph.addNode(c, {"b", "D"});
    graph.addNode(d, {"b", "D"});

    graph.addNode(e, {"c", "d"});

    // we just check that nodes were really added
    ASSERT_EQ(5, graph.size());

    auto optimized = graph.optimizedGraph();

    // we expect that OptimizedGraph has exactly 3 layer
    ASSERT_EQ(3, optimized.layers());

    // checking first layer first
    auto layer0 = optimized.layer(0);

    // we expect layer has exactly 1 OpSequence
    ASSERT_EQ(1, layer0.width());
    auto sequence = layer0[0];

    // we expect that OpSequence has exactly 2 ops
    ASSERT_EQ(2, sequence.length());

    ASSERT_EQ(5, sequence.at(0).protoContext().nodeId());
    ASSERT_EQ(6, sequence.at(1).protoContext().nodeId());

    // checking second layer now
    auto layer1 = optimized.layer(1);

    // we expect layer has exactly 2 OpSequences
    ASSERT_EQ(2, layer1.width());

    sequence = layer1[0];

    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(7, sequence.at(0).protoContext().nodeId());

    sequence = layer1[1];

    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(8, sequence.at(0).protoContext().nodeId());


    // checking last layer
    auto layer2 = optimized.layer(2);

    // we expect layer has exactly 1 OpSequence
    ASSERT_EQ(1, layer2.width());
    sequence = layer2[0];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());

    ASSERT_EQ(9, sequence.at(0).protoContext().nodeId());
}

TEST_F(GraphAnalysisTests, basic_toposort_test_4) {
    Graph graph;

    // A
    graph.addVariable("A", NDArrayFactory::create<int>('c', { 3 }, { 1, 1, 1 }));

    // B
    graph.addVariable("B", NDArrayFactory::create<int>('c', { 3 }, { 1, 1, 1 }));

    // C
    graph.addVariable("C", NDArrayFactory::create<int>('c', { 3 }, { 1, 1, 1 }));

    // D
    graph.addVariable("D", NDArrayFactory::create<int>('c', { 3 }, { 1, 1, 1 }));

    // E
    graph.addVariable("E", NDArrayFactory::create<int>('c', { 3 }, { 1, 1, 1 }));

    // F
    graph.addVariable("F", NDArrayFactory::create<int>('c', { 3 }, { 1, 1, 1 }));


    Node a1(sd::ops::multiply(), "a1");
    Node a2(sd::ops::add(), "a2");

    Node b1(sd::ops::subtract(), "b1");
    Node b2(sd::ops::add(), "b2");
    Node b3(sd::ops::multiply(), "b3");

    Node d1(sd::ops::multiply(), "d1");
    Node d2(sd::ops::add(), "d2");

    Node e(sd::ops::subtract(), "e");

    graph.addNode(a1, { "A", "B" });
    graph.addNode(a2, { "C", "D" });

    graph.addNode(b1, { "a1", "E" });
    graph.addNode(b2, { "a1", "a2" });
    graph.addNode(b3, { "a2", "F" });

    graph.addNode(d1, { "b1", "b2" });
    graph.addNode(d2, { "b3", "b2" });

    graph.addNode(e, { "d1", "d2" });

    // we just check that nodes were really added
    ASSERT_EQ(8, graph.size());

    auto optimized = graph.optimizedGraph();

    // we expect that OptimizedGraph has exactly 4 layer
    ASSERT_EQ(4, optimized.layers());

    // checking first layer first
    auto layer0 = optimized.layer(0);

    // we expect layer has exactly 2 OpSequence
    ASSERT_EQ(2, layer0.width());
    auto sequence = layer0[0];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(7, sequence.at(0).protoContext().nodeId());
    sequence = layer0[1];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(8, sequence.at(0).protoContext().nodeId());

    // checking second layer now
    auto layer1 = optimized.layer(1);

    // we expect layer has exactly 3 OpSequences
    ASSERT_EQ(3, layer1.width());

    sequence = layer1[0];

    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(9, sequence.at(0).protoContext().nodeId());

    sequence = layer1[1];

    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(10, sequence.at(0).protoContext().nodeId());
    
    sequence = layer1[2];
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(11, sequence.at(0).protoContext().nodeId());

    auto layer2 = optimized.layer(2);

    // we expect layer has exactly 2 OpSequence
    ASSERT_EQ(2, layer2.width());
    sequence = layer2[0];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(12, sequence.at(0).protoContext().nodeId());

    sequence = layer2[1];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(13, sequence.at(0).protoContext().nodeId());

    // checking last layer
    auto layer3 = optimized.layer(3);

    // we expect layer has exactly 1 OpSequence
    ASSERT_EQ(1, layer3.width());
    sequence = layer3[0];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(14, sequence.at(0).protoContext().nodeId());
}

TEST_F(GraphAnalysisTests, basic_toposort_test_5) {
    Graph graph;

    // A
    graph.addVariable("A", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // B
    graph.addVariable("B", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // C
    graph.addVariable("C", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // D
    graph.addVariable("D", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    Node a(sd::ops::multiply(), "a");
    Node b(sd::ops::add(), "b");
    Node c(sd::ops::subtract(), "c");
    Node d(sd::ops::add(), "d");
    Node e(sd::ops::multiply(), "e");
    Node f(sd::ops::multiply(), "f");

    
    Node g(sd::ops::multiply(), "g");
    Node h(sd::ops::multiply(), "h");

    graph.addNode(a, {"A", "B"});
    graph.addNode(b, {"C", "D"});

    graph.addNode(c, {"a", "b"});
    graph.addNode(d, {"a", "b"});

    graph.addNode(e, {"c", "d"});
    graph.addNode(f, {"c", "d"});

    graph.addNode(g, {"c", "e"});
    graph.addNode(h, {"d", "f"});

    // we just check that nodes were really added
    ASSERT_EQ(8, graph.size());

    auto optimized = graph.optimizedGraph();

    // we expect that OptimizedGraph has exactly 3 layer
    ASSERT_EQ(4, optimized.layers());

    // checking first layer first
    auto layer0 = optimized.layer(0);

    // we expect layer has exactly 2 OpSequence
    ASSERT_EQ(2, layer0.width());
    auto sequence = layer0[0];

    // we expect that OpSequence has exactly 2 ops
    ASSERT_EQ(1, sequence.length());

    ASSERT_EQ(5, sequence.at(0).protoContext().nodeId());
    
    sequence = layer0[1];

    // we expect that OpSequence has exactly 2 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(6, sequence.at(0).protoContext().nodeId());

    // checking second layer now
    auto layer1 = optimized.layer(1);

    // we expect layer has exactly 2 OpSequences
    ASSERT_EQ(2, layer1.width());

    sequence = layer1[0];

    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(7, sequence.at(0).protoContext().nodeId());
    
    sequence = layer1[1];

    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(8, sequence.at(0).protoContext().nodeId());

    // checking before last layer
    auto layer2 = optimized.layer(2);

    // we expect layer has exactly 2 OpSequence
    ASSERT_EQ(2, layer2.width());
    sequence = layer2[0];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(9, sequence.at(0).protoContext().nodeId());
    sequence = layer2[1];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(10, sequence.at(0).protoContext().nodeId());

    // checking last layer
    auto layer3 = optimized.layer(3);

    // we expect layer has exactly 2 OpSequence
    ASSERT_EQ(2, layer3.width());
    sequence = layer3[0];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(11, sequence.at(0).protoContext().nodeId());
    
    sequence = layer3[1];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(12, sequence.at(0).protoContext().nodeId());
}

TEST_F(GraphAnalysisTests, basic_toposort_test_6) {
    Graph graph;

    // A
    graph.addVariable("A", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // B
    graph.addVariable("B", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // C
    graph.addVariable("C", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // D
    graph.addVariable("D", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // E
    graph.addVariable("E", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // F
    graph.addVariable("F", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    Node a(sd::ops::multiply(), "a");
    Node b1(sd::ops::add(), "b1");
    Node b2(sd::ops::subtract(), "b2");
    
    Node c1(sd::ops::add(), "c1");
    Node c2(sd::ops::multiply(), "c2");
    Node c3(sd::ops::subtract(), "c3");
    
    Node d1(sd::ops::multiply(), "d1");
    Node d2(sd::ops::multiply(), "d2");

    Node e(sd::ops::add(), "e");

    graph.addNode(a, {"A", "B"});

    graph.addNode(b1, {"a", "C"});
    graph.addNode(b2, {"a", "D"});

    graph.addNode(c1, {"b1", "E"});
    graph.addNode(c2, {"b1", "b2"});
    graph.addNode(c3, {"b2", "F"});

    graph.addNode(d1, {"c1", "c2"});
    graph.addNode(d2, {"c2", "c3"});

    graph.addNode(e, {"d1", "d2"});

    // we just check that nodes were really added
    ASSERT_EQ(9, graph.size());

    auto optimized = graph.optimizedGraph();

    // we expect that OptimizedGraph has exactly 3 layer
    ASSERT_EQ(5, optimized.layers());

    // checking first layer first
    auto layer0 = optimized.layer(0);

    // we expect layer has exactly 1 OpSequence
    ASSERT_EQ(1, layer0.width());
    auto sequence = layer0[0];
    // we expect that OpSequence has exactly 2 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(7, sequence.at(0).protoContext().nodeId());

    // checking second layer now
    auto layer1 = optimized.layer(1);

    // we expect layer has exactly 2 OpSequences
    ASSERT_EQ(2, layer1.width());

    sequence = layer1[0];

    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(8, sequence.at(0).protoContext().nodeId());
    
    sequence = layer1[1];

    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(9, sequence.at(0).protoContext().nodeId());

    // checking midle layer
    auto layer2 = optimized.layer(2);

    // we expect layer has exactly 2 OpSequence
    ASSERT_EQ(3, layer2.width());
    sequence = layer2[0];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(10, sequence.at(0).protoContext().nodeId());

    sequence = layer2[1];
    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(11, sequence.at(0).protoContext().nodeId());

    sequence = layer2[2];
    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(12, sequence.at(0).protoContext().nodeId());

    // checking before last layer
    auto layer3 = optimized.layer(3);

    // we expect layer has exactly 2 OpSequence
    ASSERT_EQ(2, layer3.width());
    sequence = layer3[0];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(13, sequence.at(0).protoContext().nodeId());
    
    sequence = layer3[1];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(14, sequence.at(0).protoContext().nodeId());
    
    // checking last layer
    auto layer4 = optimized.layer(4);

    // we expect layer has exactly 2 OpSequence
    ASSERT_EQ(1, layer4.width());
    sequence = layer4[0];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(15, sequence.at(0).protoContext().nodeId());

}

TEST_F(GraphAnalysisTests, basic_toposort_test_7) {
    Graph graph;

    // A
    graph.addVariable("A", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // B
    graph.addVariable("B", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // C
    graph.addVariable("C", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    Node a(sd::ops::multiply(), "a");
    Node b(sd::ops::add(), "b");
    Node c(sd::ops::subtract(), "c");
    Node d(sd::ops::add(), "d");
    Node e(sd::ops::multiply(), "e");

    graph.addNode(a, {"A", "B"});
    graph.addNode(b, {"a", "C"});

    graph.addNode(c, {"a", "b"});
    graph.addNode(d, {"b", "c"});

    graph.addNode(e, {"b", "c", "d"});

    // we just check that nodes were really added
    ASSERT_EQ(5, graph.size());

    auto optimized = graph.optimizedGraph();

    // we expect that OptimizedGraph has exactly 3 layer
    ASSERT_EQ(5, optimized.layers());

    // checking first layer first
    auto layer0 = optimized.layer(0);

    // we expect layer has exactly 1 OpSequence
    ASSERT_EQ(1, layer0.width());
    auto sequence = layer0[0];

    // we expect that OpSequence has exactly 2 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(4, sequence.at(0).protoContext().nodeId());

    // checking second layer now
    auto layer1 = optimized.layer(1);

    // we expect layer has exactly 2 OpSequences
    ASSERT_EQ(1, layer1.width());

    sequence = layer1[0];

    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(5, sequence.at(0).protoContext().nodeId());

    // checking layer 2
    auto layer2 = optimized.layer(2);

    // we expect layer has exactly 1 OpSequence
    ASSERT_EQ(1, layer2.width());
    sequence = layer2[0];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(6, sequence.at(0).protoContext().nodeId());

    // checking layer 3
    auto layer3 = optimized.layer(3);

    // we expect layer has exactly 1 OpSequence
    ASSERT_EQ(1, layer3.width());
    sequence = layer3[0];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(7, sequence.at(0).protoContext().nodeId());

    // checking layer 3
    auto layer4 = optimized.layer(4);

    // we expect layer has exactly 1 OpSequence
    ASSERT_EQ(1, layer4.width());
    sequence = layer4[0];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(8, sequence.at(0).protoContext().nodeId());
}


TEST_F(GraphAnalysisTests, basic_toposort_test_8) {
    Graph graph;

    // A
    graph.addVariable("A", NDArrayFactory::create<int>('c', { 3 }, { 1, 1, 1 }));

    // B
    graph.addVariable("B", NDArrayFactory::create<int>('c', { 3 }, { 1, 1, 1 }));

    // C
    graph.addVariable("C", NDArrayFactory::create<int>('c', { 3 }, { 1, 1, 1 }));

    // D
    graph.addVariable("D", NDArrayFactory::create<int>('c', { 3 }, { 1, 1, 1 }));

    // E
    graph.addVariable("E", NDArrayFactory::create<int>('c', { 3 }, { 1, 1, 1 }));

    // F
    graph.addVariable("F", NDArrayFactory::create<int>('c', { 3 }, { 1, 1, 1 }));


    Node a1(sd::ops::multiply(), "a1");
    Node a2(sd::ops::add(), "a2");
    Node a3(sd::ops::add(), "a3");

    Node b1(sd::ops::subtract(), "b1");
    Node b2(sd::ops::add(), "b2");
    Node b3(sd::ops::multiply(), "b3");

    graph.addNode(a1, { "A", "B" });
    graph.addNode(a2, { "C", "D" });
    graph.addNode(a3, { "E", "F" });

    graph.addNode(b1, { "a1", "a2" });
    graph.addNode(b2, { "a1", "a2", "a3" });
    graph.addNode(b3, { "a2", "a3" });

    // we just check that nodes were really added
    ASSERT_EQ(6, graph.size());

    auto optimized = graph.optimizedGraph();

    // we expect that OptimizedGraph has exactly 2 layer
    ASSERT_EQ(2, optimized.layers());

    // checking first layer first
    auto layer0 = optimized.layer(0);

    // we expect layer has exactly 3 OpSequence
    ASSERT_EQ(3, layer0.width());
    auto sequence = layer0[0];

    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(7, sequence.at(0).protoContext().nodeId());

    sequence = layer0[1];
    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(8, sequence.at(0).protoContext().nodeId());

    sequence = layer0[2];
    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(9, sequence.at(0).protoContext().nodeId());

    // checking second layer now
    auto layer1 = optimized.layer(1);

    // we expect layer has exactly 3 OpSequences
    ASSERT_EQ(3, layer1.width());

    sequence = layer1[0];
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(10, sequence.at(0).protoContext().nodeId());

    sequence = layer1[1];

    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(11, sequence.at(0).protoContext().nodeId());
    
    sequence = layer1[2];
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(12, sequence.at(0).protoContext().nodeId());

}

TEST_F(GraphAnalysisTests, basic_toposort_test_9) {

    // start graph

    Graph graph;

    // A
    graph.addVariable("A", NDArrayFactory::create<int>('c', {3}, {1, 1, 1}));

    // B
    graph.addVariable("B", NDArrayFactory::create<int>('c', {3}, {2, 2, 2}));

    // C
    graph.addVariable("C", NDArrayFactory::create<int>('c', {3}, {3, 3, 3}));

    // D
    graph.addVariable("D", NDArrayFactory::create<int>('c', {3}, {3, 3, 3}));

    Node a(sd::ops::multiply(), "a");

    Node b1(sd::ops::add(), "b1");
    Node b2(sd::ops::multiply(), "b2");
    Node b3(sd::ops::subtract(), "b3");
    Node b4(sd::ops::Pow(), "b4");

    Node c1(sd::ops::Pow(), "c1");
    Node c2(sd::ops::subtract(), "c2");
    Node c3(sd::ops::multiply(), "c3");
    Node c4(sd::ops::add(), "c4");

    Node c5(sd::ops::Pow(), "c5");
    Node c6(sd::ops::subtract(), "c6");
    Node c7(sd::ops::multiply(), "c7");
    Node c8(sd::ops::add(), "c8");

    graph.addNode(a, {"A", "B"});

    graph.addNode(b1, {"a", "C"});
    graph.addNode(b2, {"a", "C"});
    graph.addNode(b3, {"a", "C"});
    graph.addNode(b4, {"a", "C"});

    graph.addNode(c1, {"b1", "D"});
    graph.addNode(c2, {"b2", "D"});
    graph.addNode(c3, {"b3", "D"});
    graph.addNode(c4, {"b4", "D"});
    
    graph.addNode(c5, {"b1", "D"});
    graph.addNode(c6, {"b2", "D"});
    graph.addNode(c7, {"b3", "D"});
    graph.addNode(c8, {"b4", "D"});

    // we just check that nodes were really added
    ASSERT_EQ(13, graph.size());

    auto optimized = graph.optimizedGraph();

    // we expect that OptimizedGraph has exactly 1 layer
    ASSERT_EQ(3, optimized.layers());

    auto layer = optimized.layer(0);
    // we expect layer has exactly 1 OpSequence
    ASSERT_EQ(1, layer.width());
    auto sequence = layer[0];

    // we expect that OpSequence has exactly 2 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(5, sequence.at(0).protoContext().nodeId());

    auto layer1 = optimized.layer(1);
    // we expect layer has exactly 4 OpSequence
    ASSERT_EQ(4, layer1.width());
    sequence = layer1[0];
    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(6, sequence.at(0).protoContext().nodeId());

    sequence = layer1[1];
    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(7, sequence.at(0).protoContext().nodeId());

    sequence = layer1[2];
    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(8, sequence.at(0).protoContext().nodeId());

    sequence = layer1[3];
    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(9, sequence.at(0).protoContext().nodeId());

    auto layer2 = optimized.layer(2);
    // we expect layer has exactly 4 OpSequence
    ASSERT_EQ(8, layer2.width());
    sequence = layer2[0];
    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(10, sequence.at(0).protoContext().nodeId());

    sequence = layer2[1];
    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(14, sequence.at(0).protoContext().nodeId());

    sequence = layer2[2];
    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(11, sequence.at(0).protoContext().nodeId());

    sequence = layer2[3];
    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(15, sequence.at(0).protoContext().nodeId());
    
    sequence = layer2[4];
    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(12, sequence.at(0).protoContext().nodeId());

    sequence = layer2[5];
    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(16, sequence.at(0).protoContext().nodeId());

    sequence = layer2[6];
    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(13, sequence.at(0).protoContext().nodeId());

    sequence = layer2[7];
    // we expect that OpSequence has exactly 1 ops
    ASSERT_EQ(1, sequence.length());
    ASSERT_EQ(17, sequence.at(0).protoContext().nodeId());
}
