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

#include <array/NDArray.h>
#include <graph/Graph.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/CustomOperations.h>

#include <fstream>

#include "testlayers.h"

using namespace sd;
using namespace sd::graph;

class GraphAnalysisTests : public testing::Test {
 public:
  GraphAnalysisTests() {
    ///
  }
};

TEST_F(GraphAnalysisTests, optimizedGraph_1) {

  // A*B + C
  Graph graph;

  graph.addVariable("A", NDArray('c', {3}, sd::DataType::INT32));
  graph.addVariable("B", NDArray('c', {3}, sd::DataType::INT32));
  graph.addVariable("C", NDArray('c', {3}, sd::DataType::INT32));

  Node a(sd::ops::multiply(), "multiply");
  Node b(sd::ops::add(), "add");

  graph.addNode(a, {"A", "B"});
  graph.addNode(b, {"multiply", "C"});

  // we just check that nodes were really added
  ASSERT_EQ(2, graph.size());

  const auto& optimized = graph.optimizedGraph();

  // we expect that OptimizedGraph has exactly 1 layer
  ASSERT_EQ(1, optimized.numOfLayers());

  auto layer = optimized.layer(0);

  // we expect layer has exactly 1 OpSequence
  ASSERT_EQ(1, layer.width());
  auto sequence = layer[0];

  // we expect that OpSequence has exactly 2 ops
  ASSERT_EQ(2, sequence.length());

  ASSERT_EQ(4, sequence.at(0).protoContext().nodeId());
  ASSERT_EQ(5, sequence.at(1).protoContext().nodeId());
}

TEST_F(GraphAnalysisTests, optimizedGraph_2) {

  // 0 = A*B, 1_0 = 0+C, 1_1 = 0-D
  Graph graph;

  graph.addVariable("A", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("B", NDArray('c', {3}, {2, 2, 2}, sd::DataType::INT32));
  graph.addVariable("C", NDArray('c', {3}, {3, 3, 3}, sd::DataType::INT32));
  graph.addVariable("D", NDArray('c', {3}, {4, 4, 4}, sd::DataType::INT32));


  Node a(sd::ops::multiply(), "multiply");
  Node b(sd::ops::add(), "add");
  Node c(sd::ops::subtract(), "subtract");

  graph.addNode(a, {"A", "B"});
  graph.addNode(b, {"multiply", "C"});
  graph.addNode(c, {"multiply", "D"});

  // we just check that nodes were really added
  ASSERT_EQ(3, graph.size());

  const auto& optimized = graph.optimizedGraph();

  // graph size must stay the same
  ASSERT_EQ(3, graph.size());

  // we expect that OptimizedGraph has exactly 2 layers
  ASSERT_EQ(2, optimized.numOfLayers());

  // checking first layer first
  auto layer0 = optimized.layer(0);

  // we expect layer has exactly 1 OpSequence
  ASSERT_EQ(1, layer0.width());

  // we expect that OpSequence has exactly 1 node
  ASSERT_EQ(1, layer0[0].length());

  ASSERT_EQ(5, layer0[0].at(0).protoContext().nodeId());

  // checking second layer now
  auto layer1 = optimized.layer(1);

  // we expect layer has exactly 2 OpSequences
  ASSERT_EQ(2, layer1.width());

  ASSERT_EQ(1, layer1[0].length());
  ASSERT_EQ(6, layer1[0].at(0).protoContext().nodeId());

  ASSERT_EQ(1, layer1[1].length());
  ASSERT_EQ(7, layer1[1].at(0).protoContext().nodeId());
}

TEST_F(GraphAnalysisTests, optimizedGraph_3) {

  // 0 = A*B+C, 1_0 = 0-D, 1_1 = 0+D, 2 = 1_0*1_1
  Graph graph;

  graph.addVariable("A", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("B", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("C", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("D", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));

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

  const auto& optimized = graph.optimizedGraph();

  // we expect that OptimizedGraph has exactly 3 layer
  ASSERT_EQ(3, optimized.numOfLayers());

  // checking first layer first
  auto layer0 = optimized.layer(0);

  // we expect layer has exactly 1 OpSequence
  ASSERT_EQ(1, layer0.width());
  // auto sequence = layer0[0];

  // we expect that OpSequence has exactly 2 ops
  ASSERT_EQ(2, layer0[0].length());

  ASSERT_EQ(5, layer0[0].at(0).protoContext().nodeId());
  ASSERT_EQ(6, layer0[0].at(1).protoContext().nodeId());

  // checking second layer now
  const auto& layer1 = optimized.layer(1);

  // we expect layer has exactly 2 OpSequences
  ASSERT_EQ(2, layer1.width());

  // sequence = layer1[0];

  ASSERT_EQ(1, layer1[0].length());
  ASSERT_EQ(7, layer1[0].at(0).protoContext().nodeId());

  // sequence = layer1[1];

  ASSERT_EQ(1, layer1[1].length());
  ASSERT_EQ(8, layer1[1].at(0).protoContext().nodeId());

  // checking last layer
  auto layer2 = optimized.layer(2);

  // we expect layer has exactly 1 OpSequence
  ASSERT_EQ(1, layer2.width());
  // sequence = layer2[0];

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer2[0].length());
  ASSERT_EQ(9, layer2[0].at(0).protoContext().nodeId());
}

TEST_F(GraphAnalysisTests, optimizedGraph_4) {
  Graph graph;

  graph.addVariable("A", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("B", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("C", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("D", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("E", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("F", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));

  Node a1(sd::ops::multiply(), "a1");
  Node a2(sd::ops::add(), "a2");

  Node b1(sd::ops::subtract(), "b1");
  Node b2(sd::ops::add(), "b2");
  Node b3(sd::ops::multiply(), "b3");

  Node d1(sd::ops::multiply(), "d1");
  Node d2(sd::ops::add(), "d2");

  Node e(sd::ops::subtract(), "e");

  graph.addNode(a1, {"A", "B"});
  graph.addNode(a2, {"C", "D"});

  graph.addNode(b1, {"a1", "E"});
  graph.addNode(b2, {"a1", "a2"});
  graph.addNode(b3, {"a2", "F"});

  graph.addNode(d1, {"b1", "b2"});
  graph.addNode(d2, {"b3", "b2"});

  graph.addNode(e, {"d1", "d2"});

  // we just check that nodes were really added
  ASSERT_EQ(8, graph.size());

  const auto& optimized = graph.optimizedGraph();

  // we expect that OptimizedGraph has exactly 4 layer
  ASSERT_EQ(4, optimized.numOfLayers());

  // checking first layer first
  auto layer0 = optimized.layer(0);

  // we expect layer has exactly 2 OpSequence
  ASSERT_EQ(2, layer0.width());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer0[0].length());
  ASSERT_EQ(7, layer0[0].at(0).protoContext().nodeId());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer0[1].length());
  ASSERT_EQ(8, layer0[1].at(0).protoContext().nodeId());

  // checking second layer now
  auto layer1 = optimized.layer(1);

  // we expect layer has exactly 3 OpSequences
  ASSERT_EQ(3, layer1.width());

  ASSERT_EQ(1, layer1[0].length());
  ASSERT_EQ(9, layer1[0].at(0).protoContext().nodeId());

  ASSERT_EQ(1, layer1[1].length());
  ASSERT_EQ(10, layer1[1].at(0).protoContext().nodeId());

  ASSERT_EQ(1, layer1[2].length());
  ASSERT_EQ(11, layer1[2].at(0).protoContext().nodeId());

  auto layer2 = optimized.layer(2);

  // we expect layer has exactly 2 OpSequence
  ASSERT_EQ(2, layer2.width());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer2[0].length());
  ASSERT_EQ(12, layer2[0].at(0).protoContext().nodeId());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer2[1].length());
  ASSERT_EQ(13, layer2[1].at(0).protoContext().nodeId());

  // checking last layer
  auto layer3 = optimized.layer(3);

  // we expect layer has exactly 1 OpSequence
  ASSERT_EQ(1, layer3.width());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer3[0].length());
  ASSERT_EQ(14, layer3[0].at(0).protoContext().nodeId());
}

TEST_F(GraphAnalysisTests, optimizedGraph_5) {

  Graph graph;

  graph.addVariable("A", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("B", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("C", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("D", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));

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

  const auto& optimized = graph.optimizedGraph();

  // we expect that OptimizedGraph has exactly 3 layer
  ASSERT_EQ(4, optimized.numOfLayers());

  // checking first layer first
  auto layer0 = optimized.layer(0);

  // we expect layer has exactly 2 OpSequence
  ASSERT_EQ(2, layer0.width());
  // auto sequence = layer0[0];

  // we expect that OpSequence has exactly 2 ops
  ASSERT_EQ(1, layer0[0].length());

  ASSERT_EQ(5, layer0[0].at(0).protoContext().nodeId());

  // sequence = layer0[1];

  // we expect that OpSequence has exactly 2 ops
  ASSERT_EQ(1, layer0[1].length());
  ASSERT_EQ(6, layer0[1].at(0).protoContext().nodeId());

  // checking second layer now
  auto layer1 = optimized.layer(1);

  // we expect layer has exactly 2 OpSequences
  ASSERT_EQ(2, layer1.width());

  // sequence = layer1[0];

  ASSERT_EQ(1, layer1[0].length());
  ASSERT_EQ(7, layer1[0].at(0).protoContext().nodeId());

  // sequence = layer1[1];

  ASSERT_EQ(1, layer1[1].length());
  ASSERT_EQ(8, layer1[1].at(0).protoContext().nodeId());

  // checking before last layer
  auto layer2 = optimized.layer(2);

  // we expect layer has exactly 2 OpSequence
  ASSERT_EQ(2, layer2.width());
  // sequence = layer2[0];

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer2[0].length());
  ASSERT_EQ(9, layer2[0].at(0).protoContext().nodeId());
  // sequence = layer2[1];

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer2[1].length());
  ASSERT_EQ(10, layer2[1].at(0).protoContext().nodeId());

  // checking last layer
  auto layer3 = optimized.layer(3);

  // we expect layer has exactly 2 OpSequence
  ASSERT_EQ(2, layer3.width());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer3[0].length());
  ASSERT_EQ(11, layer3[0].at(0).protoContext().nodeId());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer3[1].length());
  ASSERT_EQ(12, layer3[1].at(0).protoContext().nodeId());
}

TEST_F(GraphAnalysisTests, optimizedGraph_6) {
  Graph graph;

  graph.addVariable("A", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("B", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("C", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("D", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("E", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("F", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));

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

  const auto& optimized = graph.optimizedGraph();

  // we expect that OptimizedGraph has exactly 3 layer
  ASSERT_EQ(5, optimized.numOfLayers());

  // checking first layer first
  auto layer0 = optimized.layer(0);

  // we expect layer has exactly 1 OpSequence
  ASSERT_EQ(1, layer0.width());

  // auto sequence = layer0[0];
  // we expect that OpSequence has exactly 2 ops
  ASSERT_EQ(1, layer0[0].length());
  ASSERT_EQ(7, layer0[0].at(0).protoContext().nodeId());

  // checking second layer now
  auto layer1 = optimized.layer(1);

  // we expect layer has exactly 2 OpSequences
  ASSERT_EQ(2, layer1.width());

  // sequence = layer1[0];
  ASSERT_EQ(1, layer1[0].length());
  ASSERT_EQ(8, layer1[0].at(0).protoContext().nodeId());

  // sequence = layer1[1];
  ASSERT_EQ(1, layer1[1].length());
  ASSERT_EQ(9, layer1[1].at(0).protoContext().nodeId());

  // checking midle layer
  auto layer2 = optimized.layer(2);

  // we expect layer has exactly 2 OpSequence
  ASSERT_EQ(3, layer2.width());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer2[0].length());
  ASSERT_EQ(10, layer2[0].at(0).protoContext().nodeId());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer2[1].length());
  ASSERT_EQ(11, layer2[1].at(0).protoContext().nodeId());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer2[2].length());
  ASSERT_EQ(12, layer2[2].at(0).protoContext().nodeId());

  // checking before last layer
  auto layer3 = optimized.layer(3);

  // we expect layer has exactly 2 OpSequence
  ASSERT_EQ(2, layer3.width());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer3[0].length());
  ASSERT_EQ(13, layer3[0].at(0).protoContext().nodeId());

  // sequence = layer3[1];
  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer3[1].length());
  ASSERT_EQ(14, layer3[1].at(0).protoContext().nodeId());

  // checking last layer
  auto layer4 = optimized.layer(4);

  // we expect layer has exactly 2 OpSequence
  ASSERT_EQ(1, layer4.width());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer4[0].length());
  ASSERT_EQ(15, layer4[0].at(0).protoContext().nodeId());
}

TEST_F(GraphAnalysisTests, optimizedGraph_7) {
  Graph graph;

  graph.addVariable("A", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("B", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("C", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));

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

  const auto& optimized = graph.optimizedGraph();
  // graph.printOut();
  // we expect that OptimizedGraph has exactly 3 layer
  ASSERT_EQ(1, optimized.numOfLayers());

  auto layer = optimized.layer(0);

  ASSERT_EQ(1, layer.width());

  auto seq = layer.at(0);
  ASSERT_EQ(5, seq.length());

  // this Graph doesn't allow any variance here. Order must be exactly the same as below
  ASSERT_EQ(std::string("a"), seq[0].node().name());
  ASSERT_EQ(std::string("b"), seq[1].node().name());
  ASSERT_EQ(std::string("c"), seq[2].node().name());
  ASSERT_EQ(std::string("d"), seq[3].node().name());
  ASSERT_EQ(std::string("e"), seq[4].node().name());
}

TEST_F(GraphAnalysisTests, optimizedGraph_8) {
  Graph graph;

  graph.addVariable("A", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("B", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("C", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("D", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("E", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("F", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));

  Node a1(sd::ops::multiply(), "a1");
  Node a2(sd::ops::add(), "a2");
  Node a3(sd::ops::add(), "a3");

  Node b1(sd::ops::subtract(), "b1");
  Node b2(sd::ops::add(), "b2");
  Node b3(sd::ops::multiply(), "b3");

  graph.addNode(a1, {"A", "B"});
  graph.addNode(a2, {"C", "D"});
  graph.addNode(a3, {"E", "F"});

  graph.addNode(b1, {"a1", "a2"});
  graph.addNode(b2, {"a1", "a2", "a3"});
  graph.addNode(b3, {"a2", "a3"});

  // we just check that nodes were really added
  ASSERT_EQ(6, graph.size());

  const auto& optimized = graph.optimizedGraph();

  // we expect that OptimizedGraph has exactly 2 layer
  ASSERT_EQ(2, optimized.numOfLayers());

  // checking first layer first
  auto layer0 = optimized.layer(0);

  // we expect layer has exactly 3 OpSequence
  ASSERT_EQ(3, layer0.width());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer0[0].length());
  ASSERT_EQ(7, layer0[0].at(0).protoContext().nodeId());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer0[1].length());
  ASSERT_EQ(8, layer0[1].at(0).protoContext().nodeId());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer0[2].length());
  ASSERT_EQ(9, layer0[2].at(0).protoContext().nodeId());

  // checking second layer now
  auto layer1 = optimized.layer(1);

  // we expect layer has exactly 3 OpSequences
  ASSERT_EQ(3, layer1.width());

  ASSERT_EQ(1, layer1[0].length());
  ASSERT_EQ(10, layer1[0].at(0).protoContext().nodeId());

  ASSERT_EQ(1, layer1[1].length());
  ASSERT_EQ(11, layer1[1].at(0).protoContext().nodeId());

  ASSERT_EQ(1, layer1[2].length());
  ASSERT_EQ(12, layer1[2].at(0).protoContext().nodeId());
}

TEST_F(GraphAnalysisTests, optimizedGraph_9) {
  // start graph

  Graph graph;

  graph.addVariable("A", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("B", NDArray('c', {3}, {2, 2, 2}, sd::DataType::INT32));
  graph.addVariable("C", NDArray('c', {3}, {3, 3, 3}, sd::DataType::INT32));
  graph.addVariable("D", NDArray('c', {3}, {4, 4, 4}, sd::DataType::INT32));

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

  const auto& optimized = graph.optimizedGraph();

  // we expect that OptimizedGraph has exactly 1 layer
  ASSERT_EQ(3, optimized.numOfLayers());

  auto layer = optimized.layer(0);
  // we expect layer has exactly 1 OpSequence
  ASSERT_EQ(1, layer.width());

  // we expect that OpSequence has exactly 2 ops
  ASSERT_EQ(1, layer[0].length());
  ASSERT_EQ(5, layer[0].at(0).protoContext().nodeId());

  auto layer1 = optimized.layer(1);
  // we expect layer has exactly 4 OpSequence
  ASSERT_EQ(4, layer1.width());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer1[0].length());
  ASSERT_EQ(6, layer1[0].at(0).protoContext().nodeId());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer1[1].length());
  ASSERT_EQ(7, layer1[1].at(0).protoContext().nodeId());

  // sequence = layer1[2];
  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer1[2].length());
  ASSERT_EQ(8, layer1[2].at(0).protoContext().nodeId());

  // sequence = layer1[3];
  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer1[3].length());
  ASSERT_EQ(9, layer1[3].at(0).protoContext().nodeId());

  auto layer2 = optimized.layer(2);
  // we expect layer has exactly 4 OpSequence
  ASSERT_EQ(8, layer2.width());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer2[0].length());
  ASSERT_EQ(10, layer2[0].at(0).protoContext().nodeId());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer2[1].length());
  ASSERT_EQ(11, layer2[1].at(0).protoContext().nodeId());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer2[2].length());
  ASSERT_EQ(12, layer2[2].at(0).protoContext().nodeId());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer2[3].length());
  ASSERT_EQ(13, layer2[3].at(0).protoContext().nodeId());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer2[4].length());
  ASSERT_EQ(14, layer2[4].at(0).protoContext().nodeId());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer2[5].length());
  ASSERT_EQ(15, layer2[5].at(0).protoContext().nodeId());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer2[6].length());
  ASSERT_EQ(16, layer2[6].at(0).protoContext().nodeId());

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, layer2[7].length());
  ASSERT_EQ(17, layer2[7].at(0).protoContext().nodeId());
}

TEST_F(GraphAnalysisTests, optimizedGraph_10) {
  Graph graph;

  graph.addVariable("A", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("B", NDArray('c', {3}, {2, 2, 2}, sd::DataType::INT32));
  graph.addVariable("C", NDArray('c', {3}, {3, 3, 3}, sd::DataType::INT32));
  graph.addVariable("D", NDArray('c', {3}, {3, 3, 3}, sd::DataType::INT32));

  Node a(sd::ops::multiply(), "a");
  Node b(sd::ops::add(), "b");
  Node c(sd::ops::multiply(), "c");
  Node d(sd::ops::subtract(), "d");

  graph.addNode(a, {"A", "B"});
  graph.addNode(b, {"a", "C"});
  graph.addNode(c, {"a", "D"});
  graph.addNode(d, {"a", "b", "c"});

  // we just check that nodes were really added
  ASSERT_EQ(4, graph.size());

  const auto& optimized = graph.optimizedGraph();

  // we expect that OptimizedGraph has exactly 1 layer
  ASSERT_EQ(3, optimized.numOfLayers());

  auto layer = optimized.layer(0);

  // we expect layer has exactly 1 OpSequence
  ASSERT_EQ(1, layer.width());
  auto sequence = layer[0];

  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, sequence.length());
  ASSERT_EQ(5, sequence.at(0).protoContext().nodeId());

  auto layer1 = optimized.layer(1);

  // we expect layer has exactly 2 OpSequence
  ASSERT_EQ(2, layer1.width());
  sequence = layer1[0];
  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, sequence.length());
  ASSERT_EQ(6, sequence.at(0).protoContext().nodeId());
  sequence = layer1[1];
  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, sequence.length());
  ASSERT_EQ(7, sequence.at(0).protoContext().nodeId());

  auto layer2 = optimized.layer(2);
  // we expect layer has exactly 1 OpSequence
  ASSERT_EQ(1, layer2.width());
  sequence = layer2[0];
  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, sequence.length());
  ASSERT_EQ(8, sequence.at(0).protoContext().nodeId());
}

TEST_F(GraphAnalysisTests, optimizedGraph_11) {
  Graph graph;

  graph.addVariable("A", NDArray('c', {3}, {1, 1, 1}, sd::DataType::INT32));
  graph.addVariable("B", NDArray('c', {3}, {2, 2, 2}, sd::DataType::INT32));
  graph.addVariable("C", NDArray('c', {3}, {3, 3, 3}, sd::DataType::INT32));
  graph.addVariable("D", NDArray('c', {3}, {3, 3, 3}, sd::DataType::INT32));

  Node a(sd::ops::multiply(), "a");
  Node b(sd::ops::add(), "b");
  Node c(sd::ops::multiply(), "c");
  Node d(sd::ops::subtract(), "d");

  graph.addNode(a, {"A", "B"});
  graph.addNode(b, {"A", "C"});
  graph.addNode(c, {"B", "D"});
  graph.addNode(d, {"C", "D"});

  // we just check that nodes were really added
  ASSERT_EQ(4, graph.size());

  const auto& optimized = graph.optimizedGraph();

  // we expect that OptimizedGraph has exactly 1 layer
  ASSERT_EQ(1, optimized.numOfLayers());

  auto layer = optimized.layer(0);

  // we expect layer has exactly 1 OpSequence
  ASSERT_EQ(4, layer.width());
  auto sequence = layer[0];
  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, sequence.length());
  ASSERT_EQ(5, sequence.at(0).protoContext().nodeId());
  sequence = layer[1];
  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, sequence.length());
  ASSERT_EQ(6, sequence.at(0).protoContext().nodeId());
  sequence = layer[2];
  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, sequence.length());
  ASSERT_EQ(7, sequence.at(0).protoContext().nodeId());
  sequence = layer[3];
  // we expect that OpSequence has exactly 1 ops
  ASSERT_EQ(1, sequence.length());
  ASSERT_EQ(8, sequence.at(0).protoContext().nodeId());
}

TEST_F(GraphAnalysisTests, optimizedGraph_12) {
  Graph graph;

  graph.addVariable("start", NDArrayFactory::create<int>(0));
  graph.addVariable("step", NDArrayFactory::create<int>(1));

  graph.addVariable("const_1", NDArrayFactory::create<int>(0));
  graph.addVariable("const_2", NDArrayFactory::create<int>(2));

  // generating "stop" argument for Range op
  graph.addNode(Node(sd::ops::add(), "add"), {"const_1", "const_2"});

  // generating axis, should be equal to {0}
  graph.addNode(Node(sd::ops::range(), "range_1"), {"start", "add", "step"});

  graph.addNode(Node(sd::ops::range(), "range_2"), {"range_1", "add", "step"});

  auto &optimized = graph.optimizedGraph();

  // graph.printOut();

  // we expect exactly 1 layer
  ASSERT_EQ(1, optimized.layers());
  auto layer = optimized.layer(0);

  // we expect exactly 1 OpSequence wihtin this layer
  ASSERT_EQ(1, layer.width());
  auto seq = layer.at(0);

  // this Graph doesn't allow any variance here. Order must be exactly the same as below
  ASSERT_EQ(std::string("add"),     seq[0].node().name());
  ASSERT_EQ(std::string("range_1"), seq[1].node().name());
  ASSERT_EQ(std::string("range_2"), seq[2].node().name());
}

TEST_F(GraphAnalysisTests, optimizedGraph_13) {
  Graph graph;

  graph.addPlaceholder("input", sd::DataType::FLOAT32, {-1, 3, 244, 244});
  graph.addVariable("weights_1", NDArrayFactory::create<float>(0.1f));
  graph.addVariable("weights_2", NDArrayFactory::create<float>(0.1f));
  graph.addVariable("weights_3", NDArrayFactory::create<float>(0.1f));
  graph.addVariable("weights_4", NDArrayFactory::create<float>(0.1f));

  graph.addVariable("axis", NDArrayFactory::create<int>(1));

  graph.addNode(Node(sd::ops::tanh(), "conv_1"), {"input", "weights_1"});
  graph.addNode(Node(sd::ops::tanh(), "pooling_1"), {"conv_1"});

  // branch 1
  graph.addNode(Node(sd::ops::tanh(), "conv_2"), {"pooling_1", "weights_2"});
  graph.addNode(Node(sd::ops::tanh(), "pooling_2"), {"conv_2"});

  // branch 2
  graph.addNode(Node(sd::ops::tanh(), "conv_3"), {"pooling_1", "weights_3"});
  graph.addNode(Node(sd::ops::tanh(), "pooling_3"), {"conv_3"});

  // branch 3
  graph.addNode(Node(sd::ops::tanh(), "conv_4"), {"pooling_1", "weights_4"});
  graph.addNode(Node(sd::ops::tanh(), "pooling_4"), {"conv_4"});

  // merge branch
  graph.addNode(Node(sd::ops::concat(), "concat"), {"pooling_2", "pooling_3", "pooling_4", "axis"});

  auto &optimized = graph.optimizedGraph();

  // we expect exactly 3 layers
  ASSERT_EQ(3, optimized.layers());
  auto layer = optimized.layer(0);

  // layer 0 must have exactly 1 sequence of 2 ops: conv_1 and pooling_1
  ASSERT_EQ(1, layer.width());
  auto seq = layer[0];

  ASSERT_EQ(2, seq.length());
  ASSERT_EQ(std::string("conv_1"),     seq[0].node().name());
  ASSERT_EQ(std::string("pooling_1"), seq[1].node().name());

  layer = optimized.layer(1);
  ASSERT_EQ(3, layer.width());

  seq = layer[0];
  ASSERT_EQ(2, seq.length());
  ASSERT_EQ(std::string("conv_2"),     seq[0].node().name());
  ASSERT_EQ(std::string("pooling_2"), seq[1].node().name());

  seq = layer[1];
  ASSERT_EQ(2, seq.length());
  ASSERT_EQ(std::string("conv_3"),     seq[0].node().name());
  ASSERT_EQ(std::string("pooling_3"), seq[1].node().name());

  seq = layer[2];
  ASSERT_EQ(2, seq.length());
  ASSERT_EQ(std::string("conv_4"),     seq[0].node().name());
  ASSERT_EQ(std::string("pooling_4"), seq[1].node().name());

  layer = optimized.layer(2);
  ASSERT_EQ(1, layer.width());

  seq = layer[0];
  ASSERT_EQ(1, seq.length());
  ASSERT_EQ(std::string("concat"),     seq[0].node().name());
}

TEST_F(GraphAnalysisTests, test_cond_1) {

  auto graph = Graph::fromFlatBuffers("resources/cond_true.fb");
  const auto& optimized = graph.optimizedGraph();
  // graph.printOut();

  // we expect exactly 3 layers
  ASSERT_EQ(3, optimized.layers());

  // layer 0
  auto layer = optimized.layer(0);
  ASSERT_EQ(1, layer.width());
  auto seq = layer[0];
  ASSERT_EQ(2, seq.length());
  ASSERT_EQ(std::string("in_0/read"),   seq[0].node().name());
  ASSERT_EQ(std::string("cond/Switch"), seq[1].node().name());

  // layer 1
  layer = optimized.layer(1);
  ASSERT_EQ(2, layer.width());
  seq = layer[0];
  ASSERT_EQ(2, seq.length());
  ASSERT_EQ(std::string("cond/switch_t"), seq[0].node().name());
  ASSERT_EQ(std::string("cond/LinSpace"), seq[1].node().name());
  seq = layer[1];
  ASSERT_EQ(1, seq.length());
  ASSERT_EQ(std::string("cond/switch_f"), seq[0].node().name());

  // layer 2
  layer = optimized.layer(2);
  ASSERT_EQ(1, layer.width());
  seq = layer[0];
  ASSERT_EQ(1, seq.length());
  ASSERT_EQ(std::string("cond/Merge"), seq[0].node().name());

  // graph.execute();
  /*
  some infor that would be useful for implementation
  currently on optimization graph is passing next data

  Node name: cond/switch_f; ID: 11; Input: 9, 0; Operation type: 21;  Operation
  class: -1719689536 Node name: cond/switch_t; ID: 10; Input: 9, 1; Operation
  type: 21;  Operation class: -1719689536 Node name: cond/Switch;   ID: 9;
  Input: 1, 0; Operation type: 119; Operation class: -1719689536 Node name:
  cond/Switch;   ID: 9;  Input: 6, 0; Operation type: 119; Operation class:
  -1719689536 Node name: cond/Merge;    ID: 8;  Input: 5, 0; Operation type:
  119; Operation class: -1719689536 Node name: cond/Merge;    ID: 8;  Input: 7,
  0; Operation type: 119; Operation class: -1719689536 Node name: in_0/read; ID:
  6;  Input: 1, 0; Operation type: 21;  Operation class: -1719689536 Node name:
  cond/LinSpace; ID: 7;  Input: 2, 0; Operation type: 21;  Operation class:
  -1719689536 Node name: cond/LinSpace; ID: 7;  Input: 3, 0; Operation type: 21;
  Operation class: -1719689536 Node name: cond/LinSpace; ID: 7;  Input: 4, 0;
  Operation type: 21;  Operation class: -1719689536

  as it can be seen cond/LinSpace is not connected with any switch node(s) that
  causes wrong results of optimization. also maybe to cover all conditional
  operations will be need "Operation class", but this have to discovered deeper.

  All above is true for test_cond_2
  */
}

TEST_F(GraphAnalysisTests, test_cond_2) {

  auto graph = Graph::fromFlatBuffers("resources/cond_false.fb");
  const auto& optimized = graph.optimizedGraph();
  // graph.printOut();

  ASSERT_EQ(3, optimized.layers());

  // layer 0
  auto layer = optimized.layer(0);
  ASSERT_EQ(1, layer.width());
  auto seq = layer[0];
  ASSERT_EQ(2, seq.length());
  ASSERT_EQ(std::string("in_0/read"),   seq[0].node().name());
  ASSERT_EQ(std::string("cond/Switch"), seq[1].node().name());

  // layer 1
  layer = optimized.layer(1);
  ASSERT_EQ(2, layer.width());
  seq = layer[0];
  ASSERT_EQ(2, seq.length());
  ASSERT_EQ(std::string("cond/switch_t"), seq[0].node().name());
  ASSERT_EQ(std::string("cond/LinSpace"), seq[1].node().name());
  seq = layer[1];
  ASSERT_EQ(1, seq.length());
  ASSERT_EQ(std::string("cond/switch_f"), seq[0].node().name());

  // layer 2
  layer = optimized.layer(2);
  ASSERT_EQ(1, layer.width());
  seq = layer[0];
  ASSERT_EQ(1, seq.length());
  ASSERT_EQ(std::string("cond/Merge"), seq[0].node().name());

}

TEST_F(GraphAnalysisTests, test_while_iter_1_1) {
  auto graph = Graph::fromFlatBuffers("resources/while_iter1.fb");
  //graph.printOut();
}
