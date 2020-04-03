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
#include <flatbuffers/flatbuffers.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/Node.h>
#include <graph/Graph.h>
#include <graph/GraphUtils.h>
#include <array/NDArray.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/generic/parity_ops.cpp>

using namespace sd;
using namespace sd::graph;

class GraphTests : public testing::Test {
public:
    /*
    int cShape[] = {2, 2, 2, 2, 1, 0, 1, 99};
    int fShape[] = {2, 2, 2, 1, 2, 0, 1, 102};
     */
    GraphTests() {
        //Environment::getInstance()->setDebug(true);
        //Environment::getInstance()->setVerbose(true);
    }
};

TEST_F(GraphTests, SingleInput1) {
    Graph graph;

    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(-2.0f);

    graph.variableSpace().putVariable(-1, x);

    Node nodeA(OpType_TRANSFORM_SAME, transform::Abs);
    Node nodeB(OpType_TRANSFORM_STRICT, transform::Cosine);
    Node nodeC(OpType_TRANSFORM_SAME, transform::Abs);

    graph.addNode(nodeA, {-1});
    graph.addNode(nodeB, {1});
    graph.addNode(nodeC, {2});

    ASSERT_EQ(3, graph.size());

    graph.execute();

    ASSERT_TRUE(graph.variableSpace().hasVariable(3));

    auto node3 = graph.variableSpace().getVariable(3)->getNDArray();

    ASSERT_NEAR(0.4161468, node3->reduceNumber(reduce::Mean).e<float>(0), 1e-5);
}

TEST_F(GraphTests, DoubleInput1) {
    Graph graph;

    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(-2.0);

    auto y = NDArrayFactory::create<float>('c', {5, 5});
    y.assign(-1.0);

    auto z = NDArrayFactory::create<float>('c', {5, 5});

    graph.variableSpace().putVariable(-1, x);
    graph.variableSpace().putVariable(-2, y);
    graph.variableSpace().putVariable(-3, z);

    Node nodeA(OpType_TRANSFORM_SAME, transform::Abs);
    Node nodeB(OpType_TRANSFORM_SAME, transform::Abs);
    Node nodeC(OpType_PAIRWISE, pairwise::Add);

    graph.addNode(nodeA, {-1});
    graph.addNode(nodeB, {-2});
    graph.addNode(nodeC, {1, 2});


    ASSERT_EQ(3, graph.size());

    graph.execute();

    ASSERT_NEAR(3.0, z.reduceNumber(reduce::Mean).e<float>(0), 1e-5);
}

TEST_F(GraphTests, SingleInput3) {
    Graph graph;

    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(-2.0);

    auto v0 = NDArrayFactory::create<float>('c', {5, 5});
    auto v1 = NDArrayFactory::create<float>('c', {5, 5});

    graph.variableSpace().putVariable(-1, x);
    graph.variableSpace().putVariable(-2, v0);
    graph.variableSpace().putVariable(-3, v1);

    Node nodeA(OpType_TRANSFORM_SAME, transform::Abs);
    Node nodeB(OpType_TRANSFORM_FLOAT, transform::Sqrt);
    Node nodeC(OpType_TRANSFORM_SAME, transform::Ones);

    graph.addNode(nodeA, {-1});
    graph.addNode(nodeB, {1});
    graph.addNode(nodeC, {1});

    ASSERT_EQ(3, graph.size());

    graph.execute();

    ASSERT_NEAR(1.4142135, v0.reduceNumber(reduce::Mean).e<float>(0), 1e-5);
    ASSERT_NEAR(1.0, v1.reduceNumber(reduce::Mean).e<float>(0), 1e-5);
}

TEST_F(GraphTests, SingleInput4) {
    Graph graph;

    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(-2.0);

    auto v0 = NDArrayFactory::create<float>('c', {5, 5});
    auto v1 = NDArrayFactory::create<float>('c', {5, 5});

    graph.variableSpace().putVariable(-1, x);
    graph.variableSpace().putVariable(-2, v0);
    graph.variableSpace().putVariable(-3, v1);

    Node nodeA(OpType_TRANSFORM_SAME, transform::Abs);
    Node nodeB(OpType_TRANSFORM_FLOAT, transform::Sqrt);
    Node nodeC(OpType_TRANSFORM_SAME, transform::Neg);

    Node nodeS(OpType_TRANSFORM_SAME, transform::Ones);
    Node nodeE(OpType_TRANSFORM_SAME, transform::Identity);

    graph.addNode(nodeA, {-1});
    graph.addNode(nodeB, {1});
    graph.addNode(nodeC, {2});
    graph.addNode(nodeS, {3});
    graph.addNode(nodeE, {3});

    ASSERT_EQ(5, graph.size());

    graph.execute();

    ASSERT_NEAR(1.0, v0.reduceNumber(reduce::Mean).e<float>(0), 1e-5);
    ASSERT_NEAR(-1.4142135, v1.reduceNumber(reduce::Mean).e<float>(0), 1e-5);
}


TEST_F(GraphTests, DoubleInput2) {
    Graph graph;

    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(-2.0);

    auto y = NDArrayFactory::create<float>('c', {5, 5});
    y.assign(-1.0);

    auto z0 = NDArrayFactory::create<float>('c', {5, 5});
    auto z1 = NDArrayFactory::create<float>('c', {5, 5});

    graph.variableSpace().putVariable(-1, x);
    graph.variableSpace().putVariable(-2, y);
    graph.variableSpace().putVariable(-3, z0);
    graph.variableSpace().putVariable(-4, z1);


    Node nodeA(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
    Node nodeB(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {3});
    Node nodeC(OpType_TRANSFORM_SAME, transform::Neg, 3, {2}, {-3});

    Node nodeT(OpType_TRANSFORM_SAME, transform::Abs, 11, {-2}, {12});
    Node nodeU(OpType_TRANSFORM_FLOAT, transform::Sqrt, 12, {11}, {13});
    Node nodeV(OpType_TRANSFORM_SAME, transform::Neg, 13, {12}, {-4});

    graph.addNode(nodeA, {-1});
    graph.addNode(nodeB, {1});
    graph.addNode(nodeC, {2});
    graph.addNode(nodeT, {-2});
    graph.addNode(nodeU, {4});
    graph.addNode(nodeV, {5});

    ASSERT_EQ(6, graph.size());

    graph.execute();

    ASSERT_NEAR(-1.4142135, z0.reduceNumber(reduce::Mean).e<float>(0), 1e-5);
    ASSERT_NEAR(-1.0, z1.reduceNumber(reduce::Mean).e<float>(0), 1e-5);
}


TEST_F(GraphTests, DoubleInput3) {
    Graph graph;

    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(-2.0);

    auto y = NDArrayFactory::create<float>('c', {5, 5});
    y.assign(-1.0);

    auto z0 = NDArrayFactory::create<float>('c', {5, 5});
    auto z1 = NDArrayFactory::create<float>('c', {5, 5});


    auto w = NDArrayFactory::create<float>('c', {5, 5});

    graph.variableSpace().putVariable(-1, x);
    graph.variableSpace().putVariable(-2, y);
    graph.variableSpace().putVariable(-3, z0);
    graph.variableSpace().putVariable(-4, z1);
    graph.variableSpace().putVariable(-5, w);


    Node nodeA(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
    Node nodeB(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {3});
    Node nodeC(OpType_TRANSFORM_SAME, transform::Neg, 3, {2}, {-3, 21});

    Node nodeT(OpType_TRANSFORM_SAME, transform::Abs, 11, {-2}, {12});
    Node nodeU(OpType_TRANSFORM_FLOAT, transform::Sqrt, 12, {11}, {13});
    Node nodeV(OpType_TRANSFORM_SAME, transform::Neg, 13, {12}, {-4, 21});

    Node nodeW(OpType_PAIRWISE, pairwise::Add, 21, {3, 13}, {22});
    Node nodeZ(OpType_TRANSFORM_SAME, transform::Abs, 22, {21}, {-5});

    graph.addNode(nodeA, {-1});
    graph.addNode(nodeB, {1});
    graph.addNode(nodeC, {2});
    graph.addNode(nodeT, {-2});
    graph.addNode(nodeU, {4});
    graph.addNode(nodeV, {5});
    graph.addNode(nodeW, {3, 6});
    graph.addNode(nodeZ, {7});

    ASSERT_EQ(8, graph.size());

    graph.execute();

    ASSERT_NEAR(-1.4142135, z0.reduceNumber(reduce::Mean).e<float>(0), 1e-5);
    ASSERT_NEAR(-1.0, z1.reduceNumber(reduce::Mean).e<float>(0), 1e-5);

    ASSERT_NEAR(2.4142135, w.reduceNumber(reduce::Mean).e<float>(0), 1e-5);
}


TEST_F(GraphTests, QuadInput1) {
    Graph graph;

    auto x0 = NDArrayFactory::create<float>('c', {5, 5});
    x0.assign(0.0);

    auto x1 = NDArrayFactory::create<float>('c', {5, 5});
    x1.assign(-1.0);

    auto x2 = NDArrayFactory::create<float>('c', {5, 5});
    x2.assign(-2.0);

    auto x3 = NDArrayFactory::create<float>('c', {5, 5});
    x3.assign(-3.0);

    auto z = NDArrayFactory::create<float>('c', {5, 5});
    z.assign(119.0);

    graph.variableSpace().putVariable(-1, x0);
    graph.variableSpace().putVariable(-2, x1);
    graph.variableSpace().putVariable(-3, x2);
    graph.variableSpace().putVariable(-4, x3);
    graph.variableSpace().putVariable(-5, z);

    Node nodeA(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {11});
    Node nodeB(OpType_TRANSFORM_SAME, transform::Abs, 2, {-2}, {11});
    Node nodeC(OpType_TRANSFORM_SAME, transform::Abs, 3, {-3}, {21});
    Node nodeD(OpType_TRANSFORM_SAME, transform::Abs, 4, {-4}, {21});

    Node nodeP1(OpType_PAIRWISE, pairwise::Add, 11, {1, 2}, {31});
    Node nodeP2(OpType_PAIRWISE, pairwise::Add, 21, {3, 4}, {31});

    Node nodeZ(OpType_PAIRWISE, pairwise::Add, 31, {11, 21}, {-5});

    graph.addNode(nodeA, {-1});
    graph.addNode(nodeB, {-2});
    graph.addNode(nodeC, {-3});
    graph.addNode(nodeD, {-4});
    graph.addNode(nodeP1, {1, 2});
    graph.addNode(nodeP2, {3, 4});
    graph.addNode(nodeZ, {11, 21});

    ASSERT_EQ(7, graph.size());

    graph.execute();

    ASSERT_NEAR(6.0, z.reduceNumber(reduce::Mean).e<float>(0), 1e-5);
}

TEST_F(GraphTests, InternalBranching1) {
    Graph graph;

    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(0.0);

    auto z = NDArrayFactory::create<float>('c', {5, 5});

    graph.variableSpace().putVariable(-1, x);
    graph.variableSpace().putVariable(-2, z);

    // 1.0
    Node nodeA(OpType_TRANSFORM_SAME, transform::Ones, 1, {-1}, {11, 21});

    // -1
    Node nodeK(OpType_TRANSFORM_SAME, transform::Neg, 11, {1}, {12});

    // 2.0
    Node nodeL(OpType_TRANSFORM_SAME, transform::OneMinus, 12, {11}, {31});

    // -1
    Node nodeR(OpType_TRANSFORM_SAME, transform::Neg, 21, {1}, {22});

    // 1
    Node nodeS(OpType_TRANSFORM_SAME, transform::Neg, 22, {21}, {31});

    // 1.0
    Node nodeZ(OpType_PAIRWISE, pairwise::Add, 31, {12, 22}, {-2});

    graph.addNode(nodeA, {-1});
    graph.addNode(nodeK, {1});
    graph.addNode(nodeL, {2});
    graph.addNode(nodeR, {1});
    graph.addNode(nodeS, {1});
    graph.addNode(nodeZ, {1, 1});

    ASSERT_EQ(6, graph.size());

    graph.execute();

    ASSERT_NEAR(3.0, z.reduceNumber(reduce::Mean).e<float>(0), 1e-5);
}


TEST_F(GraphTests, ReductionsTest1) {
    Graph graph;

    auto x = NDArrayFactory::create<float>('c', {5, 5});
    for (int r = 0; r < x.rows(); r++) {
        for (int c = 0; c < x.columns(); c++) {
            x.p(r, c, -c);
        }
    }

    auto z = NDArrayFactory::create<float>('c', {5});

    graph.variableSpace().putVariable(-1, x);
    graph.variableSpace().putVariable(-2, z);

    Node nodeA(OpType_REDUCE_FLOAT, reduce::Mean, 1, {-1}, {2}, {1}, {});
    Node nodeB(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {-2});

    graph.addNode(nodeA, {-1});
    graph.addNode(nodeB, {1});

    ASSERT_EQ(2, graph.size());

    graph.execute();

    ASSERT_NEAR(2.0, z.reduceNumber(reduce::Mean).e<float>(0), 1e-5);
}


TEST_F(GraphTests, IndexReductionsTest1) {
    Graph graph;

    auto x = NDArrayFactory::create<float>('c', {5, 5});
    for (int r = 0; r < x.rows(); r++) {
        for (int c = 0; c < x.columns(); c++) {
            x.p(r, c, -c);
        }
    }

    auto z = NDArrayFactory::create<Nd4jLong>('c', {5, 1});
    auto axis = NDArrayFactory::create<Nd4jLong>('c', {1}, {1});

    graph.variableSpace().putVariable(-1, x);
    graph.variableSpace().putVariable(-2, z);
    //graph->variableSpace().putVariable(-3, axis);


    Node nodeA(OpType_INDEX_REDUCE, indexreduce::IndexMin, 1, {-1}, {2}, {1});
    Node nodeB(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {-2});

    graph.addNode(nodeA, {-1, -2});
    graph.addNode(nodeB, {1});

    ASSERT_EQ(2, graph.size());

    graph.execute();

    ASSERT_NEAR(4.0, z.reduceNumber(reduce::Mean).e<float>(0), 1e-5);
}

#if 0
TEST_F(GraphTests, AutoOutput1) {
    auto graph = new Graph();
    auto x = NDArrayFactory::create_<float>('c', {5, 5});
    x->assign(-2.0);

    graph->variableSpace().putVariable(-1, x);

    auto nodeA = new Node(OpType_TRANSFORM_FLOAT, 0, 1, {-1}, {2});
    auto nodeB = new Node(OpType_TRANSFORM_FLOAT, 35, 2, {1}, {});

    graph->addNode(nodeA);
    graph->addNode(nodeB);

    ASSERT_EQ(1, graph->rootNodes());
    ASSERT_EQ(2, graph->totalNodes());

    graph->buildGraph();

    ASSERT_TRUE(graph->variableSpace()->getVariable(2) != nullptr);

    GraphExecutioner::execute(graph);

    auto outputs = graph->fetchOutputs();

    ASSERT_EQ(1, outputs->size());

    ASSERT_TRUE(outputs->at(0) != nullptr);

    ASSERT_NEAR(-1.0, outputs->at(0)->getNDArray()->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

    delete outputs;
    delete graph;
}


TEST_F(GraphTests, AutoOutput2) {
    auto graph = new Graph();
    auto x = NDArrayFactory::create_<float>('c', {5, 5});
    x->assign(-2.0);

    graph->variableSpace().putVariable(-1, x);

    auto nodeA = new Node(OpType_TRANSFORM_SAME, 0, 1, {-1}, {2, 3, -1});
    auto nodeB = new Node(OpType_TRANSFORM_SAME, 35, 2, {1}, {});
    auto nodeC = new Node(OpType_TRANSFORM_SAME, 6, 3, {1}, {});

    graph->addNode(nodeA);
    graph->addNode(nodeB);
    graph->addNode(nodeC);

    ASSERT_EQ(1, graph->rootNodes());
    ASSERT_EQ(3, graph->totalNodes());

    graph->buildGraph();

    ASSERT_TRUE(graph->variableSpace()->getVariable(-1) != nullptr);
    ASSERT_TRUE(graph->variableSpace()->getVariable(2) != nullptr);
    ASSERT_TRUE(graph->variableSpace()->getVariable(3) != nullptr);

    GraphExecutioner::execute(graph);

    auto outputs = graph->fetchOutputs();

    ASSERT_EQ(2, outputs->size());

    ASSERT_TRUE(outputs->at(0) != nullptr);

    ASSERT_NEAR(-1.0, outputs->at(0)->getNDArray()->reduceNumber(reduce::Mean).e<float>(0), 1e-5);
    ASSERT_NEAR(-2.0, outputs->at(1)->getNDArray()->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

    delete graph;
    delete outputs;
}
#endif

TEST_F(GraphTests, BroadcastTest1) {
    Graph graph;
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(0.f);

    auto y = NDArrayFactory::create<float>('c', {1, 5});
    for (int e = 0; e < y.columns(); e++) {
        y.p(e, (float)e+1);
    }

    auto z = NDArrayFactory::create<float>('c', {5, 5});

    graph.variableSpace().putVariable(-1, x);
    graph.variableSpace().putVariable(-2, y);
    graph.variableSpace().putVariable(-3, z);

    Node nodeA(OpType_BROADCAST, broadcast::Subtract, 1, {-1, -2}, {2}, {1});
    Node nodeB(OpType_TRANSFORM_SAME, transform::Neg, 2, {1}, {-3});

    graph.addNode(nodeA, {-1, -2});
    graph.addNode(nodeB, {1});

    graph.execute();

    ASSERT_NEAR(3.0, z.reduceNumber(reduce::Mean).e<float>(0), 1e-5);
}


TEST_F(GraphTests, ScalarTest1) {
    Graph graph;

    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(-2.0);

    auto z = NDArrayFactory::create<float>('c', {5, 5});

    graph.variableSpace().putVariable(-1, x);
    graph.variableSpace().putVariable(-2, z);

    Node nodeA(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
    Node nodeB(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {3});
    Node nodeE(OpType_SCALAR, scalar::Add, 3, {2}, {-2}, {}, 1.3f);

    graph.addNode(nodeA, {-1});
    graph.addNode(nodeB, {1});
    graph.addNode(nodeE, {2});

    ASSERT_EQ(3, graph.size());

    graph.execute();

    ASSERT_NEAR(2.714213, z.reduceNumber(reduce::Mean).e<float>(0), 1e-5);
}

TEST_F(GraphTests, SymbolicLookupTest1) {
    Graph graph;

    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(-2.0);

    auto z = NDArrayFactory::create<float>('c', {5, 5});

    std::string a("alpha");
    std::string o("omega");

    auto vX = std::make_shared<Variable>(x, a, -1);
    auto vZ = std::make_shared<Variable>(z, o, -1);

    graph.variableSpace().putVariable(-1, vX);
    graph.variableSpace().putVariable(-2, vZ);

    Node nodeA(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
    Node nodeB(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {-2});

    std::string p("phi");
    std::string t("theta");

    nodeA.setName(&p);
    nodeB.setName(&t);

    graph.addNode(nodeA, {-1});
    graph.addNode(nodeB, {1});


    auto rX = graph.variableSpace().getVariable(a);
    auto rZ = graph.variableSpace().getVariable(o);

    std::string om("omicron");

    ASSERT_TRUE(rX->getNDArray() == vX->getNDArray());
    ASSERT_TRUE(rZ->getNDArray() == vZ->getNDArray());
    ASSERT_FALSE(graph.variableSpace().hasVariable(om));


    ASSERT_TRUE(graph.variableSpace().hasVariable(1));
    ASSERT_TRUE(graph.variableSpace().hasVariable(2));

    graph.execute();

    ASSERT_TRUE(graph.variableSpace().hasVariable(p));
    ASSERT_TRUE(graph.variableSpace().hasVariable(t));

    ASSERT_NEAR(1.4142135, z.reduceNumber(reduce::Mean).e<float>(0), 1e-5);
}

#if 0
TEST_F(GraphTests, Test_Clone_1) {
    auto exp = NDArrayFactory::create<float>('c', {3});
    exp.assign(3.0);


    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/reduce_dim_false.fb");
    auto variableSpace = graph->variableSpace();
    //graph->buildGraph();

    auto clone = graph->clone();

    Nd4jStatus statusOriginal = GraphExecutioner::execute(graph);

    ASSERT_EQ(ND4J_STATUS_OK, statusOriginal);
    ASSERT_TRUE(variableSpace->hasVariable(3));

    Nd4jStatus statusClone = GraphExecutioner::execute(clone);

    ASSERT_EQ(ND4J_STATUS_OK, statusClone);

    ASSERT_TRUE(variableSpace->hasVariable(3));

    auto z0 = variableSpace->getVariable(3)->getNDArray();
    auto z1 = clone->getVariableSpace()->getVariable(3)->getNDArray();

    ASSERT_TRUE(exp.isSameShape(z0));
    ASSERT_TRUE(exp.equalsTo(z0));

    ASSERT_TRUE(exp.isSameShape(z1));
    ASSERT_TRUE(exp.equalsTo(z1));

    delete graph;
    delete clone;
}




TEST_F(GraphTests, Test_Clone_2) {
    auto exp = NDArrayFactory::create<float>('c', {3});
    exp.assign(3.0);


    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/reduce_dim_false.fb");
    auto variableSpace = graph->variableSpace();
    graph->buildGraph();

    auto clone = graph->clone();

    Nd4jStatus statusOriginal = GraphExecutioner::execute(graph);

    ASSERT_EQ(ND4J_STATUS_OK, statusOriginal);
    ASSERT_TRUE(variableSpace->hasVariable(3));

    Nd4jStatus statusClone = GraphExecutioner::execute(clone);

    ASSERT_EQ(ND4J_STATUS_OK, statusClone);

    ASSERT_TRUE(variableSpace->hasVariable(3));

    auto z0 = variableSpace->getVariable(3)->getNDArray();
    auto z1 = clone->getVariableSpace()->getVariable(3)->getNDArray();

    ASSERT_TRUE(exp.isSameShape(z0));
    ASSERT_TRUE(exp.equalsTo(z0));

    ASSERT_TRUE(exp.isSameShape(z1));
    ASSERT_TRUE(exp.equalsTo(z1));

    delete graph;
    delete clone;
}

TEST_F(GraphTests, Test_Dtype_Conversion_1) {
    /*auto expD = NDArrayFactory::create<double>('c', {3}, {3.0, 3.0, 3.0});
    auto expF = NDArrayFactory::create<float>('c', {3}, {3.0, 3.0, 3.0});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/reduce_dim_false.fb");
    graph->buildGraph();


    auto gd = graph->template asT<double>();
    auto gf = gd->template asT<float>();

    // checking float graph
    Nd4jStatus statusF = GraphExecutioner::execute(gf);
    ASSERT_EQ(ND4J_STATUS_OK, statusF);

    ASSERT_TRUE(gf->getVariableSpace()->hasVariable(3));

    ASSERT_TRUE(gf->getVariableSpace()->hasVariable(3));
    auto z1 = gf->getVariableSpace()->getVariable(3)->getNDArray();

    ASSERT_TRUE(expF.isSameShape(z1));
    ASSERT_TRUE(expF.equalsTo(z1));


    // checking double graph
    Nd4jStatus statusD = GraphExecutioner<double>::execute(gd);
    ASSERT_EQ(ND4J_STATUS_OK, statusD);

    ASSERT_TRUE(gd->getVariableSpace()->hasVariable(3));
    auto z2 = gd->getVariableSpace()->getVariable(3)->getNDArray();

    ASSERT_TRUE(expD.isSameShape(z2));
    ASSERT_TRUE(expD.equalsTo(z2));


    delete graph;
    delete gd;
    delete gf;
    */
}

TEST_F(GraphTests, Test_Dtype_Conversion_2) {
   /*
    NDArray<float> expF('c', {5, 4}, {0.32454616f, -0.06604697f, 0.22593613f, 0.43166467f, -0.18320604f, 0.00102305f, -0.06963076f, 0.25266643f, 0.07568010f, -0.03009197f, 0.07805517f, 0.33180334f, -0.06220427f, 0.07249600f, -0.06726961f, -0.22998397f, -0.06343779f, 0.07384885f, -0.06891008f,  -0.23745790f});
    NDArray<double> expD('c', {5, 4}, {0.32454616f, -0.06604697f, 0.22593613f, 0.43166467f, -0.18320604f, 0.00102305f, -0.06963076f, 0.25266643f, 0.07568010f, -0.03009197f, 0.07805517f, 0.33180334f, -0.06220427f, 0.07249600f, -0.06726961f, -0.22998397f, -0.06343779f, 0.07384885f, -0.06891008f,  -0.23745790f});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/ae_00.fb");
    graph->buildGraph();


    auto gd = graph->template asT<double>();
    auto gf = gd->template asT<float>();

    // checking float
    auto resultF = GraphExecutioner::execute(gf);
    ASSERT_EQ(ND4J_STATUS_OK, resultF);
    ASSERT_TRUE(gf->getVariableSpace()->hasVariable(18));
    auto zF = gf->getVariableSpace()->getVariable(18)->getNDArray();

    ASSERT_TRUE(expF.isSameShape(zF));
    ASSERT_TRUE(expF.equalsTo(zF));


    // checking double
    auto resultD = GraphExecutioner<double>::execute(gd);
    ASSERT_EQ(ND4J_STATUS_OK, resultD);
    ASSERT_TRUE(gd->getVariableSpace()->hasVariable(18));
    auto zD = gd->getVariableSpace()->getVariable(18)->getNDArray();

    ASSERT_TRUE(expD.isSameShape(zD));
    ASSERT_TRUE(expD.equalsTo(zD));

    delete graph;
    delete gd;
    delete gf;
    */
}

TEST_F(GraphTests, Test_Hash_Function_1) {
    /*
    auto graph0 = GraphExecutioner::importFromFlatBuffers("./resources/ae_00.fb");
    auto graph1 = GraphExecutioner::importFromFlatBuffers("./resources/ae_00.fb");
    auto graph2 = GraphExecutioner::importFromFlatBuffers("./resources/conv_0.fb");

    ASSERT_EQ(graph0->hashCode(), graph1->hashCode());
    ASSERT_NE(0L, graph1->hashCode());
    ASSERT_NE(graph0->hashCode(), graph2->hashCode());

    auto graph0D = graph0->template asT<double>();
    auto graph1D = graph1->template asT<double>();

    ASSERT_NE(graph0->hashCode(), graph0D->hashCode());
    ASSERT_EQ(graph0D->hashCode(), graph1D->hashCode());

    delete graph0;
    delete graph1;
    delete graph2;
    delete graph0D;
    delete graph1D;
    */
}

TEST_F(GraphTests, OpListTest_1) {
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/ae_00.fb"); ;

    ASSERT_TRUE(graph != nullptr);
    std::vector<OpDescriptor> ops = graph->getOperations();

    ASSERT_TRUE(ops.size() == 11);
    GraphUtils::filterOperations(ops);
    ASSERT_TRUE(ops.size() == 7);

    std::string exp(" -g \"-DSD_OPS_LIST='-DOP_rank=true -DOP_range=true -DOP_subtract=true -DOP_permute=true -DOP_matmul=true -DOP_biasadd=true -DOP_TRANSFORM{15}=true '\"");
    std::string out = GraphUtils::makeCommandLine(ops);
//    nd4j_printf("EXP: >%s<\n", exp.c_str());
//    nd4j_printf("OUT: >%s<\n", out.c_str());
    ASSERT_EQ(exp, out);

    delete graph;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(GraphTests, OpListTest_2) {
    auto graph0 = GraphExecutioner::importFromFlatBuffers("./resources/ae_00.fb");
    auto graph1 = GraphExecutioner::importFromFlatBuffers("./resources/tensor_slice.fb");

    ASSERT_TRUE(graph0 != nullptr);
    ASSERT_TRUE(graph1 != nullptr);

    std::vector<OpDescriptor> ops = graph0->getOperations();
    std::vector<OpDescriptor> ops1 = graph1->getOperations();
    std::copy ( ops1.begin(), ops1.end(),  std::back_inserter(ops));

    ASSERT_TRUE(ops.size() == 13);

    GraphUtils::filterOperations(ops);

    std::string exp = " -g \"-DSD_OPS_LIST='-DOP_rank=true -DOP_range=true -DOP_subtract=true -DOP_permute=true -DOP_matmul=true -DOP_biasadd=true -DOP_TRANSFORM{15}=true -DOP_strided_slice=true -DOP_ACCUMULATION{1}=true '\"";

    ASSERT_TRUE(ops.size() == 9);
    ASSERT_EQ(exp, GraphUtils::makeCommandLine(ops));

    delete graph0;
    delete graph1;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(GraphTests, OpListTest_3) {
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/ae_00.fb"); ;

    ASSERT_TRUE(graph != nullptr);
    std::vector<OpDescriptor> ops = graph->getOperations();
    std::vector<OpDescriptor> ops2(ops);
    std::copy(ops.begin(), ops.end(),  std::back_inserter(ops2));

    ASSERT_TRUE(ops.size() == 11);
    ASSERT_TRUE(ops2.size() == 2 * ops.size());

    GraphUtils::filterOperations(ops2);
    GraphUtils::filterOperations(ops);
    ASSERT_TRUE(ops.size() == ops2.size());
    ASSERT_TRUE(ops.size() == 7);
    ASSERT_TRUE(GraphUtils::makeCommandLine(ops) == GraphUtils::makeCommandLine(ops2));

    delete graph;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(GraphTests, OpListTest_4) {
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/conv_0.fb"); ;

    ASSERT_TRUE(graph != nullptr);
    std::vector<OpDescriptor> ops = graph->getOperations();
    std::vector<OpDescriptor> ops2(ops);
    std::copy(ops.begin(), ops.end(),  std::back_inserter(ops2));

    // nd4j_printf("Total ops before %i\n", ops.size());
    ASSERT_TRUE(ops.size() == 6);
    ASSERT_TRUE(ops2.size() == 2 * ops.size());

    GraphUtils::filterOperations(ops2);
    GraphUtils::filterOperations(ops);
    ASSERT_TRUE(ops.size() == ops2.size());
    ASSERT_TRUE(ops.size() == 5);
    ASSERT_TRUE(GraphUtils::makeCommandLine(ops) == GraphUtils::makeCommandLine(ops2));

    delete graph;
}


TEST_F(GraphTests, Test_Inplace_Execution_1) {
    auto exp = NDArrayFactory::create<float>('c', {5, 4}, {0.32454616f, -0.06604697f, 0.22593613f, 0.43166467f, -0.18320604f, 0.00102305f, -0.06963076f, 0.25266643f, 0.07568010f, -0.03009197f, 0.07805517f, 0.33180334f, -0.06220427f, 0.07249600f, -0.06726961f, -0.22998397f, -0.06343779f, 0.07384885f, -0.06891008f,  -0.23745790f});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/ae_00.fb");
    // graph->printOut();
    graph->tagInplaceNodes();

    ASSERT_FALSE(graph->nodeById(8)->isInplace());
    ASSERT_TRUE(graph->nodeById(9)->isInplace());
    ASSERT_TRUE(graph->nodeById(10)->isInplace());
    ASSERT_FALSE(graph->nodeById(11)->isInplace());
    ASSERT_FALSE(graph->nodeById(12)->isInplace());
    ASSERT_TRUE(graph->nodeById(17)->isInplace());
    ASSERT_TRUE(graph->nodeById(18)->isInplace());

    auto status = GraphExecutioner::execute(graph, graph->variableSpace());
    ASSERT_EQ(Status::OK(), status);

    auto z = graph->variableSpace()->getVariable(18)->getNDArray();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    auto z_17 = graph->variableSpace()->getVariable(17)->getNDArray();
    ASSERT_TRUE(z_17 == z);

    delete graph;
}

TEST_F(GraphTests, Test_Inplace_Execution_2) {
    Graph graphA;

    auto x = NDArrayFactory::create_<float>('c', {5, 5});
    x->assign(-5.0);

    graphA.getVariableSpace()->putVariable(-1, x);

    // abs, result is 5
    auto nodeA0 = new Node(OpType_TRANSFORM_SAME, 0, 1, {-1}, {});
    // 1-, result -4
    auto nodeA1 = new Node(OpType_TRANSFORM_SAME, 35, 2, {1}, {});

    // graph should return 4: abs(-4)
    auto nodeA2 = new Node(OpType_TRANSFORM_SAME, 0, 3, {2}, {});

    // graph should return 1 - 4 = -3
    auto nodeA21 = new Node(OpType_TRANSFORM_SAME, 35, 5, {3}, {});

    // 1 - -4 = 3
    auto nodeA3 = new Node(OpType_TRANSFORM_SAME, 35, 4, {2}, {});

    // same abs = 3
    auto nodeA31 = new Node(OpType_TRANSFORM_SAME, 35, 6, {4}, {});

    graphA.addNode(nodeA0);
    graphA.addNode(nodeA1);
    graphA.addNode(nodeA2);
    graphA.addNode(nodeA3);
    graphA.addNode(nodeA21);
    graphA.addNode(nodeA31);

    graphA.buildGraph();
    graphA.tagInplaceNodes();

    // nodes have 1 output
    ASSERT_TRUE(graphA.nodeById(1)->isInplace());
    ASSERT_TRUE(graphA.nodeById(2)->isInplace());

    // this 2 nodes share same input: node 2, so they can't be inplace
    ASSERT_FALSE(graphA.nodeById(3)->isInplace());
    ASSERT_FALSE(graphA.nodeById(4)->isInplace());

    // these 2 ops are standalone, so they can be run inplace
    ASSERT_TRUE(graphA.nodeById(5)->isInplace());
    ASSERT_TRUE(graphA.nodeById(6)->isInplace());
}
#endif

TEST_F(GraphTests, Test_Inplace_Outputs_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    auto exp = NDArrayFactory::create<float>('c', {6}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    auto z = NDArrayFactory::create<float>('c', {2, 3});

    sd::ops::test_output_reshape op;
    auto result = op.execute({&x}, {&z}, {}, {}, {});
    ASSERT_EQ(Status::OK(), result);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(GraphTests, Test_Inplace_Outputs_2) {
#ifndef __APPLE_OS__
    // we dont want testing this on apple. due to try/catch

    auto x = NDArrayFactory::create<float>('c', {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    auto exp = NDArrayFactory::create<float>('c', {6}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    auto z = NDArrayFactory::create<float>('c', {3, 3});

    bool failed = false;
    sd::ops::test_output_reshape op;
    try {
        op.execute({&x}, {&z}, {}, {}, {});

    } catch (const std::runtime_error& e) {
        failed = true;
    }
    
    
    ASSERT_TRUE(failed);
#endif
}

/*
TEST_F(GraphTests, Test_Minifier_1) {
    // run preprocessor to produce single header
    // if all ok - return value is 0, if error - non-zero value will be returned
    std::string input("../include/ops/ops.h"); //declarable/CustomOperations.h");

    ASSERT_EQ(0, GraphUtils::runPreprocessor(input.c_str(), "libnd4j_mini.hpp"));
    // remove file from filesystem
#ifdef __linux__
    ASSERT_EQ(0, unlink("libnd4j_mini.hpp"));
#endif
}
*/

TEST_F(GraphTests, Test_Minifier_2) {

    // run preprocessor to produce single header
    // if all ok - return value is 0, if error - non-zero value will be returned
    ASSERT_EQ(0, GraphUtils::runPreprocessor("../include/ops/specials.h", "libnd4j_mini2.hpp"));
    // remove file from filesystem
#ifdef __linux__
    ASSERT_EQ(0, unlink("libnd4j_mini2.hpp"));
#endif
}

TEST_F(GraphTests, Test_Minifier_3) {

    // run preprocessor to produce single header
    // if all ok - return value is 0, if error - non-zero value will be returned
#ifdef __linux__
    ASSERT_EQ(0x100, GraphUtils::runPreprocessor("/include/ops/ops.h", "libnd4j_mini3.hpp"));
#endif
    // remove file from filesystem
    //ASSERT_EQ(0, unlink("libnd4j_mini3.hpp"));

}
