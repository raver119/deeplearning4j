/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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
// Created by yves on 2020.03.06.
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
#include <ops/declarable/helpers/convolutions.h>

#include <helpers/BenchmarkHelper.h>
#include <ops/declarable/helpers/scatter.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <array>

#include <ops/declarable/helpers/legacy_helpers.h>
#include <ops/declarable/helpers/addBias.h>

using namespace sd;

class BertPerfTests : public testing::Test {
public:
    int numIterations = 3;
    int poolSize = 10;

    BertPerfTests() {
    }
};

TEST_F(BertPerfTests, test_bert_full_no_prof) {
    // this test will run ONLY if this model exists

    if (sd::graph::getFileSize("resources/BertFull/model.fb") < 0)
        return;
    auto timeStart = std::chrono::system_clock::now();
    auto graph = GraphExecutioner::importFromFlatBuffers("resources/BertFull/model.fb");
    auto timeFileLoad = std::chrono::system_clock::now();
    nd4j_printf("Graph successfully loaded\n", "");
    auto t = NDArrayFactory::fromNpyFile("resources/BertFull/in0_IteratorGetNext.npy");
    auto u = NDArrayFactory::fromNpyFile("resources/BertFull/in1_IteratorGetNext_1.npy");
    auto v = NDArrayFactory::fromNpyFile("resources/BertFull/in2_IteratorGetNext_4.npy");
    auto z = NDArrayFactory::fromNpyFile("resources/BertFull/out_loss-Softmax.npy");
    auto timeInOutLoad = std::chrono::system_clock::now();

    //graph->printOut();

    graph->tagInplaceNodes();

    graph->getVariableSpace()->putVariable(658,0, t);
    graph->getVariableSpace()->putVariable(659,0, u);
    graph->getVariableSpace()->putVariable(660,0, v);
    auto timePrepare = std::chrono::system_clock::now();

    // validating graph now
    auto status = GraphExecutioner::execute(graph);
    auto timeEnd = std::chrono::system_clock::now();
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(1620));

    auto array = graph->getVariableSpace()->getVariable(1620)->getNDArray();
    ASSERT_EQ(z, *array);

    auto deltaLoad = std::chrono::duration_cast<std::chrono::microseconds>(timeFileLoad - timeStart).count();
    auto deltaInOut = std::chrono::duration_cast<std::chrono::microseconds>(timeInOutLoad - timeFileLoad).count();
    auto deltaPrepare = std::chrono::duration_cast<std::chrono::microseconds>(timePrepare- timeInOutLoad).count();
    auto deltaExec = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeInOutLoad).count();
    auto deltaTotal = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();

    nd4j_printf("Total time         : %lld us;\n", deltaTotal);
    nd4j_printf("\tLoading graph    : %lld us;\n", deltaLoad);
    nd4j_printf("\tLoading in/out   : %lld us;\n", deltaInOut);
    nd4j_printf("\tPreparing intputs: %lld us;\n", deltaPrepare);
    nd4j_printf("\tExecuting graph   : %lld us;\n", deltaExec);
    delete graph;
}
