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
// @author yves.quemener@gmail.com
//

#include "testlayers.h"
#include <flatbuffers/flatbuffers.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/profiling/GraphProfilingHelper.h>
#include <graph/Node.h>
#include <graph/Graph.h>
#include <graph/GraphUtils.h>
#include <NDArray.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/generic/parity_ops.cpp>
#include <performance/benchmarking/global_timers.h>
#include <iomanip>

using namespace nd4j;
using namespace nd4j::graph;

class ImportTests : public testing::Test {
public:
    /*
    int cShape[] = {2, 2, 2, 2, 1, 0, 1, 99};
    int fShape[] = {2, 2, 2, 1, 2, 0, 1, 102};
     */
    ImportTests() {
//        Environment::getInstance()->setDebug(true);
//        Environment::getInstance()->setVerbose(true);
        Environment::getInstance()->setProfiling(true);
    }
};

TEST_F(ImportTests, LstmMnist) {
        const char* modelFilename = "resources/lstm_mnist.fb";
    auto timeStart = std::chrono::system_clock::now();
//    nd4j_printf("Importing file:", 0);
    auto graph = GraphExecutioner::importFromFlatBuffers(modelFilename);
    auto timeEnd = std::chrono::system_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds> ((timeEnd - timeStart)).count();
//    nd4j_printf(" %d ms\n", dt);
//    nd4j_printf("Building graph\n", 0);
    graph->buildGraph();

    auto placeholders = graph->getPlaceholders();
    auto variablespace = graph->getVariableSpace()->getVariables();
//    nd4j_printf("Placeholders:\n------------\n",0);
//    for(Variable* ph: *placeholders){
//        nd4j_printf("%s\n", ph->getName()->c_str());
//    }
//    nd4j_printf("Variables:\n------------\n",0);
//    for(Variable* var: variablespace){
//        nd4j_printf("%s\n", var->getName()->c_str());
//        if(var->hasNDArray()) {
//            timeStart = std::chrono::system_clock::now();
//            if(!var->getNDArray()->isS())
//            {
//                NDArray scalar = var->getNDArray()->sumNumber();
//                dt = std::chrono::duration_cast<std::chrono::milliseconds> ((std::chrono::system_clock::now() - timeStart)).count();
//                nd4j_printf("Sum=%f, %d ms\n", scalar.e<float>(0), dt);
//            }
//        }
//    }
//    nd4j_printf("Execution:\n------------\n",0);
    int height = 28;
    int width = 28;
    int batchsize = 1;

    NDArray* inputArray = NDArrayFactory::create_<double>('c', {1, height, width});
    Variable* input = new Variable(inputArray, "input");
    graph->getVariableSpace()->replaceVariable(input);

    auto profile = GraphProfilingHelper::profile(graph, 10);
    profile->printOut();
    delete profile;

    Nd4jStatus status = GraphExecutioner::execute(graph);
    std::string outputLayerName = "output";
    NDArray* result = graph->getVariableSpace()->getVariable(&outputLayerName)->getNDArray();
    std::vector<double> rvec = result->getBufferAsVector<double>();
    for(int i=0; i<10; i++)
        nd4j_debug("(%d): %f\n", i, rvec[i]);
    ASSERT_NEAR(rvec[0], 0.046829, 0.0001);
    //0.046829
}

