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

#include <array/NDArray.h>
#include <flatbuffers/flatbuffers.h>
#include <graph/Graph.h>
#include <graph/GraphUtils.h>
#include <graph/Node.h>
#include <graph/generated/graph_generated.h>
#include <graph/generated/node_generated.h>
#include <ops/declarable/DeclarableOp.h>

#include <ops/declarable/generic/parity_ops.cpp>

#include "testlayers.h"

using namespace sd;
using namespace sd::graph;

class GraphTests : public testing::Test {
 public:
  /*
  int cShape[] = {2, 2, 2, 2, 1, 0, 1, 99};
  int fShape[] = {2, 2, 2, 1, 2, 0, 1, 102};
   */
  GraphTests() {
    // Environment::getInstance().setDebug(true);
    // Environment::getInstance().setVerbose(true);
  }
};

/*
TEST_F(GraphTests, Test_Minifier_1) {
    // run preprocessor to produce single header
    // if all ok - return value is 0, if error - non-zero value will be returned
    std::string input("../include/ops/ops.h");
//declarable/CustomOperations.h");

    ASSERT_EQ(0, GraphUtils::runPreprocessor(input.c_str(),
"libnd4j_mini.hpp"));
    // remove file from filesystem
#ifdef __linux__
    ASSERT_EQ(0, unlink("libnd4j_mini.hpp"));
#endif
}
*/

TEST_F(GraphTests, Test_Minifier_2) {
  // run preprocessor to produce single header
  // if all ok - return value is 0, if error - non-zero value will be returned
  ASSERT_EQ(0, GraphUtils::runPreprocessor("../include/ops/specials.h",
                                           "libnd4j_mini2.hpp"));
  // remove file from filesystem
#ifdef __linux__
  ASSERT_EQ(0, unlink("libnd4j_mini2.hpp"));
#endif
}

TEST_F(GraphTests, Test_Minifier_3) {
  // run preprocessor to produce single header
  // if all ok - return value is 0, if error - non-zero value will be returned
#ifdef __linux__
  ASSERT_EQ(0x100, GraphUtils::runPreprocessor("/include/ops/ops.h",
                                               "libnd4j_mini3.hpp"));
#endif
  // remove file from filesystem
  // ASSERT_EQ(0, unlink("libnd4j_mini3.hpp"));
}
