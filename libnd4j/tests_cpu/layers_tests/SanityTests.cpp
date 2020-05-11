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
// Created by raver119 on 13/11/17.
//

#include <graph/Graph.h>
#include <graph/Node.h>
#include <ops/declarable/CustomOperations.h>

#include "testlayers.h"

using namespace sd;
using namespace sd::graph;

class SanityTests : public testing::Test {
 public:
};

TEST_F(SanityTests, VariableSpace_2) {
  VariableSpace variableSpace;
  variableSpace.putVariable(1, NDArrayFactory::create<float>('c', {3, 3}));
  variableSpace.putVariable({1, 1}, NDArrayFactory::create<float>('c', {3, 3}));

  std::pair<int, int> pair(1, 2);
  variableSpace.putVariable(pair, NDArrayFactory::create<float>('c', {3, 3}));
}

TEST_F(SanityTests, Graph_1) {
  Graph graph;

  graph.variableSpace().putVariable(1,
                                    NDArrayFactory::create<float>('c', {3, 3}));
  graph.variableSpace().putVariable({1, 1},
                                    NDArrayFactory::create<float>('c', {3, 3}));

  std::pair<int, int> pair(1, 2);
  graph.variableSpace().putVariable(pair,
                                    NDArrayFactory::create<float>('c', {3, 3}));
}