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

class ExecutionLayerTests : public testing::Test {
 public:
  ExecutionLayerTests() {
    ///
  }
};

TEST_F(ExecutionLayerTests, test_reassign_1) {
  ExecutionLayer layer;
  OpSequence sequence1, sequence2;

  Node a(sd::ops::add(), "add");
  Node m(sd::ops::multiply(), "mul");
  Node d(sd::ops::divide(), "div");

  Context ctx1(1);
  Context ctx2(2);
  Context ctx3(3);

  sequence1.append(a, ctx1);
  sequence2.append(m, ctx2);
  sequence2.append(d, ctx3);

  layer.append(sequence1);
  layer.append(sequence2);

  auto seq = layer[0];
  ASSERT_EQ(1, seq.length());

  seq = layer[1];
  ASSERT_EQ(2, seq.length());
}
