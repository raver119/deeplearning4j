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
// Created by raver119 on 21.10.17.
//

#include <graph/Status.h>
#include <graph/logic/LogicSwitch.h>
#include <system/pointercast.h>

namespace sd {
namespace graph {
Nd4jStatus LogicSwitch::processNode(const Node* node, StackFrame &frame) {
  const auto &inputs = node->inputs();
  const auto &outputs = node->outputs();

  auto &varSpace = const_cast<VariableProxy&>(frame.variableProxy());

  REQUIRE_TRUE(inputs.size() == 2, 0, "Switch: op must have exactly 2 inputs");
  REQUIRE_TRUE(varSpace.hasVariable(inputs[0]), 0, "Switch: input Variable doesn't exist");
  REQUIRE_TRUE(varSpace.hasVariable(inputs[1]), 0, "Switch: condition Variable doesn't exist");

  auto input = varSpace.getVariable(inputs[0]);
  auto boolean = varSpace.getVariable(inputs[1]);

  REQUIRE_TRUE(boolean->hasNDArray(), 0, "Switch: boolean Variable must have NDArray defined");

  if (boolean->getNDArray()->e<bool>(0)) {
    // true branch
    varSpace.putVariable(std::pair<int, int>{node->id(), 1}, *input->getNDArray());
  } else {
    // false branch
    varSpace.putVariable(std::pair<int, int>{node->id(), 0}, *input->getNDArray());
  }

  return Status::OK();
};

}  // namespace graph
}  // namespace sd
