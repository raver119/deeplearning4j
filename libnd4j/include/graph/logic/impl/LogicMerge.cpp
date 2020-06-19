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
// @author raver119@gmail.com
//

#include <graph/Status.h>
#include <graph/logic/LogicMerge.h>

namespace sd {
namespace graph {
Nd4jStatus LogicMerge::processNode(const Node *node, Stack &stack, const OptimizedGraph& graph) {
  // getting current frame first
  auto &frame = stack.back();

  const auto &inputs = node->inputs();
  auto &varSpace = const_cast<VariableProxy&>(frame.variableProxy());

  REQUIRE_TRUE(inputs.size() == 2, 0, "Merge: op expects exactly 2 inputs, but only %i defined", (int) inputs.size());
  if (frame.isDisabled(inputs[0].first) && frame.isDisabled(inputs[1].first)) {
    REQUIRE_TRUE(false, 0, "Merge: only 1 input should be disabled, but got both of them down");
  }

  const auto &firstNode = graph.nodesMap().at(inputs[0].first);
  const auto &secondNode = graph.nodesMap().at(inputs[1].first);

  if ((firstNode.opType() == OpType_LOGIC && firstNode.opNum() == sd::logic::NextIteration)|| (secondNode.opType() == OpType_LOGIC && secondNode.opNum() == sd::logic::NextIteration)) {
    // if we're on NextIteration merge, we'll propagate its output regardless of first arg existence
    if (firstNode.opType() == OpType_LOGIC && firstNode.opNum() == sd::logic::NextIteration) {
      auto id = varSpace.hasVariable(inputs[0]) && varSpace.getVariable(inputs[0])->hasNDArray() ? inputs[0] : inputs[1];
      varSpace.putVariable({node->id(), 0}, *varSpace.getVariable(id)->getNDArray());
    } else {
      auto id = varSpace.hasVariable(inputs[1]) && varSpace.getVariable(inputs[1])->hasNDArray() ? inputs[1] : inputs[0];
      varSpace.putVariable({node->id(), 0}, *varSpace.getVariable(id)->getNDArray());
    }
  } else {
    // we're getting first non-disabled input and propagate it
    const auto &p = frame.isDisabled(inputs[0].first) ? inputs[1] : inputs[0];

    REQUIRE_TRUE(frame.variableProxy().hasVariable(p), 0, "Merge: Variable [%i:%i] doesn't exist", p.first, p.second);

    std::pair<int, int> t(node->id(), 0);
    auto array = varSpace.getVariable(p)->getNDArray().get();
    varSpace.putVariable(t, *array);
  }

  return Status::OK();
}

}  // namespace graph
}  // namespace sd
