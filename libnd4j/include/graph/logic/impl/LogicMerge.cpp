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
#include <helpers/StringUtils.h>
#include <graph/logic/LogicUtilities.h>

namespace sd {
namespace graph {

static bool isNextIterationCase(const OptimizedGraph& graph, int firstId, int secondId) {
  const auto firstNode = graph.nodesMap().count(firstId) > 0  ? &graph.nodesMap().at(firstId) : nullptr;
  const auto secondNode = graph.nodesMap().count(secondId) > 0 ? &graph.nodesMap().at(secondId) : nullptr;

  return (firstNode != nullptr && firstNode->opType() == OpType_LOGIC && firstNode->opNum() == sd::logic::NextIteration) ||  (secondNode != nullptr && secondNode->opType() == OpType_LOGIC && secondNode->opNum() == sd::logic::NextIteration);
}

static bool checkViability(Stack &stack, const std::pair<int, int> &first, const std::pair<int, int> &second) {
  auto &frame = stack.back();
  auto &varSpace = const_cast<VariableProxy&>(frame.variableProxy());

  return (!frame.isDisabled(first.first) && varSpace.hasVariable(first) && varSpace.getVariable(first)->hasNDArray()) || (!frame.isDisabled(second.first) && varSpace.hasVariable(second) && varSpace.getVariable(second)->hasNDArray());
}

Nd4jStatus LogicMerge::processNode(const Node *node, Stack &stack, const OptimizedGraph& graph) {
  // getting current frame first
  auto &frame = stack.back();

  const auto &inputs = node->inputs();
  auto &varSpace = const_cast<VariableProxy&>(frame.variableProxy());

  REQUIRE_TRUE(inputs.size() == 2, 0, "Merge: op expects exactly 2 inputs, but only %i defined", (int) inputs.size());

  // if both inputs are unavailable - this node is disabled and must be disabled
  if (!checkViability(stack, inputs[0], inputs[1])) {
    nd4j_printf("Both inputs absent, skipping\n", "");
    LogicUtilities::disableBranch(frame, varSpace, graph, node);
    return Status::OK();
  }

  if (frame.isDisabled(inputs[0].first) && frame.isDisabled(inputs[1].first)) {
    REQUIRE_TRUE(false, 0, "Merge: only 1 input should be disabled, but got both of them down");
  }



  // if we're on NextIteration merge, we'll propagate its output regardless of first arg existence
  if (isNextIterationCase(graph, inputs[0].first, inputs[1].first)) {
    const auto firstNode = graph.nodesMap().count(inputs[0].first) > 0  ? &graph.nodesMap().at(inputs[0].first) : nullptr;
    const auto secondNode = graph.nodesMap().count(inputs[1].first) > 0 ? &graph.nodesMap().at(inputs[1].first) : nullptr;

    if (firstNode != nullptr && firstNode->opType() == OpType_LOGIC && firstNode->opNum() == sd::logic::NextIteration) {
      // we must check, if NextIteration Node already was executed. Or, pick initial value first
      auto id = varSpace.hasVariable(inputs[0]) && varSpace.getVariable(inputs[0])->hasNDArray() ? inputs[0] : inputs[1];
      if (!varSpace.hasVariable(id) || !varSpace.getVariable(id)->hasNDArray())
        throw std::runtime_error("Non-existent NDArray requested: [" + StringUtils::valueToString(id) +"]");

      varSpace.putVariable({node->id(), 0}, *varSpace.getVariable(id)->getNDArray());
    } else {
      // we must check, if NextIteration Node already was executed. Or, pick initial value first
      auto id = varSpace.hasVariable(inputs[1]) && varSpace.getVariable(inputs[1])->hasNDArray() ? inputs[1] : inputs[0];
      if (!varSpace.hasVariable(id) || !varSpace.getVariable(id)->hasNDArray())
        throw std::runtime_error("Non-existent NDArray requested: [" + StringUtils::valueToString(id) +"]");

      varSpace.putVariable({node->id(), 0}, *varSpace.getVariable(id)->getNDArray());
    }
  } else {
    // we're getting first non-disabled input and propagate it
    const auto &p = frame.isDisabled(inputs[0].first) ? inputs[1] : inputs[0];

    REQUIRE_TRUE(frame.variableProxy().hasVariable(p), 0, "Merge: Variable [%i:%i] doesn't exist", p.first, p.second);

    varSpace.putVariable({node->id(), 0}, *varSpace.getVariable(p)->getNDArray());
  }

  return Status::OK();
}

}  // namespace graph
}  // namespace sd
