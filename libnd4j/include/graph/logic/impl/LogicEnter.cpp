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
//  @author raver119@gmail.com
//

#include <graph/Status.h>
#include <graph/logic/LogicEnter.h>

namespace sd {
namespace graph {

/**
 * This function does 2 things:
 * - Propagates input Variable
 * - Opens new StackFrame (only if that's the first Enter in this Loop)
 */
Nd4jStatus LogicEnter::processNode(const Node *node, Stack &stack, const OptimizedGraph& graph) {
  // if current frameName isn't equal to node frame name - we'll open new StackFrame then
  if (node->frameId() != stack.back().frameId()) {
    // since this is the loop entrance, we'll rewind to this Node once iteration ends
    stack.openFrame(node->frameId(), node->id());
  }

  const auto &inputs = node->inputs();
  const auto &outputs = node->outputs();

  // getting current frame (it might be the new one!)
  const auto &frame = stack.back();

  // and we need to find exit point - it has to be Exit node, with max index within OpSequence
  auto currentExitIndex = frame.exitId() >= 0 ? graph.nodeIndex(frame.exitId()) : -1;
  auto thisExitIndex = graph.nodeIndex(node->exitId());

  // we want to exit after the last Exit node
  if (thisExitIndex > currentExitIndex)
    frame.setExitId(node->exitId());


  // we need to find rewind point - it has to be NextIteration node with max index within OpSequence
  const auto &merge = graph.nodesMap().at(outputs[0].first);
  const auto &iter = graph.nodesMap().at(merge.inputs()[1].first);

  // we must compare index of this NextIteration Node within OpSequence to the current one, if it's set
  auto currentRewindIndex = frame.rewindId() >= 0 ? graph.nodeIndex(frame.rewindId()) : -1;
  auto thisRewindIndex = graph.nodeIndex(iter.id());

  // we want to rewind after the last NextIteration node
  if (thisRewindIndex > currentRewindIndex)
    frame.setRewindId(iter.id());

  auto &varSpace = const_cast<VariableProxy&>(frame.variableProxy());

  // validate Node state
  REQUIRE_TRUE(inputs.size() == 1, 0, "Enter: op must have exactly 1 inputs");
  REQUIRE_TRUE(varSpace.hasVariable(inputs[0]), 0, "Enter: input Variable doesn't exist");

  // now we propagate input as own output
  // ssince we've opened new StackFrame, this Variable will end up in new VariableProxy
  auto input = varSpace.getVariable(inputs[0]);
  varSpace.putVariable(std::pair<int, int>{node->id(), 0}, *input->getNDArray());

  return sd::Status::OK();
}

}  // namespace graph
}  // namespace sd