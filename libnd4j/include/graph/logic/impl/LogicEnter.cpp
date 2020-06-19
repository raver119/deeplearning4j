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
    stack.openFrame(node->frameId(), node->id());

    // since this is the loop entrance, we'll rewind to this Node once iteration ends
    // Enter -> Merge -> NextIteration
  }

  const auto &frame = stack.back();

  const auto &inputs = node->inputs();
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