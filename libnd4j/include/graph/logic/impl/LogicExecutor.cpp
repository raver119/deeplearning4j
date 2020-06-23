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

#include <graph/logic/LogicEnter.h>
#include <graph/logic/LogicExecutor.h>
#include <graph/logic/LogicExit.h>
#include <graph/logic/LogicLoopCond.h>
#include <graph/logic/LogicMerge.h>
#include <graph/logic/LogicNextIteration.h>
#include <graph/logic/LogicSwitch.h>

namespace sd {
namespace graph {
Nd4jStatus LogicExecutor::processNode(const Node *node, Stack &stack, const OptimizedGraph& graph) {
  switch (node->opNum()) {
    case sd::logic::Switch:
      return LogicSwitch::processNode(node, stack, graph);
    case sd::logic::Merge:
      return LogicMerge::processNode(node, stack, graph);
    case sd::logic::LoopCond:
      return LogicLoopCond::processNode(node, stack, graph);
    case sd::logic::NextIteration:
      return LogicNextIeration::processNode(node, stack, graph);
    case sd::logic::Exit:
      return LogicExit::processNode(node, stack, graph);
    case sd::logic::Enter:
      return LogicEnter::processNode(node, stack, graph);
  }

  if (node->name().empty()) {
    nd4j_printf("Unknown LogicOp used at node [%i]: [%i]\n", node->id(),
                node->opNum());
  } else {
    nd4j_printf("Unknown LogicOp used at node [%i:<%s>]: [%i]\n", node->id(),
                node->name().c_str(), node->opNum());
  }
  return ND4J_STATUS_BAD_INPUT;
}
}  // namespace graph
}  // namespace sd