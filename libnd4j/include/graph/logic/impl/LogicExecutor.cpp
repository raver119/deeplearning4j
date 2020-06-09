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
// Created by raver119 on 20.10.2017.
//

#include <graph/logic/LogicConditional.h>
#include <graph/logic/LogicEnter.h>
#include <graph/logic/LogicExecutor.h>
#include <graph/logic/LogicExit.h>
#include <graph/logic/LogicExpose.h>
#include <graph/logic/LogicLoopCond.h>
#include <graph/logic/LogicMerge.h>
#include <graph/logic/LogicNextIteration.h>
#include <graph/logic/LogicReturn.h>
#include <graph/logic/LogicScope.h>
#include <graph/logic/LogicSwitch.h>
#include <graph/logic/LogicWhile.h>

namespace sd {
namespace graph {
Nd4jStatus LogicExecutor::processNode(Graph *graph, Node *node) {
  switch (node->opNum()) {
    case sd::logic::While:
      return LogicWhile::processNode(graph, node);
    case sd::logic::Scope:
      return LogicScope::processNode(graph, node);
    case sd::logic::Conditional:
      return LogicConditional::processNode(graph, node);
    case sd::logic::Switch:
      return LogicSwitch::processNode(graph, node);
    case sd::logic::Return:
      return LogicReturn::processNode(graph, node);
    case sd::logic::Expose:
      return LogicExpose::processNode(graph, node);
    case sd::logic::Merge:
      return LogicMerge::processNode(graph, node);
    case sd::logic::LoopCond:
      return LogicLoopCond::processNode(graph, node);
    case sd::logic::NextIteration:
      return LogicNextIeration::processNode(graph, node);
    case sd::logic::Exit:
      return LogicExit::processNode(graph, node);
    case sd::logic::Enter:
      return LogicEnter::processNode(graph, node);
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