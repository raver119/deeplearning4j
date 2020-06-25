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

#include <graph/logic/LogicUtilities.h>

namespace sd {
namespace graph {

void LogicUtilities::disableBranch(StackFrame &frame, VariableProxy &varSpace, const OptimizedGraph &graph, const Node* node) {
  const auto &outputs = node->outputs();

  // we're going to disable certain external variables, if they depend on a current disabled node
  // FIXME: it can be done in a better way rather than O(n^2)
  for (const auto &var: varSpace.externalPaired()) {
    for (const auto &d: var.second->dependencies()) {
      if (d.first == node->id())
        frame.disableNode(var.second->id());
    }
  }

  // we're going to roll through all consumers
  for (const auto &o:outputs) {
    if (graph.nodesMap().count(o.first) == 0)
      throw std::runtime_error("pew-pew");

    // now fetch disabled node
    const auto &n = graph.nodesMap().at(o.first);

    // edge case here: don't disable Merge node
    if (n.opType() == OpType_LOGIC && n.opNum() == sd::logic::Merge)
      continue;

    // disable each consumer
    frame.disableNode(o.first);

    // do recursive magic
    disableBranch(frame, varSpace, graph, &n);
  }
}

void LogicUtilities::disableBranch(StackFrame &frame, VariableProxy &varSpace, const OptimizedGraph &graph, const Node* node, bool branchToDisable) {
  const auto &outputs = node->outputs();
  int second = branchToDisable ? 1 : 0;

  for (const auto &o:outputs) {
    if (o.second == second) {
      frame.disableNode(o.first);

      if (graph.nodesMap().count(o.first) == 0)
        throw std::runtime_error("pew-pew");

      const auto &n = graph.nodesMap().at(o.first);

      disableBranch(frame, varSpace, graph, &n);
    }
  }
}

}
}
