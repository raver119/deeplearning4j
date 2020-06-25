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

#ifndef SD_LOGICUTILITIES_H
#define SD_LOGICUTILITIES_H

#include <system/dll.h>
#include <graph/VariableProxy.h>
#include <graph/OptimizedGraph.h>
#include <graph/execution/StackFrame.h>

namespace sd {
namespace graph {

class SD_EXPORT LogicUtilities {
 public:
  static void disableBranch(StackFrame &frame, VariableProxy &varSpace, const OptimizedGraph &graph, const Node* node, bool branchToDisable);
  static void disableBranch(StackFrame &frame, VariableProxy &varSpace, const OptimizedGraph &graph, const Node* node);
};


}
}

#endif //SD_LOGICUTILITIES_H
