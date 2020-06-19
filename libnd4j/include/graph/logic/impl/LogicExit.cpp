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

#include <graph/logic/LogicExit.h>

namespace sd {
namespace graph {

/**
 * This funciton does 2 things:
 * - Propagates input Variable to outer StackFrame
 * - closes current StackFrame (only if this is the last Exit node in this loop)
 */
Nd4jStatus LogicExit::processNode(const Node *node, Stack &stack, const OptimizedGraph& graph) {
  throw std::runtime_error("LogicExit::processNode - Not implemented yet");
}

}  // namespace graph
}  // namespace sd