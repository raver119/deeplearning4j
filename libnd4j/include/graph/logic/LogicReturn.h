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
// Created by raver119 on 28.10.2017.
//

#ifndef SD_LOGICRETURN_H
#define SD_LOGICRETURN_H

#include <graph/Graph.h>
#include <graph/Node.h>
#include <system/pointercast.h>

namespace sd {
namespace graph {
/**
 * This class is responsible for execution logic of Return logical abstraction
 *
 * Basically we're just transferring input variable(s) to output variable(s),
 * nothing beyond that
 * @tparam T
 */
class LogicReturn {
 public:
  static Nd4jStatus processNode(const Node* node);
};

}  // namespace graph
}  // namespace sd

#endif  // SD_LOGICRETURN_H
