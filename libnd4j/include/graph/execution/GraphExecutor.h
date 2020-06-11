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

#ifndef SD_GRAPHEXECUTOR_H
#define SD_GRAPHEXECUTOR_H

#include <graph/OptimizedGraph.h>
#include <graph/VariableProxy.h>
#include <memory/GraphMemoryManager.h>
#include <graph/execution/StackFrame.h>
#include <graph/execution/Stack.h>
#include <deque>
#include <system/dll.h>

namespace sd {
namespace graph {
class Graph;

class SD_EXPORT GraphExecutor {
 protected:
  virtual Context prepareContext(const ContextPrototype &contextPrototype,
                                 VariableProxy &variableSpace,
                                 const GraphMemoryManager &memoryManager) const;

  /*
   * preprocessor call involves:
   * - ensure all inputs reside in HOT memory zone
   * - shape function call
   * - open workspace
   */
  virtual Nd4jStatus preprocess(sd::ops::DeclarableOp *op,
                                Context &context) const;

  /**
   * postporcessor call involves:
   * - remove all inputs that are not going to be used later from HOT memory
   * zone
   * - close workspace
   * @return
   */
  virtual Nd4jStatus postprocess(sd::ops::DeclarableOp *op,
                                 Context *context) const;

 public:
  GraphExecutor() = default;
  virtual ~GraphExecutor() = default;

  /**
   * This method executes OptimizedGraph instance
   * @param graph
   * @return
   */
  virtual Nd4jStatus execute(const OptimizedGraph &graph,
                             VariableProxy &proxy,
                             bool isInference = true) const;

  /**
   * This method executes OpSequence
   * @param seq
   * @param deviceId - this argument allows to override device affinity
   * specified in OpSequence, keep it < 0 to follow OpSequence
   * @return
   */
  virtual Nd4jStatus execute(const OpSequence &seq,
                             const OptimizedGraph &graph,
                             Stack &stack,
                             int deviceId) const;

  /**
   * This method executes given op
   * @param op
   * @param contextPrototype
   * @return
   */
  virtual Nd4jStatus execute(const std::shared_ptr<sd::ops::DeclarableOp> &op,
                             const ContextPrototype &contextPrototype,
                             const OpSequence &sequence,
                             const OptimizedGraph &graph,
                             VariableProxy &proxy,
                             const int deviceId) const;
};
}  // namespace graph
}  // namespace sd

#endif  // SD_GRAPHEXECUTOR_H
