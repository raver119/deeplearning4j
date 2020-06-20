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

#ifndef SD_STACKFRAME_H_
#define SD_STACKFRAME_H_

#include <system/dll.h>
#include <graph/VariableProxy.h>
#include <string>

namespace sd {
namespace graph {

class SD_EXPORT StackFrame {
 private:
  int _id;
  VariableProxy _proxy;
  StackFrame *_parent = nullptr;

  MAP_IMPL<int, int> _disabledNodes;

  // these fields are used
  int _frameId = -119;
  int _enterId = -119;
  mutable int _rewindId = -119;
  mutable int _exitId = -119;
 public:
  explicit StackFrame(const VariableProxy &proxy, int id, int frameId, int enterId);
  explicit StackFrame(const VariableProxy &proxy, int id, int frameId, int enterId, StackFrame &parent);
  ~StackFrame() = default;

  const VariableProxy& variableProxy() const { return _proxy; }



  void disableNode(int nodeId);
  bool isDisabled(int nodeId) const;

  int frameId() const;
  int enterId() const;
  int exitId() const;
  int rewindId() const;

  void setRewindId(int id) const;
  void setExitId(int id) const;

  /**
   * This method returns parent frame
   * @return
   */
  StackFrame& parent() const;

  int id() const { return _id; }
};

} // namespace graph
} // namespace sd

#endif // SD_STACKFRAME_H_
