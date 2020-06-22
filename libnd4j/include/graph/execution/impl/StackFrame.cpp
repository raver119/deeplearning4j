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


#include <graph/execution/StackFrame.h>

namespace sd {
namespace graph {

StackFrame::StackFrame(const VariableProxy &proxy, int id, int frameId, int enterId)
                      : _proxy(proxy), _frameId(frameId), _enterId(enterId), _id(id) { }

StackFrame::StackFrame(const VariableProxy &proxy, int id, int frameId, int enterId, StackFrame &parent)
                      : StackFrame(proxy, id, frameId, enterId) {
  _parent = &parent;
}

void StackFrame::disableNode(int nodeId) {
  _disabledNodes[nodeId] = 1;
}

bool StackFrame::isDisabled(int nodeId) const {
  return _disabledNodes.count(nodeId) > 0;
}

int StackFrame::frameId() const {
  return _frameId;
}

int StackFrame::enterId() const {
  return _enterId;
}

int StackFrame::exitId() const {
  return _exitId;
}

void StackFrame::setExitId(int id) const {
  _exitId = id;
}

int StackFrame::rewindId() const {
  return _rewindId;
}

void StackFrame::setRewindId(int id) const {
  _rewindId = id;
}

StackFrame &StackFrame::parent() const {
  return *_parent;
}

} // namespace graph
} // namespace sd
