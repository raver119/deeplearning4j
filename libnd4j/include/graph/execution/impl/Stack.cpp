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

#include <graph/execution/Stack.h>

namespace sd {
namespace graph {

Stack::Stack(const VariableProxy &root) {
  _frames.push_back(StackFrame(const_cast<VariableProxy&>(root), _counter++, -1, 0));
}

const VariableProxy &Stack::rootVariableSpace() const {
  return _frames.front().variableProxy();
}

StackFrame &Stack::back() {
  return _frames.back();
}

StackFrame &Stack::front() {
  return _frames.front();
}

StackFrame &Stack::root() {
  return _frames.front();
}

void Stack::openFrame(int frameId, int enterId) {
  _frames.emplace_back(StackFrame(_frames.back().variableProxy(), _counter++, frameId, enterId, _frames.back()));
  nd4j_printf("Opening frame [%i], parent: [%i]\n", _frames.back().id(), _frames.back().parent().id());
}

void Stack::iterateFrame(int frameId, int enterId) {
  auto &current = this->back();
  auto &parent = current.parent();
  _frames.emplace_back(StackFrame(_frames.back().variableProxy(), _counter++, frameId, enterId, parent));
  nd4j_printf("Iterating frame, parent: [%i]\n", parent.id());
}

void Stack::closeFrame() {
  // we should remove all frames untl we hit parent frame
  auto &parent = this->back().parent();

  nd4j_printf("Collapsed frame [%i], parent: [%i]\n", this->back().id(), parent.id());

  while (!_frames.empty()) {
    auto &current = this->back();

    // if ID's match - we'll stop
    if (current.id() == parent.id())
      break;

    _frames.pop_back();
  }
}

} // namespace graph
} // namespace sd
