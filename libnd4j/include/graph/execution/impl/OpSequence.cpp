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

#include <graph/execution/OpSequence.h>
#include <ops/declarable/OpRegistrator.h>
#include <helpers/StringUtils.h>

namespace sd {
namespace graph {
OpSequence::OpSequence(const int deviceId) : _deviceId(deviceId) {
  //
}

OpSequence::OpSequence(const std::vector<ExecutionTask> &ops,
                       const int deviceId) {
  _deviceId = deviceId;
  for (const auto &v : ops) _ops.emplace_back(v);
}

OpSequence::OpSequence(const OpSequence &other) noexcept {
  _ops.clear();

  for (const auto &v : other._ops) _ops.emplace_back(v);

  _idToIndex = other._idToIndex;
  _indexToId = other._indexToId;
}

////////////////////////////////////////////////////////////////////////
// move constructor
OpSequence::OpSequence(OpSequence &&other) noexcept: _ops(std::move(other._ops))  {
  _idToIndex = std::move(other._idToIndex);
  _indexToId = std::move(other._indexToId);
}

OpSequence &OpSequence::operator=(OpSequence &&other) noexcept {
  if (this == &other) return *this;

  _ops = std::move(other._ops);
  _idToIndex = std::move(other._idToIndex);
  _indexToId = std::move(other._indexToId);

  return *this;
}

OpSequence &OpSequence::operator=(const OpSequence &other) noexcept {
  if (this == &other) return *this;

  _ops.clear();
  for (const auto &v : other._ops) _ops.emplace_back(v);

  _idToIndex = other._idToIndex;
  _indexToId = other._indexToId;

  return *this;
}

void OpSequence::printOut() const {
  for (const auto &v : _ops) v.printOut();
}

int OpSequence::deviceId() const { return _deviceId; }

const ExecutionTask &OpSequence::at(uint64_t index) const {
  return _ops[index];
}

ExecutionTask &OpSequence::at(uint64_t index) {
  return _ops[index];
}

const ExecutionTask &OpSequence::operator[](uint64_t index) const {
  return at(index);
}

ExecutionTask &OpSequence::operator[](uint64_t index) {
  return at(index);
}

uint64_t OpSequence::length() const { return _ops.size(); }

void OpSequence::append(const Node &node,
                        const sd::graph::ContextPrototype &ctx) {
  ExecutionTask task(node, ctx);
  append(task);
}

void OpSequence::append(const ExecutionTask& task) {
  _ops.emplace_back(task);

  // updating dictionaries
  auto index = _ops.size() - 1;
  _idToIndex[task.node().id()] = index;
  _indexToId[index] = task.node().id();
}

void OpSequence::append(ExecutionTask&& task) {
  _ops.emplace_back(std::move(task));

  // updating dictionaries
  auto index = _ops.size() - 1;
  _idToIndex[task.node().id()] = index;
  _indexToId[index] = task.node().id();
}

void OpSequence::append(const OpSequence &sequence) {
  for (const auto &v:sequence._ops) {
    this->append(v);
  }
}

int OpSequence::nodeId(int index) const {
  if (index < 0 || index >= _ops.size() || _indexToId.count(index) < 1)
    throw std::runtime_error("Out-of-size index requested: " + StringUtils::valueToString(index));

  return _indexToId.at(index);
}

int OpSequence::nodeIndex(int id) const {
  if ( _idToIndex.count(id) < 1)
    throw std::runtime_error("Unknown Node ID requested: " + StringUtils::valueToString(id));

  return _idToIndex.at(id);
}

bool OpSequence::hasNode(int id) const {
  return _idToIndex.count(id) > 0;
}

OpSequence::iterator OpSequence::begin() {
  return OpSequence::iterator(*this, 0);
}

OpSequence::iterator OpSequence::end() {
  return OpSequence::iterator(*this, length());
}

OpSequence::iterator::iterator(OpSequence &container, uint64_t index)
    : _container(container), _position(index) {
  //
}

const ExecutionTask &OpSequence::iterator::operator*() const {
  return _container._ops[_position];
}

OpSequence::iterator &OpSequence::iterator::operator++() {
  _position++;
  return *this;
}

OpSequence::iterator &OpSequence::iterator::operator++(int inc) {
  return ++(*this);
}

bool OpSequence::iterator::operator!=(const OpSequence::iterator &other) const {
  return _position != other._position;
}

Nd4jStatus OpSequence::wait() const {
  // TODO: to be implemented
  return Status::OK();
}
}  // namespace graph
}  // namespace sd
