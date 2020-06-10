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
// @author raver119@gmail.com
//

#include <graph/VariableSpace.h>
#include <legacy/NativeOps.h>

namespace sd {
namespace graph {
Stash *VariableSpace::stash() const { return const_cast<Stash *>(&_stash); }

void VariableSpace::injectVariable(const std::pair<int, int> &pair,
                                   std::shared_ptr<Variable> variable) {
  if (pair.second == 0) {
    this->_variables[pair.first] = variable;
  }

  if (!variable->getName().empty())
    this->_symbolic[variable->getName()] = variable;

  this->_paired[pair] = variable;
}

const std::vector<std::shared_ptr<Variable>> &VariableSpace::placeholders()
    const {
  return _placeholders;
}

int VariableSpace::numberOfPlaceholders() const { return _placeholders.size(); }

bool VariableSpace::hasVariable(const std::string &symbol) const {
  return _symbolic.count(symbol) > 0;
}

std::shared_ptr<Variable> VariableSpace::getVariable(
    const std::string &symbol) const {
  return _symbolic.at(symbol);
}

bool VariableSpace::hasVariable(int id, int index) const {
  std::pair<int, int> pair(id, index);
  return hasVariable(pair);
}

bool VariableSpace::hasExternalVariable(int id) const {
  if (!hasVariable(id)) return false;

  auto var = getVariable(id);
  return var->isExternal();
}

bool VariableSpace::hasExternalVariable(const std::pair<int, int> &pair) const {
  if (!hasVariable(pair)) return false;

  auto var = getVariable(pair);
  return var->isExternal();
}

bool VariableSpace::hasExternalVariable(const std::string &symbol) const {
  if (!hasVariable(symbol)) return false;

  auto var = getVariable(symbol);
  return var->isExternal();
}

std::shared_ptr<Variable> VariableSpace::getVariable(int id, int index) const {
  std::pair<int, int> pair(id, index);
  return getVariable(pair);
}

std::shared_ptr<Variable> VariableSpace::getVariable(
    const std::pair<int, int> &pair) const {
  if (pair.first < 0)
    return getVariable(pair.first);
  else
    return _paired.at(pair);

  nd4j_printf("Unknown variable requested: [%i,%i]\n", pair.first, pair.second);
  throw std::runtime_error("Unknown variable requested");
}

bool VariableSpace::hasVariable(int id) const {
  return _variables.count(id) > 0;
}

bool VariableSpace::hasVariable(const std::pair<int, int> &id) const {
  return _paired.count(id) > 0;
}

void VariableSpace::putOutputVariable(std::shared_ptr<Variable> variable) {
  // putVariable(_auto_counter--, variable);
  putVariable(variable->id(), variable);
}

int VariableSpace::externalEntries() const { return _external.size(); }

int VariableSpace::internalEntries() const { return _internal.size(); }

int VariableSpace::totalEntries() const {
  return externalEntries() + internalEntries();
}

Nd4jLong VariableSpace::externalMemory() const {
  Nd4jLong size = 0;
  for (auto n : _external) {
    size += n->getNDArray()->memoryFootprint();
  }

  return size;
}

std::vector<std::shared_ptr<Variable>> VariableSpace::variables() const {
  std::vector<std::shared_ptr<Variable>> result;

  for (auto v : _internal) result.emplace_back(v);

  for (auto v : _external) result.emplace_back(v);

  return result;
}

Nd4jLong VariableSpace::internalMemory() const {
  Nd4jLong size = 0;
  for (auto n : _internal) {
    size += n->getNDArray()->memoryFootprint();
  }

  return size;
}

Nd4jLong VariableSpace::totalMemory() const {
  return externalMemory() + internalMemory();
}

std::shared_ptr<Variable> VariableSpace::putVariable(
    int id, int idx, const std::shared_ptr<NDArray> &array) {
  auto variable = std::make_shared<Variable>(array, "", id, idx);
  this->putVariable({id, idx}, variable);
  return variable;
}

std::shared_ptr<Variable> VariableSpace::putVariable(const std::string &name,
                                                     int id, int idx,
                                                     const NDArray &array) {
  auto variable = std::make_shared<Variable>(array, name, id, idx);
  this->putVariable({id, idx}, variable);
  return variable;
}

void VariableSpace::dropVariable(const std::string &pair) {
  throw std::runtime_error("VariableSpace::dropVariable - not implemented yet");
}

std::shared_ptr<Variable> VariableSpace::putVariable(
    const std::pair<int, int> &pair, const NDArray &array) {
  auto variable =
      std::make_shared<Variable>(array, "", pair.first, pair.second);
  this->putVariable(pair, variable);
  return variable;
}

std::shared_ptr<Variable> VariableSpace::putVariable(int node, int idx,
                                                     const NDArray &array) {
  std::pair<int, int> pair(node, idx);
  return this->putVariable(pair, array);
}

void VariableSpace::putVariable(const std::string &name, int node, int idx,
                                const std::shared_ptr<Variable> &variable) {
  std::pair<int, int> pair(node, idx);
  variable->setName(name);
  this->putVariable(pair, variable);
}

void VariableSpace::silentPutVariable(
    const std::pair<int, int> &pair,
    const std::shared_ptr<Variable> &variable) {
  std::lock_guard<std::mutex> lock(_varmap);

  _paired[pair] = variable;
}

void VariableSpace::putVariable(const std::pair<int, int> &pair,
                                const std::shared_ptr<Variable> &variable) {
  silentPutVariable(pair, variable);

  if (variable->isPlaceholder()) _placeholders.emplace_back(variable);

  // copying duplicate for compatibility
  if (pair.second == 0 && !this->hasVariable(pair.first)) {
    this->putVariable(pair.first, variable);
  }

  if (!variable->getName().empty()) {
    _symbolic[variable->getName()] = variable;
  }
}

void VariableSpace::putVariable(int id,
                                const std::shared_ptr<Variable> &variable) {
  // we don't want to add variables more then once
  if (_variables.count(id) > 0) {
    throw std::runtime_error("VariableSpace::putVariable - duplicate found");
  }

  {
    std::lock_guard<std::mutex> lock(_varmap);

    if (_auto_counter >= id) _auto_counter = id - 1;

    variable->setId(id);

    if (!variable->getName().empty()) {
      // std::pair<std::string, Variable *> pair(*(variable->getName()),
      // variable);
      _symbolic[variable->name()] = variable;
    }

    // we have special list for external variables to ensure graph completeness
    if (id < 0) {
      _external.emplace_back(variable);
    } else {
      _internal.emplace_back(variable);
    }

    _variables[id] = variable;
  }

  std::pair<int, int> pair(id, 0);
  if (!hasVariable(pair)) {
    this->silentPutVariable(pair, variable);

    if (variable->isPlaceholder()) _placeholders.emplace_back(variable);
  }
}

std::shared_ptr<Variable> VariableSpace::putVariable(int id,
                                                     const NDArray &array) {
  auto var = std::make_shared<Variable>(array, "", id, 0);
  this->putVariable(id, var);
  return var;
}

std::shared_ptr<Variable> VariableSpace::getVariable(int id) const {
  return _variables.at(id);
}

VariableSpace::~VariableSpace() {
  //
}

VariableSpace::VariableSpace(const VariableSpace &other) {
  _stash = other._stash;

  _paired = other._paired;
  _symbolic = other._symbolic;
  _variables = other._variables;

  _external = other._external;
  _internal = other._internal;

  _lists = other._lists;
  _placeholders = other._placeholders;

  _auto_counter = other._auto_counter;
}

VariableSpace::VariableSpace(VariableSpace &&other) {
  _stash = std::move(other._stash);

  _paired = std::move(other._paired);
  _symbolic = std::move(other._symbolic);
  _variables = std::move(other._variables);

  _external = std::move(other._external);
  _internal = std::move(other._internal);

  _lists = std::move(other._lists);
  _placeholders = std::move(other._placeholders);

  _auto_counter = other._auto_counter;
}

VariableSpace &VariableSpace::operator=(VariableSpace &&other) {
  if (this == &other) return *this;

  _stash = std::move(other._stash);

  _paired = std::move(other._paired);
  _symbolic = std::move(other._symbolic);
  _variables = std::move(other._variables);

  _external = std::move(other._external);
  _internal = std::move(other._internal);

  _lists = std::move(other._lists);
  _placeholders = std::move(other._placeholders);

  _auto_counter = other._auto_counter;

  return *this;
}

VariableSpace &VariableSpace::operator=(const VariableSpace &other) {
  if (this == &other) return *this;

  _stash = other._stash;

  _paired = other._paired;
  _symbolic = other._symbolic;
  _variables = other._variables;

  _external = other._external;
  _internal = other._internal;

  _lists = other._lists;
  _placeholders = other._placeholders;

  _auto_counter = other._auto_counter;

  return *this;
}

void VariableSpace::replaceVariable(std::shared_ptr<Variable> variable) {
  bool replaced = false;
  // trying name lookup first
  if (!variable->getName().empty()) {
    if (hasVariable(variable->getName())) {
      auto vs = getVariable(variable->getName());
      dropVariable(vs->id(), vs->index());

      putVariable({vs->id(), vs->index()}, variable);

      // if we're on zero index, we also must update index-less reference
      if (vs->index() == 0)
        _variables[vs->id()] = variable;

      replaced = true;
    }
  } else {
    if (hasVariable(variable->id(), variable->index())) {
      auto vs = getVariable(variable->id(), variable->index());
      dropVariable(variable->id(), variable->index());
      putVariable({vs->id(), vs->index()}, variable);

      // if we're on zero index, we also must update index-less reference
      if (vs->index() == 0)
        _variables[vs->id()] = variable;

      replaced = true;
    }
  }

  if (!replaced) {
    putVariable({variable->id(), variable->index()}, variable);
  }
}

void VariableSpace::dropVariable(const std::pair<int, int> &pair) {
  dropVariable(pair.first, pair.second);
}

void VariableSpace::dropVariable(int id, int idx) {}

VariableSpace::VariableSpace() {}
}  // namespace graph
}  // namespace sd