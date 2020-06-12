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

#include <graph/VariableProxy.h>
#include <system/dll.h>

namespace sd {
namespace graph {

VariableProxy::VariableProxy(const VariableSpace *ref) {
  if (ref == nullptr) _backed = new VariableSpace();

  _backed = ref;
}

VariableProxy::~VariableProxy() { }

int VariableProxy::numberOfPlaceholders() const {
  return _backed->numberOfPlaceholders();
}

const std::vector<std::shared_ptr<Variable>> &VariableProxy::placeholders()
    const {
  return _backed->placeholders();
}

bool VariableProxy::hasExternalVariable(int it) const {
  return _backed->hasExternalVariable(it);
}

bool VariableProxy::hasExternalVariable(const std::pair<int, int> &pair) const {
  return _backed->hasExternalVariable(pair);
}

bool VariableProxy::hasExternalVariable(const std::string &symbol) const {
  return _backed->hasExternalVariable(symbol);
}

bool VariableProxy::hasVariable(int id) const {
  return _current.hasVariable(id) || _backed->hasVariable(id);
}

bool VariableProxy::hasVariable(int id, int idx) const {
  return _current.hasVariable(id, idx) || _backed->hasVariable(id, idx);
}

bool VariableProxy::hasVariable(const std::pair<int, int> &pair) const {
  return _current.hasVariable(pair) || _backed->hasVariable(pair);
}

void VariableProxy::dropVariable(const std::pair<int, int> &pair) {
  dropVariable(pair.first, pair.second);
}

void VariableProxy::dropVariable(int id, int idx) {
  assert(_current.hasVariable(id, idx));

  _current.dropVariable(id, idx);
}

std::vector<std::shared_ptr<Variable>> VariableProxy::variables() const {
  std::vector<std::shared_ptr<Variable>> result;

  auto b = _backed->variables();
  auto c = _current.variables();

  for (auto v : b) result.emplace_back(v);

  for (auto v : c) result.emplace_back(v);

  return result;
}

bool VariableProxy::hasVariable(const std::string &symbol) const {
  return _current.hasVariable(symbol) || _backed->hasVariable(symbol);
}

std::shared_ptr<Variable> VariableProxy::getVariable(int id) const {
  if (_current.hasVariable(id)) return _current.getVariable(id);

  if (_backed->hasVariable(id)) return _backed->getVariable(id);

  nd4j_printf("Unable to get Variable from proxy: [%i]\n", id);
  throw std::runtime_error("Bad arguments");
}

std::shared_ptr<Variable> VariableProxy::getVariable(int id, int idx) const {
  if (_current.hasVariable(id, idx)) return _current.getVariable(id, idx);

  if (_backed->hasVariable(id, idx)) return _backed->getVariable(id, idx);

  nd4j_printf("Unable to get Variable from proxy: [%i:%i]\n", id, idx);
  throw std::runtime_error("Bad arguments");
}

std::shared_ptr<Variable> VariableProxy::getVariable(
    const std::pair<int, int> &pair) const {
  if (_current.hasVariable(pair)) return _current.getVariable(pair);

  if (_backed->hasVariable(pair)) return _backed->getVariable(pair);

  nd4j_printf("Unable to get Variable from proxy: [%i:%i]\n", pair.first,
              pair.second);
  throw std::runtime_error("Bad arguments");
}

std::shared_ptr<Variable> VariableProxy::getVariable(
    const std::string &symbol) const {
  if (_current.hasVariable(symbol)) return _current.getVariable(symbol);

  if (_backed->hasVariable(symbol)) return _backed->getVariable(symbol);

  nd4j_printf("Unable to get Variable from proxy: [%s]\n", symbol.c_str());
  throw std::runtime_error("Bad arguments");
}

void VariableProxy::replaceVariable(std::shared_ptr<Variable> variable) {
  if (!variable->getName().empty()) {
    // if variable has name defined - we should resolve it via backing var space
    if (_backed->hasVariable(variable->getName())) {
      auto origVar = _backed->getVariable(variable->getName());
      variable->setId(origVar->id(), origVar->index());
      _current.replaceVariable(variable);
    } else
      _current.replaceVariable(variable);
  } else  // if proxy has variable - that's one story
    _current.replaceVariable(variable);
}

std::shared_ptr<Variable> VariableProxy::putVariable(const std::string &name,
                                                     int id, int idx,
                                                     const NDArray &array) {
  return _current.putVariable(name, id, idx, array);
}

void VariableProxy::putOutputVariable(std::shared_ptr<Variable> variable) {
  _current.putOutputVariable(variable);
}

std::shared_ptr<Variable> VariableProxy::putVariable(
    const std::pair<int, int> &pair, const NDArray &array) {
  return _current.putVariable(pair, array);
}

const MAP_IMPL<std::pair<int, int>, std::shared_ptr<Variable>> &VariableProxy::externalPaired() const {
  return _backed->_paired;
}

void VariableProxy::putVariable(const std::pair<int, int> &pair,
                                const std::shared_ptr<Variable> &variable) {
  _current.putVariable(pair, variable);
}

void VariableProxy::putVariable(int id,
                                const std::shared_ptr<Variable> &variable) {
  _current.putVariable(id, variable);
}

std::shared_ptr<Variable> VariableProxy::putVariable(int id,
                                                     const NDArray &array) {
  return _current.putVariable(id, array);
}

std::shared_ptr<Variable> VariableProxy::putVariable(int id, int idx,
                                                     const NDArray &array) {
  return _current.putVariable(id, idx, array);
}

void VariableProxy::putVariable(const std::string &name, int id, int idx,
                                const std::shared_ptr<Variable> &array) {
  _current.putVariable(name, id, idx, array);
}

Stash *VariableProxy::stash() const { return _current.stash(); }

Nd4jLong VariableProxy::externalMemory() const {
  return _backed->externalMemory() + _current.externalMemory();
}

Nd4jLong VariableProxy::internalMemory() const {
  return _backed->internalMemory() + _current.internalMemory();
}

Nd4jLong VariableProxy::totalMemory() const {
  return _backed->totalMemory() + _current.totalMemory();
}

int VariableProxy::externalEntries() const {
  return _backed->externalEntries() + _current.externalEntries();
}

int VariableProxy::internalEntries() const {
  return _backed->internalEntries() + _current.internalEntries();
}

int VariableProxy::totalEntries() const {
  return _backed->totalEntries() + _current.totalEntries();
}

VariableSpace &VariableProxy::operator=(const VariableSpace &other) {
  if (this == &other) return *this;

  nd4j_printf("VariableProxy = not implemented\n", "");

  return *this;
}

void VariableProxy::pullFrom(const VariableProxy &proxy) {
  for (const auto &v:proxy._current.variables()) {
    _current.replaceVariable(v);
  }
}

void VariableProxy::pushTo(VariableProxy &proxy) const {
  for (const auto &v:_current.variables()) {
    proxy._current.replaceVariable(v);
  }
}

}  // namespace graph
}  // namespace sd
