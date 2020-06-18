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

#include <array/ByteOrderUtils.h>
#include <array/DataTypeConversions.h>
#include <array/DataTypeUtils.h>
#include <graph/FlatUtils.h>
#include <graph/Variable.h>
#include <helpers/EnumUtils.h>
#include <helpers/StringUtils.h>

namespace sd {
namespace graph {
Variable::Variable(const NDArrayList &arrayList, const std::string &name, int id, int idx)
  : Variable(std::make_shared<sd::NDArrayList>(arrayList), name, id, idx) { }

Variable::Variable(std::shared_ptr<sd::NDArrayList> list, const std::string &name, int id, int idx) {
  _list = list;
  if (!name.empty()) _name = name;

  _id = id;
  _index = idx;
  _variableType = VariableType::ARRAY_LIST;
}

Variable::Variable(const NDArray &array, const std::string &name, int id,
                   int idx) {
  _ndarray = std::make_shared<NDArray>(array);

  if (!name.empty()) _name = name;

  _id = id;
  _index = idx;
}

Variable::Variable() {
  //
}

void sd::graph::Variable::setIndex(int index) { _index = index; }

bool sd::graph::Variable::hasNDArray() const {
  return _ndarray.get() != nullptr;
}

void sd::graph::Variable::setVariableType(VariableType variableType) {
  _variableType = variableType;
}

bool sd::graph::Variable::hasNDArrayList() const { return _list != nullptr; }

bool sd::graph::Variable::isPlaceholder() const { return _placeholder; }

const std::string &sd::graph::Variable::name() const { return _name; }

const std::string &sd::graph::Variable::getName() const { return _name; }

void sd::graph::Variable::setName(const std::string &name) { _name = name; }

int sd::graph::Variable::id() const { return _id; }

int sd::graph::Variable::index() const { return _index; }

void sd::graph::Variable::setId(int id) { _id = id; }

bool sd::graph::Variable::isEmpty() const {
  if (_variableType == VariableType::NDARRAY)
    return _ndarray == nullptr || !_ndarray->nonNull();
  else if (_variableType == VariableType::ARRAY_LIST)
    return _list == nullptr;

  return false;
}

bool sd::graph::Variable::isExternal() const { return _external; }

bool sd::graph::Variable::isReadOnly() const { return _readOnly; }

void sd::graph::Variable::markExternal(bool reallyExternal) {
  this->_external = reallyExternal;
}

void sd::graph::Variable::markRemovable(bool reallyRemovable) {
  if (!reallyRemovable) nd4j_debug("", "");
  this->_removable = reallyRemovable;
}

void sd::graph::Variable::markReadOnly(bool reallyReadOnly) {
  this->_readOnly = reallyReadOnly;
}

std::shared_ptr<sd::NDArray> sd::graph::Variable::getNDArray() const {
  if (_variableType != VariableType::NDARRAY) {
    nd4j_printf(
        "Variable[%i:%i/<%s>] is has [%s] type, but NDArray was requested\n",
        this->_id, this->_index, this->_name.c_str(),
        EnumUtils::_VariableTypeToString(_variableType));
  }

  if (this->_ndarray.get() == nullptr) {
    if (_name.empty()) {
      auto nodeId = StringUtils::valueToString<int>(this->id());
      auto outputIndex = StringUtils::valueToString<int>(this->index());
      throw std::runtime_error("Array doesn't exist for Variable <" + nodeId +
                               ":" + outputIndex + ">");
    } else {
      auto outputIndex = StringUtils::valueToString<int>(this->index());
      throw std::runtime_error("Array doesn't exist for Variable <" +
                               this->_name + ":" + outputIndex + ">");
    }
  }

  return this->_ndarray;
}

std::shared_ptr<sd::NDArrayList> sd::graph::Variable::getNDArrayList() const {
  if (_variableType != VariableType::ARRAY_LIST) {
    nd4j_debug(
        "Variable[%i:%i/<%s>] is has [%s] type, but NDArrayList was "
        "requested\n",
        this->_id, this->_index, this->_name.c_str(),
        EnumUtils::_VariableTypeToString(_variableType));
  }
  return this->_list;
}

bool Variable::isRemovable() const { return _removable; }

void sd::graph::Variable::setNDArrayList(
    std::shared_ptr<sd::NDArrayList> list) {
  this->_variableType = VariableType::ARRAY_LIST;
  this->_list = list;
}

const std::vector<std::pair<int, int>>& Variable::dependencies() const {
  return _dependencies;
}

void Variable::actualizeDependencies(const MAP_IMPL<std::string, int> &lookupTable) const {
  for (const auto &v: _stringDependencies) {
    if (lookupTable.count(v) == 0)
      throw std::runtime_error("Unknown Variable dependency found: [" + v + "]");

    const_cast<Variable*>(this)->_dependencies.emplace_back(std::pair<int, int>{lookupTable.at(v), 0});
  }
}

void sd::graph::Variable::setNDArray(std::shared_ptr<sd::NDArray> array) {
  this->_variableType = VariableType::NDARRAY;
  this->_ndarray = array;
}

VariableType sd::graph::Variable::variableType() const { return _variableType; }

sd::graph::Variable::Variable(const sd::graph::FlatVariable *flatVariable) {
  auto vid = flatVariable->id();
  this->_id = vid->first();
  this->_index = vid->second();

  if (flatVariable->name() != nullptr && flatVariable->name()->size() != 0)
    this->_name = flatVariable->name()->str();

  _external = true;
  _readOnly = false;

  int8_t *buffer = nullptr;

  // reading control deps, and filling _dependencies field
  if (flatVariable->controlDepsForVar() != nullptr && flatVariable->controlDepsForVar()->size() > 0) {
    for (int e = 0; e < flatVariable->controlDepsForVar()->size(); e++)
      _stringDependencies.emplace_back(flatVariable->controlDepsForVar()->Get(e)->str());
  }

  if (flatVariable->controlDepForOp() != nullptr && flatVariable->controlDepForOp()->size() > 0) {
    for (int e = 0; e < flatVariable->controlDepForOp()->size(); e++)
      _stringDependencies.emplace_back(flatVariable->controlDepForOp()->Get(e)->str());
  }

  if (flatVariable->controlDeps() != nullptr && flatVariable->controlDeps()->size() > 0) {
    for (int e = 0; e < flatVariable->controlDeps()->size(); e++)
      _stringDependencies.emplace_back(flatVariable->controlDeps()->Get(e)->str());
  }

  switch (flatVariable->variabletype()) {
    case VarType_VARIABLE: {
      // ?????
      if (flatVariable->ndarray() != nullptr) {
        auto ar = flatVariable->ndarray();
        _ndarray = std::make_shared<sd::NDArray>(
            sd::graph::FlatUtils::fromFlatArray(ar));
      }

      _variableType = VariableType::NDARRAY;
    } break;
    case VarType_CONSTANT: {
      if (flatVariable->ndarray() == nullptr)
        throw std::runtime_error("CONSTANT variable must have NDArray bundled");

      auto ar = flatVariable->ndarray();
      if (ar->dtype() == DType_UTF8) {
        _ndarray = std::make_shared<sd::NDArray>(
            sd::graph::FlatUtils::fromFlatArray(ar));
      } else {
        _ndarray = std::make_shared<sd::NDArray>(
            sd::graph::FlatUtils::fromFlatArray(ar));
      }

      _variableType = VariableType::NDARRAY;
    } break;
    case VarType_ARRAY: {
      // ?????
      if (flatVariable->ndarray() != nullptr) {
        auto ar = flatVariable->ndarray();
        _ndarray = std::make_shared<sd::NDArray>(
            sd::graph::FlatUtils::fromFlatArray(ar));
        // _ndarray->triggerAllocationFlag(true);
      }

      _variableType = VariableType::NDARRAY;
    } break;
    case VarType_PLACEHOLDER: {
      if (flatVariable->shape() == nullptr &&
          flatVariable->ndarray() == nullptr)
        throw std::runtime_error(
            "PLACEHOLDER variable must have shape defined");

      if (flatVariable->ndarray() != nullptr) {
        auto ar = flatVariable->ndarray();
        _ndarray = std::make_shared<sd::NDArray>(
            sd::graph::FlatUtils::fromFlatArray(ar));
        // _ndarray->triggerAllocationFlag(true);

        _variableType = VariableType::NDARRAY;
      }

      if (flatVariable->shape() != nullptr) {
        int shapeLen = flatVariable->shape()->Length();
        for (int i = 0; i < flatVariable->shape()->size(); i++)
          _shape.emplace_back(flatVariable->shape()->Get(i));

        if (_ndarray == nullptr) _variableType = VariableType::PLACEHOLDER;
      }
    } break;
    default:
      throw std::runtime_error("Unknown variable type used");
  }
}

const std::vector<Nd4jLong> &sd::graph::Variable::shape() const {
  return _shape;
}

sd::graph::Variable::Variable(bool placeholder, DataType dataType,
                              const std::vector<Nd4jLong> &shape) {
  _placeholder = placeholder;
  _dtype = dataType;
  _shape = shape;
}

sd::graph::Variable::Variable(std::shared_ptr<NDArray> array,
                              const char *name) {
  _ndarray = array;

  _external = false;
  _readOnly = false;

  if (name != nullptr) _name = std::string(name);

  if (_ndarray != nullptr) _variableType = VariableType::NDARRAY;
}

DataType Variable::dataType() const { return _dtype; }

sd::graph::Variable::Variable(std::shared_ptr<NDArray> array,
                              const std::string &name, int id, int idx)
    : Variable(array, name.c_str()) {
  _id = id;
  _index = idx;
}

sd::graph::Variable::~Variable() {
  //
}

void Variable::setId(int id, int idx) {
  _id = id;
  _index = idx;
}

flatbuffers::Offset<FlatVariable> Variable::asFlatVariable(
    flatbuffers::FlatBufferBuilder &builder) {
  if (this->hasNDArray()) {
    auto array = this->getNDArray();
    auto fShape = builder.CreateVector(array->getShapeInfoAsFlatVector());

    auto fBuffer = builder.CreateVector(array->asByteVector());

    // packing array
    auto fArray = CreateFlatArray(builder, fShape, fBuffer,
                                  (sd::graph::DType)array->dataType());

    // packing id/index of this var
    auto fVid = CreateIntPair(builder, this->_id, this->_index);

    // name is still optional
    flatbuffers::Offset<flatbuffers::String> stringId = 0;
    if (!this->_name.empty()) stringId = builder.CreateString(this->_name);

    // returning array
    return CreateFlatVariable(builder, fVid, stringId,
                              static_cast<sd::graph::DType>(array->dataType()),
                              0, fArray);
  } else {
    throw std::runtime_error(
        "Variable::asFlatVariable isn't possible for NDArrayList");
  }
}
}  // namespace graph
}  // namespace sd

namespace std {

size_t hash<std::pair<int, int>>::operator()(
    const std::pair<int, int> &k) const {
  auto v = std::hash<int>()(k.first);
  v ^= std::hash<int>()(k.second) + 0x9e3779b9 + (v << 6) + (v >> 2);
  return v;
}

size_t hash<bfloat16>::operator()(const bfloat16 &k) const {
  return std::hash<float>()((float)k);
}

size_t hash<float16>::operator()(const float16 &k) const {
  return std::hash<float>()((float)k);
}
}  // namespace std