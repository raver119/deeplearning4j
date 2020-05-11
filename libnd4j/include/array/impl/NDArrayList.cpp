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

#include <array/NDArrayList.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/stack.h>

#include <iterator>

namespace sd {
NDArrayList::InternalArrayList::InternalArrayList(int height, bool expandable) {
  _expandable = expandable;
  _elements.store(0);
  _counter.store(0);
  _id.first = 0;
  _id.second = 0;
  _height = height;
}

NDArrayList::NDArrayList(const NDArrayList &other) { _state = other._state; }

NDArrayList::NDArrayList(NDArrayList &&other) {
  _state = std::move(other._state);
}

NDArrayList &NDArrayList::operator=(const NDArrayList &other) noexcept {
  if (this == &other) return *this;

  _state = other._state;

  return *this;
}

NDArrayList &NDArrayList::operator=(NDArrayList &&other) noexcept {
  if (this == &other) return *this;

  _state = std::move(other._state);

  return *this;
}

NDArrayList::NDArrayList(int height, bool expandable) {
  _state = std::make_shared<InternalArrayList>(height, expandable);
}

NDArrayList::~NDArrayList() {}

NDArray NDArrayList::read(int idx) { return readRaw(idx); }

sd::DataType NDArrayList::dataType() const { return _state->_dtype; }

NDArray NDArrayList::readRaw(int idx) {
  if (_state->_chunks.count(idx) < 1) {
    nd4j_printf("Non-existent chunk requested: [%i]\n", idx);
    throw std::invalid_argument("Bad index");
  }

  return _state->_chunks.at(idx);
}

Nd4jStatus NDArrayList::write(int idx, const NDArray &array) {
  if (_state->_chunks.count(idx) == 0)
    _state->_elements++;
  else {
    _state->_chunks.erase(idx);
  }

  // we store reference shape on first write
  if (_state->_chunks.empty()) {
    _state->_dtype = array.dataType();

    if (_state->_shape.empty()) {
      // adding leading 1 to shape
      _state->_shape.emplace_back(1);
      for (int e = 0; e < array.rankOf(); e++)
        _state->_shape.emplace_back(array.sizeAt(e));
    } else {
      // if shape is inferred (say, from split_list)
      if (array.rankOf() == _state->_shape.size()) {
        // skipping first dim
        for (int e = 1; e < _state->_shape.size(); e++) {
          if (_state->_shape[e] != array.sizeAt(e))
            return Status::CODE(ND4J_STATUS_BAD_INPUT,
                                "NDArrayList: all arrays must have same size "
                                "along inner dimensions");
        }
      } else if (array.rankOf() == _state->_shape.size() - 1) {
        // case like 2d _shape, and 1D rows
        for (int e = 1; e < _state->_shape.size(); e++)
          if (_state->_shape[e] != array.sizeAt(e - 1))
            return Status::CODE(ND4J_STATUS_BAD_INPUT,
                                "NDArrayList: all arrays must have same size "
                                "along inner dimensions");
      } else
        return Status::CODE(ND4J_STATUS_BAD_INPUT,
                            "NDArrayList: all arrays must have same size along "
                            "inner dimensions");
    }
  } else {
    if (array.dataType() != _state->_dtype)
      return Status::CODE(ND4J_STATUS_BAD_INPUT,
                          "NDArrayList: all arrays must have same data type");

    // if shape is inferred (say, from split_list)
    if (array.rankOf() == _state->_shape.size()) {
      // skipping first dim
      for (int e = 1; e < _state->_shape.size(); e++) {
        if (_state->_shape[e] != array.sizeAt(e))
          return Status::CODE(ND4J_STATUS_BAD_INPUT,
                              "NDArrayList: all arrays must have same size "
                              "along inner dimensions");
      }
    } else if (array.rankOf() == _state->_shape.size() - 1) {
      // case like 2d _shape, and 1D rows
      for (int e = 1; e < _state->_shape.size(); e++)
        if (_state->_shape[e] != array.sizeAt(e - 1))
          return Status::CODE(ND4J_STATUS_BAD_INPUT,
                              "NDArrayList: all arrays must have same size "
                              "along inner dimensions");
    } else
      return Status::CODE(
          ND4J_STATUS_BAD_INPUT,
          "NDArrayList: all arrays must have same size along inner dimensions");
  }

  // storing reference
  _state->_chunks.insert({idx, array});

  return Status::OK();
}

const std::vector<Nd4jLong> &NDArrayList::shape() const {
  return _state->_shape;
}

int NDArrayList::counter() const { return _state->_counter++; }

void NDArrayList::unstack(const NDArray &array, int axis) {
  _state->_axis = axis;
  std::vector<int> args({axis});
  auto newAxis = ShapeUtils::evalDimsToExclude(array.rankOf(), args);
  auto result = array.allTensorsAlongDimension(newAxis);
  for (int e = 0; e < result.size(); e++) {
    auto chunk = result.at(e);
    write(e, chunk.dup(array.ordering()));
  }
}

void NDArrayList::setShape(const std::vector<Nd4jLong> &shape) {
  _state->_shape = shape;
}

NDArray NDArrayList::stack() const {
  // FIXME: this is bad for perf, but ok as poc

  int numElements = _state->_elements.load();
  std::vector<const NDArray *> inputs(numElements);
  for (int e = 0; e < numElements; e++) {
    _state->_chunks.at(e).syncToDevice();
    inputs[e] = &_state->_chunks.at(e);
  }

  auto inShapeInfo = inputs[0]->shapeInfo();
  int rank = shape::rank(inShapeInfo);
  NDArray array;

  if (shape::isEmpty(inShapeInfo)) {
    switch (rank) {
      case 0: {
        if (numElements == 1) {
          array = NDArray(inputs[0]->ordering(), {0},
                          ArrayOptions::dataType(inShapeInfo),
                          inputs[0]->getContext());
        } else {
          array = NDArray('c', {(Nd4jLong)numElements, 0},
                          ArrayOptions::dataType(inShapeInfo),
                          inputs[0]->getContext());
        }
      }
    }
  } else {
    std::vector<Nd4jLong> outShape(inShapeInfo + 1, inShapeInfo + 1 + rank);
    outShape.insert(outShape.begin(), (Nd4jLong)numElements);
    array =
        NDArray(shape::order(inShapeInfo), outShape,
                ArrayOptions::dataType(inShapeInfo), inputs[0]->getContext());
  }

  ops::helpers::stack(inputs[0]->getContext(), inputs, array, 0);

  return array;
}

const std::pair<int, int> &NDArrayList::id() const { return _state->_id; }

const std::string &NDArrayList::name() const { return _state->_name; }

int NDArrayList::elements() const { return (int)_state->_chunks.size(); }

int NDArrayList::height() const { return (int)_state->_chunks.size(); }

bool NDArrayList::isWritten(int index) const {
  if (_state->_chunks.count(index) > 0)
    return true;
  else
    return false;
}

NDArray NDArrayList::pick(const std::vector<int> &indices) {
  std::vector<Nd4jLong> shape(_state->_shape);

  // shape.insert(shape.begin() + _axis, indices.size());
  shape[_state->_axis] = indices.size();
  // do we have to enforce C order here?
  NDArray array('c', shape, _state->_chunks.at(0).dataType());
  std::vector<int> axis =
      ShapeUtils::evalDimsToExclude(shape.size(), {_state->_axis});
  auto tads = array.allTensorsAlongDimension(axis);
  int indicesSize = indices.size();

  if (tads.size() != indicesSize)
    throw std::runtime_error("Number of TADs should match number of indices");

  for (int e = 0; e < indicesSize; e++)
    tads.at(e).assign(_state->_chunks.at(indices[e]));

  return array;
}

NDArrayList NDArrayList::clone() {
  NDArrayList list(_state->_height, _state->_expandable);
  list._state->_axis = _state->_axis;
  list._state->_id.first = _state->_id.first;
  list._state->_id.second = _state->_id.second;
  list._state->_name = _state->_name;
  list._state->_elements.store(_state->_elements.load());

  for (auto const &v : _state->_chunks) {
    list._state->_chunks.insert({v.first, v.second.dup()});
  }

  return list;
}

bool NDArrayList::equals(NDArrayList &other) {
  if (_state->_axis != other._state->_axis) return false;

  if (_state->_chunks.size() != other._state->_chunks.size()) return false;

  for (auto const &v : _state->_chunks) {
    if (other._state->_chunks.count(v.first) == 0) return false;

    auto arrThis = _state->_chunks.at(v.first);
    auto arrThat = other._state->_chunks.at(v.first);

    if (!arrThis.equalsTo(arrThat)) return false;
  }

  return true;
}
}  // namespace sd