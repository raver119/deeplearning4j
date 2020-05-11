/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// Created by raver on 2/28/2019.
//

#include "../OpBenchmark.h"

namespace sd {
OpBenchmark::OpBenchmark(const std::string &name, const NDArray &x,
                         const NDArray &y, const NDArray &z) {
  _testName = name;
  _x = x;
  _y = y;
  _z = z;
}

OpBenchmark::OpBenchmark(const std::string &name, const NDArray &x,
                         const NDArray &z) {
  _testName = name;
  _x = x;
  _z = z;
}

OpBenchmark::OpBenchmark(const std::string &name, const NDArray &x,
                         const NDArray &z, const std::vector<int> &axis) {
  _testName = name;
  _x = x;
  _z = z;
  _axis = axis;

  if (_axis.size() > 1) std::sort(_axis.begin(), _axis.end());
}

OpBenchmark::OpBenchmark(const std::string &name, const NDArray &x,
                         const NDArray &y, const NDArray &z,
                         const std::vector<int> &axis) {
  _testName = name;
  _x = x;
  _y = y;
  _z = z;
  _axis = axis;

  if (_axis.size() > 1) std::sort(_axis.begin(), _axis.end());
}

NDArray &OpBenchmark::x() { return _x; }

int OpBenchmark::opNum() const { return _opNum; }
const std::string &OpBenchmark::testName() const { return _testName; }

void OpBenchmark::setOpNum(int opNum) { _opNum = opNum; }

void OpBenchmark::setTestName(const std::string &name) { _testName = name; }

void OpBenchmark::setX(const NDArray &array) { _x = array; }

void OpBenchmark::setY(const NDArray &array) { _y = array; }

void OpBenchmark::setZ(const NDArray &array) { _z = array; }

void OpBenchmark::setAxis(std::vector<int> axis) { _axis = axis; }

void OpBenchmark::setAxis(std::initializer_list<int> axis) { _axis = axis; }

std::vector<int> OpBenchmark::getAxis() { return _axis; }

std::string OpBenchmark::extra() { return "N/A"; }

std::string OpBenchmark::shape() {
  if (_x.shapeInfo() != nullptr)
    return ShapeUtils::shapeAsString(_x);
  else if (_z.shapeInfo() != nullptr)
    return ShapeUtils::shapeAsString(_z);
  else
    return "N/A";
}

std::string OpBenchmark::dataType() {
  if (_x.shapeInfo() != nullptr)
    return DataTypeUtils::asString(_x.dataType());
  else if (_z.shapeInfo() != nullptr)
    return DataTypeUtils::asString(_z.dataType());
  else
    return "N/A";
}
}  // namespace sd