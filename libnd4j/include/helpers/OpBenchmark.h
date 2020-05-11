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

#ifndef SD_OPEXECUTIONER_H
#define SD_OPEXECUTIONER_H

#include <array/NDArray.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <legacy/NativeOpExecutioner.h>

namespace sd {
class SD_EXPORT OpBenchmark {
 protected:
  int _opNum = 0;
  std::string _testName;
  NDArray _x;
  NDArray _y;
  NDArray _z;
  std::vector<int> _axis;

 public:
  OpBenchmark() = default;
  OpBenchmark(const std::string &name, const NDArray &x, const NDArray &y,
              const NDArray &z);
  OpBenchmark(const std::string &name, const NDArray &x, const NDArray &z);
  OpBenchmark(const std::string &name, const NDArray &x, const NDArray &z,
              const std::vector<int> &axis);
  OpBenchmark(const std::string &name, const NDArray &x, const NDArray &y,
              const NDArray &z, const std::vector<int> &axis);

  void setOpNum(int opNum);
  void setTestName(const std::string &testName);
  void setX(const NDArray &array);
  void setY(const NDArray &array);
  void setZ(const NDArray &array);
  void setAxis(std::vector<int> axis);
  void setAxis(std::initializer_list<int> axis);

  NDArray &x();
  int opNum() const;
  const std::string &testName() const;
  std::vector<int> getAxis();

  virtual std::string extra();
  virtual std::string dataType();
  virtual std::string axis() = 0;
  virtual std::string orders() = 0;
  virtual std::string strides() = 0;
  virtual std::string shape();
  virtual std::string inplace() = 0;

  virtual void executeOnce() = 0;

  virtual OpBenchmark *clone() = 0;
};
}  // namespace sd

#endif  // SD_OPEXECUTIONER_H
