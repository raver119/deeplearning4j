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

#ifndef SD_EXTRAARGUMENTS_H
#define SD_EXTRAARGUMENTS_H

#include <array/DataType.h>
#include <stdlib.h>
#include <system/dll.h>
#include <system/pointercast.h>

#include <initializer_list>
#include <vector>

namespace sd {
class SD_EXPORT ExtraArguments {
 private:
  std::vector<double> _fpArgs;
  std::vector<Nd4jLong> _intArgs;

  std::vector<Nd4jPointer> _pointers;

  template <typename T>
  void convertAndCopy(Nd4jPointer pointer, Nd4jLong offset);

  void* allocate(size_t length, size_t elementSize);

 public:
  explicit ExtraArguments(std::initializer_list<double> arguments);
  explicit ExtraArguments(std::initializer_list<Nd4jLong> arguments);

  explicit ExtraArguments(const std::vector<double>& arguments);
  explicit ExtraArguments(const std::vector<int>& arguments);
  explicit ExtraArguments(const std::vector<Nd4jLong>& arguments);

  explicit ExtraArguments();
  ~ExtraArguments();

  template <typename T>
  void* argumentsAsT(Nd4jLong offset = 0);

  void* argumentsAsT(sd::DataType dataType, Nd4jLong offset = 0);

  size_t length();
};
}  // namespace sd

#endif  // SD_EXTRAARGUMENTS_H
