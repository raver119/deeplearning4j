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

#ifndef LIBND4J_BENCHMARKSUIT_H
#define LIBND4J_BENCHMARKSUIT_H

#include <array/NDArrayFactory.h>
#include <helpers/BenchmarkHelper.h>
#include <system/dll.h>
#include <system/pointercast.h>

#include <string>

namespace sd {
class SD_EXPORT BenchmarkSuit {
 public:
  BenchmarkSuit() = default;
  ~BenchmarkSuit() = default;

  virtual std::string runSuit() = 0;
};
}  // namespace sd

#endif  // SD_BENCHMARKSUIT_H
