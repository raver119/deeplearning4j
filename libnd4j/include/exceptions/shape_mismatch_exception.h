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

#ifndef SD_SHAPE_MISMATCH_EXCEPTION_H
#define SD_SHAPE_MISMATCH_EXCEPTION_H

#include <exceptions/graph_exception.h>
#include <system/dll.h>
#include <system/op_boilerplate.h>
#include <system/pointercast.h>

#include <stdexcept>
#include <vector>

#if defined(_MSC_VER)

// we're ignoring warning about non-exportable parent class, since
// std::runtime_error is a part of Standard C++ Library
#pragma warning(disable : 4275)

#endif

namespace sd {
class SD_EXPORT shape_mismatch_exception : public std::runtime_error {
 public:
  shape_mismatch_exception(const std::string &message);
  ~shape_mismatch_exception() = default;

  static shape_mismatch_exception build(const std::string &message,
                                        const std::vector<Nd4jLong> &expected,
                                        const std::vector<Nd4jLong> &actual);
};
}  // namespace sd

#endif  // SD_SHAPE_MISMATCH_EXCEPTION_H
