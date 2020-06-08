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
//  @author raver119@gmail.com
//

#ifndef SD_CONSTANTHELPER_H
#define SD_CONSTANTHELPER_H

#include <array/ConstantDataBuffer.h>
#include <array/ConstantDescriptor.h>
#include <array/ConstantHolder.h>
#include <memory/Workspace.h>
#include <system/dll.h>
#include <system/op_boilerplate.h>
#include <system/pointercast.h>

#include <map>
#include <mutex>
#include <vector>

namespace sd {
class SD_EXPORT ConstantHelper {
 private:

  ConstantHelper();

  std::vector<MAP_IMPL<ConstantDescriptor, ConstantHolder*>> _cache;

  // tracking of per-device constant memory buffers (CUDA only atm)
  std::vector<Nd4jPointer> _devicePointers;
  std::vector<Nd4jLong> _deviceOffsets;
  std::mutex _mutex;
  std::mutex _mutexHolder;

  std::vector<Nd4jLong> _counters;

 public:
  ~ConstantHelper();

  static ConstantHelper& getInstance();
  static int getCurrentDevice();
  static int getNumberOfDevices();
  void* replicatePointer(void* src, size_t numBytes,
                         memory::Workspace* workspace = nullptr);

  ConstantDataBuffer* constantBuffer(const ConstantDescriptor& descriptor,
                                     sd::DataType dataType);

  Nd4jLong getCachedAmount(int deviceId);
};
}  // namespace sd

#endif  // SD_CONSTANTHELPER_H
