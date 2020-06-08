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

#ifndef SD_HOTZONEMANAGER_H
#define SD_HOTZONEMANAGER_H

#include <memory/ZoneManager.h>

#include <atomic>

namespace sd {
namespace memory {
class SD_EXPORT HotZoneManager : public ZoneManager {
 protected:
  std::atomic<uint64_t> _used = {0};
  std::atomic<uint64_t> _available = {0};

 public:
  HotZoneManager() = default;
  ~HotZoneManager() = default;

  MemoryZone zone() const override;

  uint64_t available() const override;

  uint64_t used() const override;

  virtual MemoryDescriptor allocate(uint64_t numBytes) override = 0;

  virtual void release(MemoryDescriptor &descriptor) override = 0;
};
}  // namespace memory
}  // namespace sd

#endif  // SD_HOTZONEMANAGER_H
