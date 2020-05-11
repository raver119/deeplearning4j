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

#include <memory/ColdZoneManager.h>

namespace sd {
namespace memory {
ColdZoneManager::ColdZoneManager(const char *filename) {
  //
}

MemoryZone ColdZoneManager::zone() const { return COLD; }

uint64_t ColdZoneManager::available() const { return 0; }

uint64_t ColdZoneManager::used() const { return 0; }

MemoryDescriptor ColdZoneManager::allocate(uint64_t numBytes) {
  return MemoryDescriptor(nullptr, COLD, numBytes);
}

void ColdZoneManager::release(MemoryDescriptor &descriptor) {
  //
}
}  // namespace memory
}  // namespace sd
