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

#include <memory/MemoryDescriptor.h>

namespace sd {
namespace memory {
MemoryDescriptor::MemoryDescriptor(void *ptr, MemoryZone zone, uint64_t bytes)
    : _ptr(ptr), _zone(zone), _bytes(bytes) {
  //
}

MemoryDescriptor::MemoryDescriptor(const MemoryDescriptor &other) noexcept
    : _ptr(other._ptr), _zone(other._zone), _bytes(other._bytes) {
  //
}

MemoryDescriptor &MemoryDescriptor::operator=(
    const MemoryDescriptor &other) noexcept {
  if (this == &other) return *this;

  _ptr = other._ptr;
  _zone = other._zone;
  _bytes = other._bytes;

  return *this;
}

MemoryDescriptor::MemoryDescriptor(MemoryDescriptor &&other) noexcept
    : _ptr(other._ptr), _zone(other._zone), _bytes(other._bytes) {
  //
}

MemoryDescriptor &MemoryDescriptor::operator=(
    MemoryDescriptor &&other) noexcept {
  if (this == &other) return *this;

  _ptr = other._ptr;
  _zone = other._zone;
  _bytes = other._bytes;

  return *this;
}

void *MemoryDescriptor::address() const { return _ptr; }

MemoryZone MemoryDescriptor::zone() const { return _zone; }

uint64_t MemoryDescriptor::bytes() const { return _bytes; }
}  // namespace memory
}  // namespace sd
