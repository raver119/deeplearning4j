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

#ifndef SD_MEMORYDESCRIPTOR_H
#define SD_MEMORYDESCRIPTOR_H

#include <memory/MemoryZone.h>
#include <system/dll.h>
#include <cstdint>

namespace sd {
    namespace memory {
        class ND4J_EXPORT MemoryDescriptor {
        private:
            void* _ptr;
            MemoryZone _zone;
            uint64_t _bytes;
        public:
            MemoryDescriptor(void *ptr, MemoryZone zone, uint64_t bytes);
            ~MemoryDescriptor() = default;

            MemoryDescriptor(const MemoryDescriptor& other) noexcept;

            MemoryDescriptor& operator=(const MemoryDescriptor& other) noexcept;

            // move constructor
            MemoryDescriptor(MemoryDescriptor&& other) noexcept;

            // move assignment operator
            MemoryDescriptor& operator=(MemoryDescriptor&& other) noexcept;
        };
    }
}


#endif //SD_MEMORYDESCRIPTOR_H
