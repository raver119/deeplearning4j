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

#include <memory/GraphMemoryManager.h>
#include <memory/HotRamZoneManager.h>
#include <memory/ColdZoneManager.h>

namespace sd {
    namespace graph {
        GraphMemoryManager::GraphMemoryManager() {
            // first of all we initialize all memory managers
            // CPU backend only has two: HOT and COLD

            _zones[MemoryZone::HOT] = new memory::HotRamZoneManager();
            _zones[MemoryZone::COLD] = new memory::ColdZoneManager();
        }

        GraphMemoryManager::~GraphMemoryManager() {
            delete _zones[MemoryZone::HOT];
            delete _zones[MemoryZone::COLD];
        }

        MemoryDescriptor GraphMemoryManager::allocate(size_t numBytes, MemoryZone zone) {
            if (zone == MemoryZone::WARM)
                zone = MemoryZone::HOT;

            return _zones[zone]->allocate(numBytes);
        }

        void GraphMemoryManager::release(MemoryDescriptor &descriptor) {
            _zones[descriptor.zone()]->release(descriptor);
        }
    }
}