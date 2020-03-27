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

#ifndef SD_MANAGEDDATABUFFER_H
#define SD_MANAGEDDATABUFFER_H

#include <array/DataBuffer.h>
#include <memory/GraphMemoryManager.h>

namespace sd {
    /**
     * This class provides special DataBuffer implementation for use within Graphs
     */
    class ND4J_EXPORT ManagedDataBuffer : public DataBuffer  {
    private:
        graph::GraphMemoryManager &_manager;

    protected:
        memory::MemoryZone _zone;
        MemoryDescriptor _descriptor;

    public:
        ManagedDataBuffer(graph::GraphMemoryManager &manager, uint64_t numberOfBytes, DataType dtype, memory::MemoryZone zone);
        ~ManagedDataBuffer();

        void* primary() override;
        void* special() override;
    };
}

#endif //SD_MANAGEDDATABUFFER_H
