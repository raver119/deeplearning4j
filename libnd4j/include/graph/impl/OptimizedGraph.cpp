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

#include <graph/OptimizedGraph.h>
#include <graph/Graph.h>

namespace sd {
    namespace graph {
        OptimizedGraph::OptimizedGraph(GraphMemoryManager &memoryManager) {
            _memoryManager = &memoryManager;
        }

        OptimizedGraph::OptimizedGraph(const OptimizedGraph &other) noexcept {
            _onion = other._onion;
            _memoryManager = other._memoryManager;
        }

        OptimizedGraph &OptimizedGraph::operator=(const OptimizedGraph &other) noexcept {
            if (this == &other)
                return *this;

            _onion = other._onion;
            _memoryManager = other._memoryManager;

            return *this;
        }

        OptimizedGraph::OptimizedGraph(OptimizedGraph &&other) noexcept {
            _onion = std::move(other._onion);
            _memoryManager = other._memoryManager;
        }

        OptimizedGraph &OptimizedGraph::operator=(OptimizedGraph &&other) noexcept {
            if (this == &other)
                return *this;

            _onion = std::move(other._onion);
            _memoryManager = other._memoryManager;

            return *this;
        }

        uint64_t OptimizedGraph::layers() const {
            return _onion.size();
        }

        const ExecutionLayer &OptimizedGraph::layer(uint64_t index) const {
            return _onion.at(index);
        }

        void OptimizedGraph::append(const std::vector<OpSequence> &layer) {
            std::lock_guard<std::mutex> lock(_mutex);
            _onion[_onion.size()] = layer;
        }

        void OptimizedGraph::append(OpSequence &sequence) {
            append(ExecutionLayer({sequence}));
        }

        void OptimizedGraph::append(const ExecutionLayer &layer) {
            std::lock_guard<std::mutex> lock(_mutex);
            _onion[_onion.size()] = layer;
        }

        const GraphMemoryManager &OptimizedGraph::memoryManager() const {
            return *_memoryManager;
        }

        const Graph &OptimizedGraph::originalGraph() const {
            return *_originalGraph;
        }
    }
}