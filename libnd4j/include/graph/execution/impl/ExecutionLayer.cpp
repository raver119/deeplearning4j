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


#include <graph/execution/ExecutionLayer.h>

namespace sd {
    namespace graph {
        ExecutionLayer::ExecutionLayer(const std::vector<OpSequence> &sequences) {
            _sequences = sequences;
        }

        uint64_t ExecutionLayer::width() const {
            return _sequences.size();
        }

        const OpSequence& ExecutionLayer::at(uint64_t index) const {
            return _sequences[index];
        }

        const OpSequence& ExecutionLayer::operator[](uint64_t index) const {
            return at(index);
        }

        void ExecutionLayer::append(const OpSequence &sequence) {
            _sequences.emplace_back(sequence);
        }

        ExecutionLayer::ExecutionLayer(const ExecutionLayer &other) noexcept {
            _sequences = other._sequences;
        }

        ExecutionLayer &ExecutionLayer::operator=(const ExecutionLayer &other) noexcept {
            if (this == &other)
                return *this;

            _sequences = other._sequences;

            return *this;
        }

        ExecutionLayer::ExecutionLayer(ExecutionLayer &&other) noexcept {
            _sequences = std::move(other._sequences);
        }

        ExecutionLayer &ExecutionLayer::operator=(ExecutionLayer &&other) noexcept {
            if (this == &other)
                return *this;

            _sequences = std::move(other._sequences);

            return *this;
        }
    }
}