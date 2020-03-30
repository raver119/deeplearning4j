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

#include <graph/execution/OpSequence.h>

namespace sd {
    namespace graph {
        OpSequence::OpSequence(const int deviceId) : _deviceId(deviceId) {
            //
        }

        OpSequence::OpSequence(const std::vector<std::pair<sd::ops::DeclarableOp*, sd::graph::ContextPrototype*>> &ops, const int deviceId) {
            _deviceId = deviceId;

            for (const auto v : ops)
                _ops.emplace_back(v);
        }

        OpSequence::OpSequence(const OpSequence& other) noexcept{
            for (const auto v : other._ops)
                _ops.emplace_back(v);
        }

        ////////////////////////////////////////////////////////////////////////
        // move constructor
        OpSequence::OpSequence(OpSequence&& other) noexcept {
            _ops = std::move(other._ops);
        }

        OpSequence& OpSequence::operator=(OpSequence&& other) noexcept {
            if (this == &other)
                return *this;

            _ops = std::move(other._ops);

            return *this;
        }

        OpSequence& OpSequence::operator=(const OpSequence& other) noexcept {
            if (this == &other)
                return *this;

            for (const auto v : other._ops)
                _ops.emplace_back(v);

            return *this;
        }

        int OpSequence::deviceId() const {
            return _deviceId;
        }

        std::pair<sd::ops::DeclarableOp *, sd::graph::ContextPrototype *> OpSequence::at(uint64_t index) const {
            return _ops[index];
        }

        std::pair<sd::ops::DeclarableOp *, sd::graph::ContextPrototype *> OpSequence::operator[](uint64_t index) const {
            return at(index);
        }

        uint64_t OpSequence::length() const {
            return _ops.size();
        }

        void OpSequence::append(sd::ops::DeclarableOp *op, sd::graph::ContextPrototype *ctx) {
            _ops.emplace_back(std::pair<sd::ops::DeclarableOp *, sd::graph::ContextPrototype *>{op, ctx});
        }


        OpSequence::iterator
        OpSequence::begin() {
            return OpSequence::iterator(*this, 0);
        }

        OpSequence::iterator
        OpSequence::end() {
            return OpSequence::iterator(*this, length());
        }

        OpSequence::iterator::iterator(OpSequence &container, uint64_t index) :_container(container), _position(index) {
            //
        }

        std::pair<sd::ops::DeclarableOp*, sd::graph::ContextPrototype*> OpSequence::iterator::operator*() const {
            return _container._ops[_position];
        }

        OpSequence::iterator &OpSequence::iterator::operator++() {
            _position++;
            return *this;
        }

        OpSequence::iterator &OpSequence::iterator::operator++(int inc) {
            return ++(*this);
        }

        bool OpSequence::iterator::operator!=(const OpSequence::iterator &other) const {
            return _position != other._position;
        }
    }
}
