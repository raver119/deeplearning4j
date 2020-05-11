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

#include <graph/execution/ExecutionTask.h>

namespace sd {
    namespace graph {
        ExecutionTask::ExecutionTask(const std::shared_ptr<sd::ops::DeclarableOp> &op, const ContextPrototype &ctx) : _op(op), _context(ctx) {
            //
        }

        std::shared_ptr<sd::ops::DeclarableOp> ExecutionTask::op() const {
            return _op;
        }

        const ContextPrototype &ExecutionTask::protoContext() const {
            return _context;
        }

        ExecutionTask::ExecutionTask(const ExecutionTask &other) : _op(other._op), _context(other._context) {
            //
        }

        ExecutionTask &ExecutionTask::operator=(const ExecutionTask &other) noexcept {
            if (this == &other)
                return *this;

            _op = other._op;
            const_cast<ContextPrototype &>(_context) = other._context;

            return *this;
        }

        ExecutionTask::ExecutionTask(ExecutionTask &&other) : _op(other._op), _context(other._context) {
            //
        }

        void ExecutionTask::printOut() const {
            if (_context.name().empty()) {
                if (_op != nullptr)
                    printf("   <%i:0>: {Op: %s}; ", _context.nodeId(), _op->getOpName().c_str());
                else
                    printf("   <%i:0>: ", _context.nodeId());
            } else {
                printf("   <%s> <%i>: ",  _context.name().c_str(), _context.nodeId());
            }

            auto sz = _context.inputs().size();
            if (sz) {
                printf(" Inputs: [");
                int cnt = 0;
                for (const auto &v:_context.inputs()) {
                    printf("<%i:%i>", v.first, v.second);

                    if (cnt < sz - 1)
                        printf(", ");
                    cnt++;
                }

                printf("]; ");
            } else {
                printf(" No inputs; ");
            }

            printf("\n");
            fflush(stdout);
        }

        ExecutionTask &ExecutionTask::operator=(ExecutionTask &&other) noexcept {
            if (this == &other)
                return *this;

            _op = std::move(other._op);
            const_cast<ContextPrototype &>(_context) = std::move(other._context);

            return *this;
        }
    }
}