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

#ifndef SD_EXECUTIONTASK_H
#define SD_EXECUTIONTASK_H

#include <memory>
#include <system/dll.h>
#include <ops/declarable/DeclarableOp.h>

namespace sd {
    namespace graph {
        class SD_EXPORT ExecutionTask {
        protected:
            std::shared_ptr<sd::ops::DeclarableOp> _op;
            const ContextPrototype &_context;

        public:
            ExecutionTask(const std::shared_ptr<sd::ops::DeclarableOp> &op, const ContextPrototype &ctx);

            ~ExecutionTask() = default;

            ExecutionTask(const ExecutionTask& other);

            ExecutionTask& operator=(const ExecutionTask& other) noexcept;

            // move constructor
            ExecutionTask(ExecutionTask&& other);

            // move assignment operator
            ExecutionTask& operator=(ExecutionTask&& other) noexcept;


            std::shared_ptr<sd::ops::DeclarableOp> op() const;

            const ContextPrototype &protoContext() const;
        };
    }
}


#endif //SD_EXECUTIONTASK_H
