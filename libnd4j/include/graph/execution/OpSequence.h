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

#ifndef SD_OPSEQUENCE_H
#define SD_OPSEQUENCE_H

#include <vector>
#include <ops/declarable/DeclarableOp.h>


namespace sd {
    namespace graph {
    /**
    * This class represents independent and immutable sequence of operations
    */
    class OpSequence : public std::iterator<std::output_iterator_tag, std::pair<sd::ops::DeclarableOp*, sd::graph::Context*>> {
        // our internal iterator for OpSequence
        class iterator;
        protected:
            // main thing here. sorted list of operations and their contexts
            std::vector<std::pair<sd::ops::DeclarableOp*, sd::graph::Context*>> _ops;
        public:
            explicit OpSequence(const std::vector<std::pair<sd::ops::DeclarableOp*, sd::graph::Context*>> &ops);
            OpSequence() = default;
            ~OpSequence() = default;

            OpSequence(const OpSequence& other) noexcept;

            OpSequence& operator=(const OpSequence& other) noexcept;

            // move constructor
            OpSequence(OpSequence&& other) noexcept;

            // move assignment operator
            OpSequence& operator=(OpSequence&& other) noexcept;

            /**
             * This method returns number of individual operations within this sequence
             * @return
             */
            uint64_t length() const;

            /**
             * This method allows to add DeclarableOp to the end of execution queue
             * @param op - Op to be executed
             * @param ctx - Context for this operation with inputs/outputs/args defined
             */
            void append(sd::ops::DeclarableOp *op, sd::graph::Context *ctx);

            /**
             * Iterator functionality for OpSequence
             * @return
             */

            iterator begin();
            iterator end();

            // additional private section
        private:
            class iterator : public std::iterator<std::output_iterator_tag, std::pair<sd::ops::DeclarableOp*, sd::graph::Context*>> {
            private:
                uint64_t _position = 0;
                OpSequence & _container;
            public:
                explicit iterator(OpSequence & container, uint64_t index = 0);
                std::pair<sd::ops::DeclarableOp*, sd::graph::Context*> operator*() const;
                iterator & operator++();
                iterator & operator++(int);
                bool operator!=(const iterator &) const;
            };
        };
    }
}


#endif //SD_OPSEQUENCE_H
