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
#ifndef SD_OPTIMIZEDGRAPH_H
#define SD_OPTIMIZEDGRAPH_H

#include <graph/execution/OpSequence.h>
#include <vector>
#include <map>

namespace sd {
    namespace graph {
        /**
         * This class acts as a topologically sorted & optimized for top
         */
        class OptimizedGraph {
        protected:
            // here we store independent OpSequences
            // Graph starts from layer 0, and goes deeper step by step
            // on each layer we can have 1+ OpSequences that can be executed independent
            std::map<uint64_t, std::vector<OpSequence>> _onion;
        public:
            OptimizedGraph() = default;
            ~OptimizedGraph() = default;

            /**
             * This method returns number of layers within OptimizedGraph
             * @return
             */
            uint64_t layers() const;

            /**
             * This method returns OpSequences stored in a given layer
             * @param index
             * @return
             */
            const std::vector<OpSequence>& layer(uint64_t index) const;

            /**
             * This method allows to append layer to this OptimizedGraph instance
             */
             // FIXME: this method should be removed or made private
            void append(const std::vector<OpSequence> &layer);
        };
    }
}


#endif //SD_OPTIMIZEDGRAPH_H
