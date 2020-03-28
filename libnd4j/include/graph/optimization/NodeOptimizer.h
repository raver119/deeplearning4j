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

#ifndef SD_NODEOPTIMIZER_H
#define SD_NODEOPTIMIZER_H

#include <system/dll.h>
#include <graph/Node.h>
#include <string>

namespace sd {
    namespace graph {
        /**
         * This abstract class defines basic methods needed for Inputs/Outputs optimizations. I.e. weight format changes or data types changes for a specific backend
         */
        class SD_EXPORT NodeOptimizer {
        protected:
            std::string _target = {};

        public:
            NodeOptimizer() = default;
            virtual ~NodeOptimizer() = default;

            /**
             * This method applu
             * @param node
             */
            virtual void optimize(Node &node) = 0;

            /**
             * This method returns target Op name for this optimizer
             * @return
             */
            const std::string& targetOp() const;
        };
    }
}



#endif //DEV_TESTS_NODEOPTIMIZER_H
