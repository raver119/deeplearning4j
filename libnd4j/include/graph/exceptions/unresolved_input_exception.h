/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

#ifndef SD_UNRESOLVED_INPUT_H
#define SD_UNRESOLVED_INPUT_H

#include <utility>
#include <string>
#include <stdexcept>

namespace sd {
    namespace graph {
        class unresolved_input_exception : public std::runtime_error {
        public:
            unresolved_input_exception(const std::string &message);
            ~unresolved_input_exception() = default;

            static unresolved_input_exception build(const std::string &message, int nodeId, std::pair<int, int> &varIndex);
            static unresolved_input_exception build(const std::string &message, std::pair<int, int> &varIndex);
            static unresolved_input_exception build(const std::string &message, const std::string &varName);
        };
    }
}

#endif //SD_UNRESOLVED_INPUT_H
