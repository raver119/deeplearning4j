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
#ifndef SD_INFERENCEREQUEST_H
#define SD_INFERENCEREQUEST_H

#include <system/op_boilerplate.h>
#include <system/pointercast.h>
#include <system/dll.h>
#include <graph/Variable.h>
#include "ExecutorConfiguration.h"

namespace sd {
    namespace graph {
        class SD_EXPORT InferenceRequest {
        private:
            Nd4jLong _id;
            std::vector<std::shared_ptr<Variable>> _variables;

            ExecutorConfiguration _configuration;

            void insertVariable(std::shared_ptr<Variable> variable);
        public:

            InferenceRequest(Nd4jLong graphId, const ExecutorConfiguration &configuration);
            ~InferenceRequest();

            void appendVariable(int id, const NDArray &array);
            void appendVariable(int id, int index, const NDArray &array);
            void appendVariable(const std::string &name, const NDArray &array);
            void appendVariable(const std::string &name, int id, int index, const NDArray &array);
            void appendVariable(std::shared_ptr<Variable> variable);

#ifndef __JAVACPP_HACK__
            flatbuffers::Offset<FlatInferenceRequest> asFlatInferenceRequest(flatbuffers::FlatBufferBuilder &builder);
#endif
        };
    }
}



#endif //SD_INFERENCEREQUEST_H
