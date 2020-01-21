/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
// Created by raver119 on 21.01.2020.
//

#include "cnmlUtils.h"

namespace nd4j {
    namespace ops {
        namespace platforms {
            PLATFORM_IMPL(add, ENGINE_MLU) {
                // that's our inputs/outputs, and we assume they all have proper dtype and shapy by now
                auto x = INPUT_VARIABLE(0);
                auto y = INPUT_VARIABLE(1);
                auto z = OUTPUT_VARIABLE(0);

                // FIXME: temporary code. we want to assume that arrays at this point have CNML Tensor representation
                // creating tensors
                cnmlTensor_t input_tensor_1, input_tensor_2, output_tensor;
                cnrtQueue_t queue;

                // creating an op
                cnmlBaseOp_t op;
                auto status = cnmlCreateAddOp(&op, input_tensor_1, input_tensor_2, output_tensor);
                if (status != CNML_STATUS_SUCCESS)
                    throw std::runtime_error("MLU add: cnmlCreateAddOp failed");

                // executing an op
                status = cnmlComputeAddOpForward_V4(op, input_tensor_1, nullptr, input_tensor_2, nullptr, output_tensor, nullptr, queue, nullptr);
                if (status != CNML_STATUS_SUCCESS)
                    throw std::runtime_error("MLU add: cnmlComputeAddOpForward_V4 failed");

                // FIXME: temporary code. we typically assume that arrays at this point

                // destroy stuff
                cnmlDestroyBaseOp(&op);

                return Status::OK();
            }

            PLATFORM_CHECK(add, ENGINE_MLU) {
                auto x = INPUT_VARIABLE(0);
                auto y = INPUT_VARIABLE(1);
                auto z = OUTPUT_VARIABLE(0);

                auto r = x->dataType() == y->dataType() && x->dataType() == z->dataType() && (x->dataType() == nd4j::DataType::FLOAT32 || x->dataType() == nd4j::DataType::HALF) && x->rankOf() == 4 && y->rankOf() == 4;

                nd4j_printf("CNML add platform: %s\n", r ? "true" : "false");
                return r;
            }
        }
    }
}