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

                return Status::OK();
            }

            PLATFORM_CHECK(add, ENGINE_MLU) {
                auto x = INPUT_VARIABLE(0);
                auto y = INPUT_VARIABLE(1);
                auto z = OUTPUT_VARIABLE(0);



                auto r = x->dataType() == y->dataType() && x->dataType() == z->dataType() && (x->dataType() = nd4j::DataType::FLOAT32 || x->dataType() = nd4j::DataType::HALF) && x->rankOf() == 4 && y->rankOf() == 4;

                nd4j_printf("CNML add platform: %s\n", r ? "true" : false);
                return r;
            }
        }
    }
}