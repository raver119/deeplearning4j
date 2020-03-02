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
// Created by raver119 on 15.10.2017.
//

#include "ops/declarable/LogicOp.h"

namespace sd {
    namespace ops {
        LogicOp::LogicOp(const char *name) : DeclarableOp::DeclarableOp(name, true) {
            // just using DeclarableOp constructor
            //this->_descriptor->
        }

        Nd4jStatus LogicOp::validateAndExecute(sd::graph::Context &block) {
            nd4j_logger("WARNING: LogicOps should NOT be ever called\n", "");
            return ND4J_STATUS_BAD_INPUT;
        }

        ShapeList* LogicOp::calculateOutputShape(ShapeList *inputShape, sd::graph::Context &block) {
            // FIXME: we probably want these ops to evaluate scopes
            return SHAPELIST();
        }
    }
}