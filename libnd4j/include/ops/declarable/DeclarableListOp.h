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
//  @author raver119@gmail.com
//

#ifndef LIBND4J_DECLARABLE_LIST_OP_H
#define LIBND4J_DECLARABLE_LIST_OP_H

#include <array/ResultSet.h>
#include <graph/Context.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/DeclarableOp.h>

using namespace sd::graph;

namespace sd {
    namespace ops {
        class SD_EXPORT DeclarableListOp : public sd::ops::DeclarableOp {
        protected:
            Nd4jStatus validateAndExecute(Context& block) override = 0;

            sd::NDArray* getZ(Context& block, int inputId) ;
            void setupResult(const NDArray &array, Context& block);
            void setupResultList(const NDArrayList &arrayList, Context& block);

        public:
            DeclarableListOp(int numInputs, int numOutputs, const char* opName, int tArgs, int iArgs);
            
            Nd4jStatus execute(Context* block) override;

            ResultSet execute(const NDArrayList &list, const std::vector<NDArray*>& inputs, const std::vector<double>& tArgs = {}, const std::vector<int>& iArgs = {});

            ShapeList* calculateOutputShape(ShapeList* inputShape, sd::graph::Context& block) override;
        };
    }
}

#endif