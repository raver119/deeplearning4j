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

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_unstack)

#include <ops/declarable/CustomOperations.h>
#include <helpers/ConstantTadHelper.h>
#include <performance/benchmarking/global_timers.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(unstack, 1, -1, false, 0, 1) {
            auto timers = nd4j::GlobalTimers::getInstance();
            timers->reset();
            timers->stopWatch(__LINE__, 1);
            auto input = INPUT_VARIABLE(0);
            timers->stopWatch(__LINE__, 1);
            auto dim = INT_ARG(0);
            if (dim < 0)
                dim += input->rankOf();

            timers->stopWatch(__LINE__, 1);
            REQUIRE_TRUE(dim < input->rankOf(), 0, "Unstack dimension should be lower then rank of input %i, but got dimension=%i !", input->rankOf(), dim);
            REQUIRE_TRUE(dim >= 0, 0, "Unstack dimension should be non-negative value, but got %i !", dim);

            if(input->isEmpty())
                return Status::OK();
            timers->stopWatch(__LINE__, 1);
            std::vector<int> dims;
            for (int e = 0; e < input->rankOf(); e++)
                if (e != dim)
                    dims.emplace_back(e);
            timers->stopWatch(__LINE__, 1);
            if (dims.size() == 0 && input->rankOf() == 1) { // split vector into lenthOf scalars
                timers->stopWatch(__LINE__, 1);
                for (Nd4jLong e = 0; e < input->lengthOf(); e++) {
                    auto outE = OUTPUT_VARIABLE(e);
                    outE->assign(input->e(e));
                }
            }
            timers->stopWatch(__LINE__, 1);

            auto tads = input->allTensorsAlongDimension(dims);
            timers->stopWatch(__LINE__, 1);
            //nd4j_printf("Tad size: %d\n",tads.size());
            for (int e = 0; e < tads.size(); e++) {
                //nd4j_printf("Calling assign at index %d\n",e);
                timers->stopWatch(__LINE__, 1);
                auto outE = OUTPUT_VARIABLE(e);
                timers->stopWatch(__LINE__, 1);
                auto tadAtE = tads.at(e);
                timers->stopWatch(__LINE__, 1);

                outE->assign(tadAtE);
timers->stopWatch(__LINE__, 1);
                this->storeResult(block, e, *outE);
timers->stopWatch(__LINE__, 1);
            }
            timers->stopWatch(__LINE__, 1);

            return Status::OK();
        }

        DECLARE_SYN(unpack, unstack);

        DECLARE_SHAPE_FN(unstack) {
auto timers = nd4j::GlobalTimers::getInstance();
            auto inShape = inputShape->at(0);
timers->stopWatch(__LINE__, 1);
            auto dim = INT_ARG(0);
            if (dim < 0)
                dim += shape::rank(inShape);
timers->stopWatch(__LINE__, 1);
            REQUIRE_TRUE(dim < inShape[0], 0, "UNSTACK op: dimension should be lower then rank of input %i, but got dimension=%i !", inShape[0], dim);
            REQUIRE_TRUE(dim >= 0, 0, "UNSTACK op: dimension should be non-negative value, but got %i !", dim);
timers->stopWatch(__LINE__, 1);
            if(ArrayOptions::arrayType(inShape) == ArrayType::EMPTY) {
                if(shape::shapeOf(inShape)[dim] == 0)
                    return SHAPELIST();
                const Nd4jLong numTads = shape::shapeOf(inShape)[dim];
                timers->stopWatch(__LINE__, 1);
                std::vector<Nd4jLong> outShape;
                for(uint i = 0; i < shape::rank(inShape); ++i)
                    if(i != dim)
                        outShape.push_back(shape::shapeOf(inShape)[i]);
timers->stopWatch(__LINE__, 1);
                auto result = SHAPELIST();
timers->stopWatch(__LINE__, 1);
                for(uint i = 0; i < numTads; ++i)
                    result->push_back(ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inShape), shape::order(inShape), outShape));
timers->stopWatch(__LINE__, 1);
                return result;
            }

            std::vector<int> dims;
timers->stopWatch(__LINE__, 1);
            for (int e = 0; e < shape::rank(inShape); e++)
                if (e != dim)
                    dims.emplace_back(e);
            timers->stopWatch(__LINE__, 1);
            if (dims.size() == 0 && shape::rank(inShape) == 1) { // split vector into lenthOf scalars
                //
                auto result = SHAPELIST();
                timers->stopWatch(__LINE__, 1);
                for (Nd4jLong e = 0; e < shape::length(inShape); e++)
                    result->push_back(ConstantShapeHelper::getInstance()->scalarShapeInfo(ArrayOptions::dataType(inShape)));
                timers->stopWatch(__LINE__, 1);
                return result;
            }
timers->stopWatch(__LINE__, 1);
            auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(inShape, dims);
timers->stopWatch(__LINE__, 1);
            auto numTads = tadPack.numberOfTads();
timers->stopWatch(__LINE__, 1);
            std::vector<Nd4jLong> shape(shape::rank(tadPack.primaryShapeInfo()));
timers->stopWatch(__LINE__, 1);
            for (int e = 0; e < shape::rank(tadPack.primaryShapeInfo()); e++)
                shape[e] = shape::shapeOf(tadPack.primaryShapeInfo())[e];
timers->stopWatch(__LINE__, 1);
            // remove leading and trailing 1
            if (inShape[0] == 2 && shape.size() == 2) {
                if (shape[0] == 1) {
                    shape.erase(shape.begin());
                } else if (shape[1] == 1) {
                    shape.erase(shape.end());
                }
            }
timers->stopWatch(__LINE__, 1);
            auto result = SHAPELIST();
timers->stopWatch(__LINE__, 1);
            for (int e = 0; e < numTads; e++) {
                auto newShape = ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inShape), shape::order(inShape), shape);
                result->push_back(newShape);
            }
timers->stopWatch(__LINE__, 1);
            return result;
        }

        DECLARE_TYPES(unstack) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_FLOATS, ALL_INTS})
                    ->setSameMode(true);
        }
    }
}

#endif
