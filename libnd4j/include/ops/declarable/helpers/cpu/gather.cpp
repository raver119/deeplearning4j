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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 07.03.2019
//

#include <ops/declarable/helpers/gather.h>
#include <numeric>
#include <execution/Threads.h>
#include <helpers/ShapeUtils.h>
#include <helpers/ConstantTadHelper.h>
#include <performance/benchmarking/global_timers.h>

namespace sd {
namespace ops {
namespace helpers {

////////////////////////////////////////////////////////////////////////
void gather(sd::LaunchContext * context, const NDArray* input, const NDArray* indices, NDArray* output, const std::vector<int>& intArgs) {
    sd::GlobalTimers* timers = sd::GlobalTimers::getInstance();
    timers->stopWatch(__LINE__, 7);

    int axis = intArgs.size() > 0 ? intArgs[0] : 0;
    const int inputRank = input->rankOf();
    if(axis < 0)
        axis += inputRank;
    timers->stopWatch(__LINE__, 7);

    const int numOfIntArgs = intArgs.size();

    if (indices != nullptr) {

        timers->stopWatch(__LINE__, 7);
        // first case: indices consist of only one scalar
        if(indices->isScalar()) {

            if(input->rankOf() <= 1){
                //For scalar indices, rank 0 or 1 input: can't do tensor along dimension 0 as this is whole array... instead, we want to get a scalar
                timers->stopWatch(__LINE__, 7);
                auto idx = indices->e<Nd4jLong>(0);
                timers->stopWatch(__LINE__, 7);
                auto scalarNDArray = input->e(idx);
                timers->stopWatch(__LINE__, 7);
                output->assign(scalarNDArray);
                timers->stopWatch(__LINE__, 7);
            }
            else {
                timers->stopWatch(__LINE__, 7);
                NDArray inSubArr = (*input)(indices->e<Nd4jLong>(0), {axis});
                timers->stopWatch(__LINE__, 7);
                output->assign(inSubArr);
                timers->stopWatch(__LINE__, 7);
            }
        }
        else {
            if(input->rankOf() == 1 && output->rankOf() == 1) {

                timers->stopWatch(__LINE__, 7);
                auto func = PRAGMA_THREADS_FOR {
                    for (auto i = start; i < stop; i++)
                        output->p(i, input->e(indices->e<Nd4jLong>(i)));
                };

            timers->stopWatch(__LINE__, 7);
                samediff::Threads::parallel_for(func, 0, output->lengthOf());
                timers->stopWatch(__LINE__, 7);

            }
            else {

            timers->stopWatch(__LINE__, 7);
                std::vector<int> dimsOut;
                for (int i = 0; i < axis; ++i)
                    dimsOut.push_back(i);
                timers->stopWatch(__LINE__, 7);
                for (int i = axis+indices->rankOf(); i < output->rankOf(); ++i)
                    dimsOut.push_back(i);

                timers->stopWatch(__LINE__, 7);
                std::vector<int> dimsIn = ShapeUtils::evalDimsToExclude(input->rankOf(), {axis});
                timers->stopWatch(__LINE__, 7);

                const Nd4jLong numOfSubArrs = indices->lengthOf();
                timers->stopWatch(__LINE__, 7);

                auto inTadPack  = ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimsIn);
                timers->stopWatch(__LINE__, 7);
                auto outTadPack = ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimsOut);
                timers->stopWatch(__LINE__, 7);

                Nd4jLong* inTadShapeInfo  = inTadPack.primaryShapeInfo();
                timers->stopWatch(__LINE__, 7);
                Nd4jLong* outTadShapeInfo = outTadPack.primaryShapeInfo();
                timers->stopWatch(__LINE__, 7);

                if (shape::order(inTadShapeInfo) == shape::order(outTadShapeInfo) && shape::order(inTadShapeInfo) == 'c' && input->dataType() == output->dataType() && shape::elementWiseStride(inTadShapeInfo) == 1 && shape::elementWiseStride(outTadShapeInfo) == 1) {
                    timers->stopWatch(__LINE__, 7);

                    //auto func = PRAGMA_THREADS_FOR {

                        for (auto i = 0; i < numOfSubArrs; i++) {

                            void* inBuff  =  input->bufferWithOffset(inTadPack.primaryOffsets()[indices->e<Nd4jLong>(i)]);
                            void* outBuff = output->bufferWithOffset(outTadPack.primaryOffsets()[i]);

                            memcpy(outBuff, inBuff, shape::length(inTadShapeInfo) * input->sizeOfT());
                        }
                        timers->stopWatch(__LINE__, 7);
                    //};
                    //samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
                }
                else {
                    timers->stopWatch(__LINE__, 7);
                    auto func = PRAGMA_THREADS_FOR {
                        for (auto i = start; i < stop; i++) {

                            void* inBuff  =  input->bufferWithOffset(inTadPack.primaryOffsets()[indices->e<Nd4jLong>(i)]);
                            void* outBuff = output->bufferWithOffset(outTadPack.primaryOffsets()[i]);

                            NativeOpExecutioner::execTransformAny(input->getContext(), transform::Assign,
                                                                 inBuff,  inTadShapeInfo,  nullptr/*input specialBuffer*/, nullptr/*input specialShapeInfo*/,
                                                                 outBuff, outTadShapeInfo, nullptr/*output specialBuffer*/, nullptr/*output specialShapeInfo*/,
                                                                 nullptr, nullptr, nullptr, false/*allowParallelism*/);
                        }
                    };

                timers->stopWatch(__LINE__, 7);
                    samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
                    timers->stopWatch(__LINE__, 7);
                }
            }
        }
    }
    else {

        // we only allow scalar/vector case here
        if (numOfIntArgs == 2) { // scalar case

            timers->stopWatch(__LINE__, 7);
            output->assign((*input)(intArgs[1], {axis}));
            timers->stopWatch(__LINE__, 7);
        }
        else { // vector case

timers->stopWatch(__LINE__, 7);
            const Nd4jLong numOfSubArrs = intArgs.size() - 1;

            timers->stopWatch(__LINE__, 7);
            std::vector<int> dims  = ShapeUtils::evalDimsToExclude(input->rankOf(), {axis});
            timers->stopWatch(__LINE__, 7);

            auto inTadPack  = ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dims);
            timers->stopWatch(__LINE__, 7);
            auto outTadPack = ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dims);

            timers->stopWatch(__LINE__, 7);
            Nd4jLong* inTadShapeInfo  = inTadPack.primaryShapeInfo();
            timers->stopWatch(__LINE__, 7);
            Nd4jLong* outTadShapeInfo = outTadPack.primaryShapeInfo();

            timers->stopWatch(__LINE__, 7);
            if (shape::order(inTadShapeInfo) == shape::order(outTadShapeInfo) && shape::order(inTadShapeInfo) == 'c' && input->dataType() == output->dataType() && shape::elementWiseStride(inTadShapeInfo) == 1 && shape::elementWiseStride(outTadShapeInfo) == 1) {

                timers->stopWatch(__LINE__, 7);
                auto func = PRAGMA_THREADS_FOR {

                    for (auto i = start; i < stop; i++) {

                        void* inBuff  =  input->bufferWithOffset(inTadPack.primaryOffsets()[intArgs[i + 1]]);
                        void* outBuff = output->bufferWithOffset(outTadPack.primaryOffsets()[i]);

                        std::memcpy(outBuff, inBuff, shape::length(inTadShapeInfo) * input->sizeOfT());
                    }
                };
            timers->stopWatch(__LINE__, 7);
                samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
                timers->stopWatch(__LINE__, 7);

            }
            else {

timers->stopWatch(__LINE__, 7);
                auto func = PRAGMA_THREADS_FOR {

                    for (auto i = start; i < stop; i++) {

                        void* inBuff  =  input->bufferWithOffset(inTadPack.primaryOffsets()[intArgs[i + 1]]);
                        void* outBuff = output->bufferWithOffset(outTadPack.primaryOffsets()[i]);

                        NativeOpExecutioner::execTransformAny(input->getContext(), transform::Assign,
                                                             inBuff,  inTadShapeInfo,  nullptr/*input specialBuffer*/, nullptr/*input specialShapeInfo*/,
                                                             outBuff, outTadShapeInfo, nullptr/*output specialBuffer*/, nullptr/*output specialShapeInfo*/,
                                                             nullptr, nullptr, nullptr, false/*allowParallelism*/);

                    }
                };
timers->stopWatch(__LINE__, 7);
                samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
                timers->stopWatch(__LINE__, 7);
            }

        }
    }
}


}
}
}
