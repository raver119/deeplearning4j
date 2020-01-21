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
                // setting current device
                cnrtDev_t dev;
                cnrtQueue_t queue;

                auto res = cnrtInit(0);
                if (res != CNRT_RET_SUCCESS)
                    throw std::runtime_error("MLU add: cnrtInit failed");

                res = cnrtGetDeviceHandle(&dev, 0);
                if (res != CNRT_RET_SUCCESS)
                    throw std::runtime_error("MLU add: cnrtGetDeviceHandle failed");

                res = cnrtSetCurrentDevice(dev);
                if (res != CNRT_RET_SUCCESS)
                    throw std::runtime_error("MLU add: cnrtSetCurrentDevice failed");

                res = cnrtCreateQueue(&queue);
                if (res != CNRT_RET_SUCCESS)
                    throw std::runtime_error("MLU add: cnrtCreateQueue failed");

                // creating tensors
                cnmlTensor_t input_tensor_1, input_tensor_2, output_tensor;
                cnmlTensorType_t type = CNML_TENSOR;
                auto status = cnmlCreateTensor_V2(&input_tensor_1, type);
                if (status != CNML_STATUS_SUCCESS)
                    throw std::runtime_error("MLU add: cnmlCreateTensor_V2 failed");

                cnmlCreateTensor_V2(&input_tensor_2, type);
                cnmlCreateTensor_V2(&output_tensor, type);

                int xShape[] = {x->sizeAt(0), x->sizeAt(1), x->sizeAt(2), x->sizeAt(3)};
                int yShape[] = {y->sizeAt(0), y->sizeAt(1), y->sizeAt(2), y->sizeAt(3)};
                int zShape[] = {z->sizeAt(0), z->sizeAt(1), z->sizeAt(2), z->sizeAt(3)};

                int xStrides[] = {x->strideAt(0), x->strideAt(1), x->strideAt(2), x->strideAt(3)};
                int yStrides[] = {y->strideAt(0), y->strideAt(1), y->strideAt(2), y->strideAt(3)};
                int zStrides[] = {z->strideAt(0), z->strideAt(1), z->strideAt(2), z->strideAt(3)};

                status = cnmlSetTensorShape_V2(input_tensor_1, 4, xShape, xStrides);
                if (status != CNML_STATUS_SUCCESS)
                    throw std::runtime_error("MLU add: cnmlSetTensorShape_V2 failed");

                cnmlSetTensorShape_V2(input_tensor_2, 4, yShape, yStrides);
                cnmlSetTensorShape_V2(output_tensor, 4, zShape, zStrides);

                status = cnmlSetTensorDataType(input_tensor_1, CNML_DATA_FLOAT32);
                if (status != CNML_STATUS_SUCCESS)
                    throw std::runtime_error("MLU add: cnmlSetTensorDataType failed");

                cnmlSetTensorDataType(input_tensor_2, CNML_DATA_FLOAT32);
                cnmlSetTensorDataType(output_tensor, CNML_DATA_FLOAT32);

                void *xBuffer, *yBuffer, *zBuffer;
                res = cnrtMalloc(&xBuffer, x->memoryFootprint());
                if (res != CNRT_RET_SUCCESS)
                    throw std::runtime_error("MLU add: cnrtMalloc failed");

                cnrtMalloc(&yBuffer, y->memoryFootprint());
                cnrtMalloc(&zBuffer, z->memoryFootprint());

                res = cnrtMemcpy(xBuffer, x->buffer(), x->memoryFootprint(), CNRT_MEM_TRANS_DIR_HOST2DEV);
                if (res != CNRT_RET_SUCCESS)
                    throw std::runtime_error("MLU add: cnrtMemcpy failed");

                cnrtMemcpy(yBuffer, y->buffer(), y->memoryFootprint(), CNRT_MEM_TRANS_DIR_HOST2DEV);
                cnrtMemcpy(zBuffer, z->buffer(), z->memoryFootprint(), CNRT_MEM_TRANS_DIR_HOST2DEV);

                // creating an op
                cnmlBaseOp_t op;
                status = cnmlCreateAddOp(&op, input_tensor_1, input_tensor_2, output_tensor);
                if (status != CNML_STATUS_SUCCESS)
                    throw std::runtime_error("MLU add: cnmlCreateAddOp failed");

                // executing an op
                status = cnmlComputeAddOpForward_V4(op, input_tensor_1, xBuffer, input_tensor_2, yBuffer, output_tensor, zBuffer, queue, nullptr);
                if (status != CNML_STATUS_SUCCESS)
                    throw std::runtime_error("MLU add: cnmlComputeAddOpForward_V4 failed");

                // sync the queue
                res = cnrtSyncQueue(queue);
                if (res != CNRT_RET_SUCCESS)
                    throw std::runtime_error("MLU add: cnrtSyncQueue failed");

                // FIXME: temporary code. we typically assume that arrays at this point
                res = cnrtMemcpy(z->buffer(), zBuffer, z->memoryFootprint(), CNRT_MEM_TRANS_DIR_DEV2HOST);
                if (res != CNRT_RET_SUCCESS)
                    throw std::runtime_error("MLU add: cnrtMemcpy final failed");

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