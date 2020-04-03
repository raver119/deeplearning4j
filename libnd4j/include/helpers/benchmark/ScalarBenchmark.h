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
#include "../OpBenchmark.h"

#ifndef SD_SCALARBENCHMARK_H
#define SD_SCALARBENCHMARK_H

using namespace sd::graph;

namespace sd {
    class SD_EXPORT ScalarBenchmark : public OpBenchmark {
    public:
        ScalarBenchmark() : OpBenchmark() {
            //
        }

        ~ScalarBenchmark(){

        }

        ScalarBenchmark(scalar::Ops op) : OpBenchmark() {
            _opNum = (int) op;
        }

        ScalarBenchmark(scalar::Ops op, const std::string &testName) : OpBenchmark() {
            _opNum = (int) op;
            _testName = testName;
        }

        ScalarBenchmark(scalar::Ops op, const std::string &testName, const NDArray &x, const NDArray &y, const NDArray &z) : OpBenchmark(testName, x, y, z) {
            _opNum = (int) op;
        }

        void executeOnce() override {
            PointersManager manager(LaunchContext::defaultContext(), "ScalarBM");

            if (_z.shapeInfo() == nullptr)
                NativeOpExecutioner::execScalar(LaunchContext::defaultContext(), _opNum, _x.buffer(), _x.shapeInfo(), _x.specialBuffer(), _x.specialShapeInfo(), _x.buffer(), _x.shapeInfo(), _x.specialBuffer(), _x.specialShapeInfo(), _y.buffer(), _y.shapeInfo(), _y.specialBuffer(), _y.specialShapeInfo(), nullptr);
            else
                NativeOpExecutioner::execScalar(LaunchContext::defaultContext(), _opNum, _x.buffer(), _x.shapeInfo(), _x.specialBuffer(), _x.specialShapeInfo(), _z.buffer(), _z.shapeInfo(), _z.specialBuffer(), _z.specialShapeInfo(), _y.buffer(), _y.shapeInfo(), _y.specialBuffer(), _y.specialShapeInfo(), nullptr);

            manager.synchronize();
        }

        std::string orders() override {
            std::string result;
            result += _x.ordering();
            result += "/";
            result += _z.shapeInfo() == nullptr ? _x.ordering() : _z.ordering();
            return result;
        }

        std::string strides() override {
            std::string result;
            result += ShapeUtils::strideAsString(_x);
            result += "/";
            result += _z.shapeInfo() == nullptr ? ShapeUtils::strideAsString(_x) : ShapeUtils::strideAsString(_z);
            return result;
        }

        std::string axis() override {
            return "N/A";
        }

        std::string inplace() override {
            return _x == _z ? "true" : "false";
        }

        OpBenchmark* clone() override  {
            return new ScalarBenchmark((scalar::Ops) _opNum, _testName, _x.shapeInfo() == nullptr ? _x : NDArray(_x.dup()) , _y.shapeInfo() == nullptr ? _y : NDArray(_y.dup()), _z.shapeInfo() == nullptr ? _z : NDArray(_z.dup()));
        }
    };
}

#endif //SD_SCALARBENCHMARK_H