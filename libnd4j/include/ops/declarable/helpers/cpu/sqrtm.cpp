/*******************************************************************************
 * Copyright (c) Konduit K.K.
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
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/sqrtm.h>
#include <ops/declarable/helpers/qr.h>

namespace sd {
namespace ops {
namespace helpers {

    template <typename T>
    void upperTriangularSqrt(sd::LaunchContext* context, NDArray const* inputTriangular, NDArray* outputTriangular) {
        auto n = inputTriangular->sizeAt(-1);
        auto inputTriangularPart = inputTriangular->allTensorsAlongDimension({-2, -1});
        auto outputTriangularPart = outputTriangular->allTensorsAlongDimension({-2, -1});

        for (auto batch = 0; batch < inputTriangularPart.size(); ++batch) {
            // compute diagonals
            auto input = inputTriangularPart[batch];
            auto output = outputTriangularPart[batch];
            for (auto r = 0; r < n; r++) {
                output->t<T>(r, r) = sd::math::nd4j_sqrt<T,T>(input->t<T>(r, r));
            }

            // compute upper diagonal
            for (auto r = 0; r < n - 1; r++) {
                output->t<T>(r, r + 1) = input->t<T>(r, r + 1) / (output->t<T>(r, r) + output->t<T>(r + 1, r + 1));
            }

            // loop for diagonals
            for (auto d = 2; d < n; d++) {
                for (auto r = 0; r < n - d; r++) {
                    auto sum = T(0.f);
                    for (auto k = r + 1; k < r + d; k++) {
                        sum += output->t<T>(r, k) * output->t<T>(k, d + r);
                    }
                    output->t<T>(r, r + d) = (input->t<T>(r, r + d) - sum) / (output->t<T>(r, r) + output->t<T>(r + d, r + d));
                }
            }
        }
    }

    template <typename T>
    int sqrtMatrixFunctor_(sd::LaunchContext* context, NDArray const* input, NDArray* output) {
        auto pInput = const_cast<NDArray*>(input);
        auto outputQ = pInput->ulike();
        auto outputR = outputQ.ulike();

        //helpers::qr(context, pInput, &outputQ, &outputR, false);

//        auto outputT = outputR.ulike();

        upperTriangularSqrt<T>(context, pInput, output);

        return Status::OK();
    }

    int sqrtMatrixFunctor(sd::LaunchContext* context, NDArray const* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return sqrtMatrixFunctor_, (context, input, output), FLOAT_TYPES);
    }
}
}
}
