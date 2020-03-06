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
#include <helpers/MmulHelper.h>
#include <array/NDArrayFactory.h>

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

    //
    // Input/output 2D arrays
    //
    template <typename T>
    static void computeTriangulars(sd::LaunchContext* context, NDArray const& input, NDArray& outputPlus, NDArray& outputMinus) {
        outputPlus.nullify();
        outputMinus.nullify();
        auto n = input.sizeAt(-1);
        for (auto r = 0; r < n; r++) {
            outputPlus.t<T>(r,r) = sd::math::nd4j_sqrt<T,T>(input.t<T>(r,r));
            outputMinus.t<T>(r,r) = sd::math::nd4j_sqrt<T,T>(input.t<T>(r,r));
        }
        for (auto r = 0; r < n; r++) {
            for (auto c = r + 1; c < n; c++) {
                auto sumPlus = T(0.f);
                auto sumMinus = T(0.f);
                for (auto j = r + 1; j < c; j++) {
                    sumPlus += outputPlus.t<T>(r, j) * outputPlus.t<T>(j, c);
                    sumMinus += outputMinus.t<T>(r, j) * outputMinus.t<T>(j, c);
                }
                outputPlus.t<T>(r,c) = (input.t<T>(r,c) - sumPlus) / (outputPlus.t<T>(r,r) + outputPlus.t<T>(c,c));
                outputMinus.t<T>(r,c) = (input.t<T>(r,c) - sumMinus) / (outputMinus.t<T>(r,r) + outputMinus.t<T>(c,c));
            }
        }
    }
    template <typename T>
    static void computeMarker(sd::LaunchContext* context, NDArray const& input, NDArray& outputMarker) {
        auto n = input.sizeAt(-1);
        outputMarker.nullify();

        for (auto j = 0; j < n; j++) {
            for (auto i = 0; i < j; i++) {
                outputMarker.t<T>(i,j) += math::nd4j_abs(input.t<T>(i,j));
            }
        }
    }

    template <typename T>
    static void fillUpTriangularOutput(LaunchContext* context, NDArray const& outputPlus, NDArray const& outputMinus,
            NDArray const& outputMarkerPlus, NDArray const& outputMarkerMinus, NDArray& output) {

        output.nullify();
        auto n = output.sizeAt(-1);

        for (auto j = 0; j < n; j++) {
            for (auto i = 0; i < j; i++) {
                if (outputMarkerMinus.t<T>(j) >= outputMarkerPlus.t<T>(j)) {
                    output.t<T>(i,j) = outputPlus.t<T>(i,j);
                }
                else {
                    output.t<T>(i,j) = outputMinus.t<T>(i,j);
                }
            }
        }
    }

    template <typename T>
    static void quasyTriangularCompute(sd::LaunchContext* context, NDArray const* inputR, NDArray* outputT) {
        auto inputTriangularPart = inputR->allTensorsAlongDimension({-2, -1});
        auto outputTriangularPart = outputT->allTensorsAlongDimension({-2, -1});
        auto n = inputR->sizeAt(-1);

        for (auto batch = 0; batch < inputTriangularPart.size(); ++batch) {
            auto input = inputTriangularPart[batch];
            auto output = outputTriangularPart[batch];
            auto outputPlus = output->ulike();
            auto outputMinus = output->ulike();
            computeTriangulars<T>(context, *input, outputPlus, outputMinus);
            auto outputMarkerPlus = NDArrayFactory::create<T>({n});
            auto outputMarkerMinus = outputMarkerPlus.ulike();
            computeMarker<T>(context, outputPlus, outputMarkerPlus);
            computeMarker<T>(context, outputMinus, outputMarkerMinus);
            fillUpTriangularOutput(context, outputPlus, outputMinus, outputMarkerPlus, outputMarkerMinus, *output);
        }
    }

    /*
     * Hessenberg reduction|decomposition
     * A = QHQ*, where Q - orthogonal, H - upper hessenberg quasytriangular matrix
     *
     * function HessenbergReduction( A::Matrix )
      //# Reduce A to a Hessenberg matrix H so that A and H are similar:

    n = A.rows() // n - rows()/columns()
    H = A
    if ( n > 2 ) // if input matrix more then 2x2
        a1 = A[2:n, 1] // select first column of the matrix
        e1 = zeros(n-1); e1[1] = 1 //e1 - orth with 1 on the first position
        sgn = sign(a1[1]) // -1 or +1 of the first matrix element (e.g. a[1,1])
        v = (a1 + sgn * norm(a1) * e1); v = v./norm(v) // Householder vector
        Q1 = eye(n-1) - 2*(v*v') // orthogonal matrix on step 1
        A[2:n,1] = Q1*A[2:n,1] // the first column of the matrix set up with proper multiplication
        A[1,2:n] = Q1*A[1,2:n] // the first row of the matrix set up with proper multiplication
        A[2:n,2:n] = Q1*A[2:n,2:n]*Q1' // reduce to rest (from the second row and the second column submatrice) and produce the step of transformation
        H = HessenbergReduction( A[2:n,2:n] ) // process all above for submatrix from the second row/column
    else
        H = copy(A) // only with matrix shape equals 2x2
    end
   return A
     * */

    template <typename T>
    void hessenbergReduction(NDArray const& input, NDArray& hessenberg) {
        auto n = input.rows();
        hessenberg.assign(input);
        if (n > 2) {
            auto a1 = hessenberg({1, n, 0, n}); // the first column
            auto c1 = hessenberg({1, n, 0, n}); // the first column
            auto r1 = hessenberg({0, n, 1, n}); // the first column

            auto e1 = NDArrayFactory::create<T>('c', {n - 1});
            e1.template t<T>(0) = T(1.f);
            auto sgn = math::nd4j_sign<T,T>(a1.t<T>(0));
            auto v = a1 + sgn * a1.reduceNumber(reduce::Norm1) * e1;
            v /= v.reduceNumber(reduce::Norm1);
            auto a2 = hessenberg({1, n-1, 1, n-1});
            auto h2 = hessenberg({1, n-1, 1, n-1});
            auto I = NDArrayFactory::create<T>('c', {n - 1, n - 1});
            I.setIdentity();
            auto V = I.ulike();
            MmulHelper::matmul(&v, &v, false, true);
            auto Q = I - T(2.f) * V;
            MmulHelper::matmul(&Q, &a1, &c1, false, false);
            MmulHelper::matmul(&Q, &r1, &r1, false, false);
            MmulHelper::matmul(&Q, &a2, &I, false, false);
            MmulHelper::matmul(&I, &Q, &a2, false, true);
            hessenbergReduction<T>(a2, h2);
        }
    }

    template <typename T>
    void schurDecomposition(sd::LaunchContext* context, NDArray const* input, NDArray* qMatrix, NDArray* tMatrix) {
        qMatrix->setIdentity();
        tMatrix->assign(input);
    }

    template <typename T>
    int sqrtMatrixFunctor_(sd::LaunchContext* context, NDArray const* input, NDArray* output) {
        auto pInput = const_cast<NDArray*>(input);
        auto outputQ = pInput->ulike();
        auto outputT = outputQ.ulike();

        schurDecomposition<T>(context, pInput, &outputQ, &outputT);

//        auto outputT = outputR.ulike();

        upperTriangularSqrt<T>(context, &outputT, output);
        MmulHelper::matmul(&outputQ, output, &outputT, false, false);
        MmulHelper::matmul(&outputT, &outputQ, output, false, true);

        return Status::OK();
    }

    int sqrtMatrixFunctor(sd::LaunchContext* context, NDArray const* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return sqrtMatrixFunctor_, (context, input, output), FLOAT_TYPES);
    }
}
}
}
