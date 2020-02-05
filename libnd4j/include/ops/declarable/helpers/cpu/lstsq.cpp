/*******************************************************************************
 * Copyright (c) 2020 Konduit, K.K.
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
#include <op_boilerplate.h>
#include <NDArray.h>
#include <execution/Threads.h>
#include <MmulHelper.h>
#include <ShapeUtils.h>

#include "../lup.h"
#include "../triangular_solve.h"
#include "../lstsq.h"
#include "../qr.h"

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void fillRegularizer(NDArray& ioMatrix, double const value) {
        auto lastDims = ioMatrix.allTensorsAlongDimension({-2, -1});
        auto rows = ioMatrix.sizeAt(-2);
        //auto cols = ioMatrix.sizeAt(-1);

        for (auto x = 0; x < lastDims.size(); x++) {
            for (auto r = 0; r < rows; r++) {
                 lastDims[x]->t<T>(r,r) = (T)value;
            }
        }

    }

    template <typename T>
    int leastSquaresSolveFunctor_(nd4j::LaunchContext* context, NDArray const* leftInput, NDArray const* rightInput, double const l2Regularizer, bool const fast, NDArray* output) {
        NDArray::preparePrimaryUse({output}, {leftInput, rightInput});
        if (fast) { // Cholesky decomposition approach
            // Equation for solve A^T * Ax = A^T * b, so
            // 1. Computing A2:
            auto tAtShape = ShapeUtils::evalShapeForMatmul(leftInput->getShapeInfo(), leftInput->getShapeInfo(), true, false);
            //tAtShape[tAtShape.size() - 2] = output->sizeAt(-2);
            NDArray leftOutput('c', tAtShape, output->dataType(), context);
            MmulHelper::matmul(leftInput, leftInput, &leftOutput, true, false); // Computing A2 = A^T * A
            // 2. Computing B' = A^T * b
            auto rightOutput = output->ulike();

            MmulHelper::matmul(leftInput, rightInput, &rightOutput, true, false); // Computing B' = A^T * b
            // 3. due l2Regularizer = 0, skip regularization ( indeed A' = A2 - l2Regularizer * I)
            auto regularizer = leftOutput.ulike();
            fillRegularizer<T>(regularizer, l2Regularizer);
//            regularizer *= l2Regularizer;
            leftOutput += regularizer;
            // 4. Cholesky decomposition -- output matrix is square and lower triangular
//            auto leftOutputT = leftOutput.ulike();
            auto err = helpers::cholesky(context, &leftOutput, &leftOutput, true); // inplace decomposition
            if (err) return err;
            // alternate moment: inverse lower triangular matrix to solve equation A'x = b' => L^Tx = L^-1 * b'
            // solve one upper triangular system (to avoid float problems)
            
            // 5. Solve two triangular systems:
            auto rightB = rightOutput.ulike();
            helpers::triangularSolveFunctor(context, &leftOutput, &rightOutput, true, false, &rightB);
            rightB.printIndexedBuffer("Stage1 output");
            leftOutput.transposei();
            helpers::triangularSolveFunctor(context, &leftOutput, &rightB, false, false, output);
            // All done
        }
        else { // QR decomposition approach
            // Equation for solve Rx = Q^T * b, where A = Q * R, where Q - orthogonal matrix, and R - upper triangular
            // 1. QR decomposition
            auto qShape = leftInput->getShapeAsVector();
            auto rShape = leftInput->getShapeAsVector();
            qShape[leftInput->rankOf() - 1] = leftInput->sizeAt(-2);

            NDArray Q(leftInput->ordering(), qShape, leftInput->dataType(), context);// = leftInput->ulike();
            NDArray R(leftInput->ordering(), rShape, leftInput->dataType(), context); // = rightInput->ulike();
            helpers::qr(context, leftInput, &Q, &R, true);
            // 2. b` = Q^t * b:
            auto rightOutput = rightInput->ulike();
            MmulHelper::matmul(&Q, rightInput, &rightOutput, true, false);
            // 3. Solve triangular system
            helpers::triangularSolveFunctor(context, &R, &rightOutput, false, false, output);
        }
        NDArray::registerPrimaryUse({output}, {leftInput, rightInput});
        return Status::OK();
    }

    int leastSquaresSolveFunctor(nd4j::LaunchContext* context, NDArray const* leftInput, NDArray const* rightInput, double const l2Regularizer, bool const fast, NDArray* output) {
        BUILD_SINGLE_SELECTOR(leftInput->dataType(), return leastSquaresSolveFunctor_, (context, leftInput, rightInput, l2Regularizer, fast, output), FLOAT_TYPES);
    }

}
}
}
