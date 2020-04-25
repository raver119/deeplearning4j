/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#ifndef LIBND4J_TRIANGULARSOLVER_H
#define LIBND4J_TRIANGULARSOLVER_H

#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

// this class solves equation A*x = b for x, assuming A is a triangular matrix, x and b may be vectors or 2d matrices
template <typename T>
class TriangularSolver {

    public:

        // A{M,K} * x{K} = b{M}, b and x are required to be vectors here
        // if lower = true then that means lower triangular matrix, otherwise matrix is considered as upper triangular
        // if unitsOnDiag = true, then we assume that A has unities on diagonal
        static void solveVector(const NDArray& A, const NDArray& b, const bool lower, const bool unitsOnDiag, NDArray& x);

        // general case A{M,K} * x{K,N} = b{M,N} or A{M,K} * x{K} = b{M}
        static void solve(const NDArray& A, const NDArray& b, const bool lower, const bool unitsOnDiag, NDArray& x);
};



}
}
}


#endif //LIBND4J_TRIANGULARSOLVER_H
