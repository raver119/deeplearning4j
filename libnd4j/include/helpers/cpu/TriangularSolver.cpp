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

#include <helpers/TriangularSolver.h>
#include <execution/Threads.h>

namespace sd      {
namespace ops     {
namespace helpers {

// vector case A{M,K} * x{K} = b{M}
//////////////////////////////////////////////////////////////////////////
template <typename T>
void TriangularSolver<T>::solveVector(const NDArray& A, const NDArray& b, const bool lower, const bool unitsOnDiag, NDArray& x) {

    if(lower) {

        for (int i = 0; i < A.sizeAt(0); ++i) {
            T sum = b.t<T>(i);
            for (int j = 0; j < i; ++j)
                sum -= A.t<T>(i,j) * x.t<T>(j);
            x.t<T>(i) = unitsOnDiag ? sum : sum / A.t<T>(i,i);
        }
    }
    else {

        for (int i = A.sizeAt(0) - 1; i >= 0; --i) {
            T sum = b.t<T>(i);
            for (int j = i + 1; j < A.sizeAt(1); ++j)
                sum -= A.t<T>(i,j) * x.t<T>(j);
            x.t<T>(i) = unitsOnDiag ? sum : sum / A.t<T>(i,i);
        }
    }
}

// general case A{M,K} * x{K,N} = b{M,N} or A{M,K} * x{K} = b{M}
//////////////////////////////////////////////////////////////////////////
template <typename T>
void TriangularSolver<T>::solve(const NDArray& A, const NDArray& b, const bool lower, const bool unitsOnDiag, NDArray& x) {

     if(A.rankOf() != 2)
        throw std::runtime_error("TriangularSolver::solve: input matrix A must be 2D !");

    int temp;

    const bool isBvector = b.isCommonVector(temp);
    const bool isXvector = x.isCommonVector(temp);

    if(A.sizeAt(0) != (isBvector ? b.lengthOf() : b.sizeAt(0)))
        throw std::runtime_error("TriangularSolver::solve: A and b must have the same number of rows !");

    if( A.sizeAt(1) != (isXvector ? x.lengthOf() : x.sizeAt(0)))
        throw std::runtime_error("TriangularSolver::solve: columns number of array A must be equal to rows number of array x !");

    if(isBvector) {
        TriangularSolver<T>::solveVector(A,b,lower,unitsOnDiag,x);
    }
    else {

        // if(lower) {

        //     for (int bCol = 0; bCol < b.sizeAt(1); ++bCol) {
        //         for (int i = 0; i < A.sizeAt(0); ++i) {
        //             T sum = b.t<T>(i, bCol);
        //             for (int j = 0; j < i; ++j)
        //                 sum -= A.t<T>(i,j) * x.t<T>(j, bCol);
        //             x.t<T>(i, bCol) = unitsOnDiag ? sum : sum / A.t<T>(i,i);
        //        }
        //     }
        // }
        // else {

        //     for (int bCol = 0; bCol < b.sizeAt(1); ++bCol) {
        //         for (int i = A.sizeAt(0) - 1; i >= 0; --i) {
        //             T sum = b.t<T>(i, bCol);
        //             for (int j = i + 1; j < A.sizeAt(1); ++j)
        //                 sum -= A.t<T>(i,j) * x.t<T>(j, bCol);
        //             x.t<T>(i, bCol) = unitsOnDiag ? sum : sum / A.t<T>(i,i);
        //         }
        //     }
        // }

        auto bSet = b.allTensorsAlongDimension({0});
        auto xSet = x.allTensorsAlongDimension({0});

        auto func = PRAGMA_THREADS_FOR {

            for (auto i = start; i < stop; i++)
                TriangularSolver<T>::solveVector(A,*bSet.at(i),lower,unitsOnDiag,*xSet.at(i));
        };

        samediff::Threads::parallel_tad(func, 0, bSet.size());
    }
}

template class ND4J_EXPORT TriangularSolver<float>;
template class ND4J_EXPORT TriangularSolver<float16>;
template class ND4J_EXPORT TriangularSolver<bfloat16>;
template class ND4J_EXPORT TriangularSolver<double>;

}
}
}