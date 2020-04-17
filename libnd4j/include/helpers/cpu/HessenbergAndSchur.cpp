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

#include <helpers/HessenbergAndSchur.h>
#include <helpers/householder.h>
#include <helpers/hhSequence.h>


namespace sd {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
Hessenberg<T>::Hessenberg(const NDArray& matrix) {

    if(matrix.rankOf() != 2)
        throw std::runtime_error("ops::helpers::Hessenberg constructor: input 2D matrix must be 2D matrix !");

    if(matrix.sizeAt(0) == 1) {
        _Q = NDArray(matrix.ordering(), {1,1}, matrix.dataType(), matrix.getContext());
        _Q = 1;
        _H = matrix.dup();
        return;
    }

    if(matrix.sizeAt(0) != matrix.sizeAt(1))
        throw std::runtime_error("ops::helpers::Hessenberg constructor: input array must be square 2D matrix !");

    _H = matrix.dup();
    _Q = matrix.ulike();

    evalData();
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void Hessenberg<T>::evalData() {

    const int rows = _H.sizeAt(0);

    NDArray hhCoeffs(_H.ordering(), {rows - 1}, _H.dataType(), _H.getContext());

    // calculate _H
    for(uint i = 0; i < rows - 1; ++i) {

        T coeff, norm;

        NDArray tail1 = _H({i+1,-1, i,i+1});
        NDArray tail2 = _H({i+2,-1, i,i+1}, true);

        Householder<T>::evalHHmatrixDataI(tail1, coeff, norm);

        _H({0,0, i,i+1}).t<T>(i+1) = norm;
        hhCoeffs.t<T>(i) = coeff;

        NDArray bottomRightCorner = _H({i+1,-1, i+1,-1}, true);
        Householder<T>::mulLeft(bottomRightCorner, tail2, coeff);

        NDArray rightCols = _H({0,0, i+1,-1}, true);
        Householder<T>::mulRight(rightCols, tail2.transpose(), coeff);
    }

    // calculate _Q
    HHsequence hhSeq(_H, hhCoeffs, 'u');
    hhSeq._diagSize = rows - 1;
    hhSeq._shift = 1;
    hhSeq.applyTo_<T>(_Q);

    // fill down with zeros starting at first subdiagonal
    _H.fillAsTriangular<T>(0, -1, 0, _H, 'l');
}

template class ND4J_EXPORT Hessenberg<float>;
template class ND4J_EXPORT Hessenberg<float16>;
template class ND4J_EXPORT Hessenberg<bfloat16>;
template class ND4J_EXPORT Hessenberg<double>;



    // NDArray colOfCoeffs(_HHmatrix.ordering(),  {diagSize, 1}, _HHmatrix.dataType(), _HHmatrix.getContext());

    // if(type == 'u')
    //     for(int i = 0; i < diagSize; ++i)
    //         colOfCoeffs.t<T>(i) = _HHmatrix.t<T>(i,i);
    // else
    //     for(int i = 0; i < diagSize; ++i)
    //         colOfCoeffs.t<T>(i) = _HHmatrix.t<T>(i,i+1);

    // HHsequence result(_HHmatrix, colOfCoeffs, type);

// HouseholderSequence(const VectorsType& v, const CoeffsType& h)                                                                                                                                    │
//             : m_vectors(v), m_coeffs(h), m_trans(false), m_length(v.diagonalSize()),                                                                                                                        │
//   >           m_shift(0)                                                                                                                                                                                    │
//           {                                                                                                                                                                                                 │
//           }

    // HouseholderSequenceType matrixQ() const
    // {
    //   eigen_assert(m_isInitialized && "HessenbergDecomposition is not initialized.");
    //   return HouseholderSequenceType(m_matrix, m_hCoeffs.conjugate())
    //          .setLength(m_matrix.rows() - 1)
    //          .setShift(1);
    // }
}
}
}

