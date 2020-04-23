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

#ifndef LIBND4J_EIGENVALSANDVECS_H
#define LIBND4J_EIGENVALSANDVECS_H

#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

// this class calculates eigenvalues and eigenvectors of given input matrix
template <typename T>
class EigenValsAndVecs {

    public:
        // suppose we got input square NxN matrix

        NDArray _Vals;      // vector of eigenvalues with shape {2,N}, 2 means real and imaginary part
        NDArray _Vecs;      // square NxN matrix of eigenvectors, whose columns are the eigenvectors.

        explicit EigenValsAndVecs(const NDArray& matrix);

    private:

        void calcEigenVals(const NDArray& schurMatrixT);
        void calcEigenVecs(NDArray& schurMatrixT);

};


}
}
}


#endif //LIBND4J_EIGENVALSANDVECS_H
