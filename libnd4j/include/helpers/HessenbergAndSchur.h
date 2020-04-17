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

#ifndef LIBND4J_HESSENBERGANDSCHUR_H
#define LIBND4J_HESSENBERGANDSCHUR_H

#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

// this class performs Hessenberg decomposition of square matrix using orthogonal similarity transformation
// A = Q H Q^T
// Q - orthogonal matrix
// H - Hessenberg matrix
template <typename T>
class Hessenberg {

    public:

        NDArray _Q;
        NDArray _H;

        explicit Hessenberg(const NDArray& matrix);

    private:
        void evalData();
};



}
}
}


#endif //LIBND4J_HESSENBERGANDSCHUR_H
