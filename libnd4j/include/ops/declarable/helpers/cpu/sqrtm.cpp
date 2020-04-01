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
#include <helpers/ShapeUtils.h>

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
    bool isDiagonal(NDArray const* matrix) {
         bool res = true;
         for (auto r = 0; r < matrix->rows(); r++)
             for (auto c = 0; c < matrix->columns(); c++) {
                 if (r == c) {
                     if (math::nd4j_abs(matrix->t<T>(r,c)) < T(1.e-5f)) return false;
                 }
                 else if (math::nd4j_abs(matrix->t<T>(r,c)) > T(1.e-5)) return false;
             }
         return res;
    }

    template <typename T>
    void hessenbergReduction(NDArray const& input, NDArray& hessenberg, NDArray& transformQ) {
        auto n = input.sizeAt(-1);
        hessenberg.assign(input);
        transformQ.setIdentity();
        if (n > 2) {
            auto a1 = hessenberg({1, n, 0, 1}).reshape('c', {n-1, 1}); // the first column shifted by 1
            auto c1 = hessenberg({1, n, 0, 1}); c1.reshapei({n-1, 1}); // the first column skipped the first
            auto r1 = hessenberg({0, 1, 1, n}).reshape('c', {1, n - 1}); // the first row skipped the first
            auto rr = hessenberg({0, 1, 1, n}); rr.reshapei({1, n-1});

            auto e1 = a1.ulike();//NDArrayFactory::create<T>('c', {n - 1});
            e1.template t<T>(0) = T(1.f);
            auto sgn = math::nd4j_sign<T,T>(a1.t<T>(0));
            auto v = a1 + sgn * a1.reduceNumber(reduce::Norm2) * e1;
            v /= v.reduceNumber(reduce::Norm2);
            auto a2 = hessenberg({1, n, 1, n});
            auto h2 = hessenberg({1, n, 1, n});
            auto I = NDArrayFactory::create<T>('c', {n - 1, n - 1});
            auto cr = NDArrayFactory::create<T>('c', {n - 1, 1});
            auto rc = NDArrayFactory::create<T>('c', {1, n - 1});
            I.setIdentity();
            auto V = I.ulike();
            v.reshapei({n - 1, 1});rr.reshapei({1, n-1});
            //auto v1 = v.reshape({1, n - 1});
            MmulHelper::matmul(&v, &v, &V, false, true);
            auto Q = transformQ({1, n, 1, n});
            I -= T(2.f)* V; V.nullify();

            MmulHelper::matmul(&I, &a1, &cr, false, false);c1.assign(cr);
            MmulHelper::matmul(&r1, &I, &rc, false, false);rr.assign(rc);
            MmulHelper::matmul(&I, &a2, &V, false, false);
            MmulHelper::matmul(&V, &I, &a2, false, true);

            V.assign(I); //I.setIdentity();
            hessenbergReduction<T>(a2, h2, I);
            MmulHelper::matmul(&V, &I, &Q, false, false);
        }
    }

    /*
     * When real schur decomposition the diagonal elements are 1x1 or 2x2 (when complex eigenvals)
     * to process complex eigenvals matrix (2x2) follow procedure
     * */
    template <typename T>
    void complexEigenSqrt(sd::LaunchContext* context, NDArray const* input, NDArray* output) {
        auto r11 = input->t<T>(0, 0);
        auto r22 = input->t<T>(1, 1);
        auto r12 = input->t<T>(0, 1);
        auto r21 = input->t<T>(1, 0);
        auto theta = (r11 + r22) / T(2.f);
        auto mu = math::p_sqrt(-(r11 - r22) * (r11 - r22) - T(4.f) * r21 * r12) / T(2.f);
        auto alpha = (theta > 0)?math::p_sqrt((theta + math::p_sqrt(theta * theta + mu * mu))/T(2.f)):mu / math::p_sqrt(T(2.f) * (-theta + math::p_sqrt(theta*theta + mu * mu)));

        output->t<T>(0,0) = alpha + (r11 - r22)/(T(4.f)* alpha);
        output->t<T>(1,1) = alpha - (r11 - r22)/(T(4.f)* alpha);

        output->t<T>(0,1) = r12/(T(2.f)* alpha);
        output->t<T>(1,0) = r21/(T(2.f)* alpha);
    }


    template <typename T>
    void francisStep(sd::LaunchContext* context, NDArray const* input, NDArray* qMatrix, NDArray* tMatrix) {
         auto n = input->sizeAt(-1);
         auto p = n - 1;

         while(p > 1) {
             auto q = p - 1;
             auto s = input->t<T>(q,q) + input->t<T>(p,p);
             auto t = input->t<T>(q,q) * input->t<T>(p,p) - input->t<T>(q,p) * input->t<T>(p,q);
             auto x = input->t<T>(0,0) * input->t<T>(0,0) + input->t<T>(0,1) * input->t<T>(1,0) - s * input->t<T>(0,0) + t;
             auto y = input->t<T>(1,0) * (input->t<T>(0,0) + input->t<T>(1,1) - s);
             auto z = input->t<t>(1,0) * input->t<T>(2,1);

             for (auto k = 0; k < p-2; k++) {

             }
         }
    }

    /** \internal Look for single small sub-diagonal element and returns its index
     *
     * @tparam MatrixType
     * @param iu
     * @return
     */
    template<typename T>
    static inline Nd4jLong findSmallerSubdiagonalEntry(NDArray const* matrix, Nd4jLong initialIndex)
    {
        Nd4jLong res = initialIndex;
        T const epsilon = T(1.e-5);

        while (res > 0)
        {
            auto s = math::nd4j_abs(matrix->t<T>(res-1,res-1)) + abs(matrix->t<T>(res,res));
            if (math::nd4j_abs(matrix->t<T>(res,res-1)) <= epsilon * s)
                break;
            res--;
        }
        return res;
    }

    template <typename T>
    struct GivenceRotate {
         T _c, _s;

         void rotate(T const p, T const q, T* r = nullptr) {
            if(q == T(0.f))
            {
                _c = math::nd4j_sign<T,T>(p);// < T(0.f) ? Scalar(-1) : Scalar(1);
                _s = T(0.f);
                //*r = math::nd4j_abs(p);
            }
            else if( p == T(0.f))
            {
                _c = p;
                _s = -math::nd4j_sign<T,T>(q);
                //*r = math::nd4j_abs(q);
            }
            else if(math::nd4j_abs(p) > math::nd4j_abs(q))
            {
                T t = q/p;
                T u = math::p_sqrt(T(1.f) + t * t);
                if(p < T(0.f))
                    u = -u;
                _c = T(1.f) / u;
                _s = -t * _c;
                //if(r) *r = p * u;
            }
            else
            {
                T t = p/q;
                T u = math::p_sqrt(T(1.f) + t*t);
                if(q < T(0.f))
                    u = -u;
                _s = - T(1.f) / u;
                _c = -t * _s;
                //*r = q * u;
            }
        }

        GivenceRotate<T> adjointRotate() const{
             GivenceRotate<T> res;
             res._c = _c;
             res._s = -_s;

             return res;
         }

    };

    /* applyOnTheLeft
    RowXpr x(this->row(p)); // retrieve p-row from matrix
    RowXpr y(this->row(q)); // retrieve q-row from matrix
    internal::apply_rotation_in_the_plane(x, y, j); // rotate with given c and s params


     *
     * applyOnTheRight
     *  RowXpr x(this->row(p)); // retrieve p-row from matrix  = *ioMatrixT({p, p + 1, 0, n})
     *  RowXpr y(this->row(q)); // retrieve q-row from matrix  = *ioMatrixT({q, q + 1, 0, n})
     *  internal::apply_rotation_in_the_plane(x, y, j.transpose()); // j.transpose() == j.adjointRotate() // with c and -s params
     *
     *  ---------------------------------------------------
     *  internal::apply_rotation_in_the_plane:
     *
     * template<typename VectorX, typename VectorY, typename OtherScalar>
     * void  apply_rotation_in_the_plane(DenseBase<VectorX>& xpr_x, DenseBase<VectorY>& xpr_y, const JacobiRotation<OtherScalar>& j)
     * {
     *   typedef typename VectorX::Scalar Scalar;
     *   const bool Vectorizable =    (VectorX::Flags & VectorY::Flags & PacketAccessBit)
     *                            && (int(packet_traits<Scalar>::size) == int(packet_traits<OtherScalar>::size));
     *
     * eigen_assert(xpr_x.size() == xpr_y.size());
     * Index size = xpr_x.size();
     * Index incrx = xpr_x.derived().innerStride();
     * Index incry = xpr_y.derived().innerStride();
     *
     * Scalar* EIGEN_RESTRICT x = &xpr_x.derived().coeffRef(0);
     * Scalar* EIGEN_RESTRICT y = &xpr_y.derived().coeffRef(0);
     *
     * OtherScalar c = j.c();
     * OtherScalar s = j.s();
     * if (c==OtherScalar(1) && s==OtherScalar(0))
     * return;
     *
     * apply_rotation_in_the_plane_selector<
     *       Scalar,OtherScalar,
     *       VectorX::SizeAtCompileTime,
     *       EIGEN_PLAIN_ENUM_MIN(evaluator<VectorX>::Alignment, evaluator<VectorY>::Alignment),
     *       Vectorizable>::run(x,incrx,y,incry,size,c,s);
     * }
     *
     *  ---------------------------------------------------
     */

//    template<typename X, typename Y>
//    struct ApplyRotationInThePlaneSelector {
//        void operator()(X *x, Nd4jLong incrx, X *y, Nd4jLong incry, Nd4jLong size, Y c, Y s) {
//            for(Nd4jLong i=0; i<size; ++i)
//            {
//                X xi = *x;
//                X yi = *y;
//                *x =  c * xi + s * yi;
//                *y = -s * xi + c * yi;
//                x += incrx;
//                y += incry;
//            }
//        }
//    };

    template <typename T>
    static void applyGivenceRotate(NDArray& xRow, NDArray& yRow, const GivenceRotate<T>& rotation) {
        T c = rotation._c;
        T s = rotation._s;
        auto size = xRow.lengthOf();
        for(Nd4jLong i=0; i < size; ++i) {
            xRow.t<T>(i) = c * xRow.t<T>(i) + s * yRow.t<T>(i);
            yRow.t<T>(i) = -s * xRow.t<T>(i) + c * yRow.t<T>(i);
        }
    }

/** \internal Update T given that rows initialIndex - 1 and initialIndex decouple from the rest. *
 *
 * @tparam ioMatrixT - hessenberg reduced input
 * @param ioMatrixQ - transformation matrix as H = QTQ*
 * @param initialIndex - given row for process
 * @param exshift - shift for Francis QR step
 */
template<typename T>
    static inline  void splitOffTwoRows(NDArray* ioMatrixT, NDArray* ioMatrixQ, Nd4jLong initialIndex, const T exshift)
    {
//!  CAUSSION:  initialIndex should be > 0
        const auto size = ioMatrixT->sizeAt(-1);

        // The eigenvalues of the 2x2 matrix [a b; c d] are
        // trace +/- sqrt(discr/4) where discr = tr^2 - 4*det, tr = a + d, det = ad - bc
        T p = T(0.5f) * (ioMatrixT->t<T>(initialIndex - 1, initialIndex - 1) - ioMatrixT->t<T>(initialIndex,initialIndex));
        T q = p * p + ioMatrixT->t<T>(initialIndex, initialIndex - 1) * ioMatrixT->t<T>(initialIndex - 1, initialIndex);   // q = tr^2 / 4 - det = discr/4
        ioMatrixT->t<T>(initialIndex,initialIndex) += exshift;
        ioMatrixT->t<T>(initialIndex - 1, initialIndex - 1) += exshift;

        if (q >= T(0.f)) // Two real eigenvalues
        {
            T z = math::p_sqrt(math::nd4j_abs(q));
            // Givens rotation:
            GivenceRotate<T> rot;
            if (p >= T(0.f))
                rot.rotate(p + z, ioMatrixT->t<T>(initialIndex, initialIndex - 1));
            else
                rot.rotate(p - z, ioMatrixT->t<T>(initialIndex, initialIndex - 1));
            auto rightCols = (*ioMatrixT)({0, 0, size-initialIndex + 1, size}); // set of right columns to rotate by givens
            auto topRows = (*ioMatrixT)({0, initialIndex + 1, 0, 0}); // set of upper rows to rotate by givens

            // rotate rows with shifts
            auto xRow = rightCols({initialIndex -1, initialIndex, 0, 0});
            auto yRow = rightCols({initialIndex, initialIndex + 1, 0, 0});
            applyGivenceRotate(xRow, yRow, rot.adjointRotate());

            // rotate rows without shifts
            xRow = topRows({initialIndex - 1, initialIndex, 0, 0});
            yRow = topRows({initialIndex, initialIndex + 1, 0, 0});
            applyGivenceRotate(xRow, yRow, rot.adjointRotate());

            ioMatrixT->t<T>(initialIndex, initialIndex - 1) = T(0.f);

            // rotate transformation matrix rows
            xRow = (*ioMatrixQ)({initialIndex - 1, initialIndex, 0, 0});
            yRow = (*ioMatrixQ)({initialIndex, initialIndex + 1, 0, 0});
            applyGivenceRotate(xRow, yRow, rot.adjointRotate());
        }

        if (initialIndex > 1) // for next bands
            ioMatrixT->t<T>(initialIndex - 1, initialIndex - 2) = T(0.f);
    }

    template <typename T>
    struct ShiftInfo {
                T x,y,z;
            };

    /** \internal Form shift in shiftInfo, and update exshift if an exceptional shift is performed. */
    template<typename T>
    inline void computeShift(NDArray& inputMatrix, Nd4jLong initialIndex, Nd4jLong iteration, T& exshift, ShiftInfo<T>& shiftInfo) {
        shiftInfo.x = inputMatrix.t<T>(initialIndex, initialIndex);
        shiftInfo.y = inputMatrix.t<T>(initialIndex - 1, initialIndex - 1);
        shiftInfo.z = inputMatrix.t<T>(initialIndex, initialIndex - 1) * inputMatrix.t<T>(initialIndex - 1, initialIndex);

        // Wilkinson's original ad hoc shift
        if (iteration == 10) {
            exshift += shiftInfo.x;
            for (Nd4jLong i = 0; i <= initialIndex; ++i)
                inputMatrix.t<T>(i, i) -= shiftInfo.x;
            T wilkinsonShift = math::nd4j_abs(inputMatrix.t<T>(initialIndex, initialIndex - 1)) + math::nd4j_abs(inputMatrix.t<T>(initialIndex - 1, initialIndex - 2));
            shiftInfo.x = T(0.75) * wilkinsonShift;
            shiftInfo.y = T(0.75) * wilkinsonShift;
            shiftInfo.z = T(-0.4375) * wilkinsonShift * wilkinsonShift;
        }

        // MATLAB's new ad hoc shift
        if (iteration == 30)
        {
            T wilkinsonShift = (shiftInfo.y - shiftInfo.x) / T(2.0);
            wilkinsonShift = wilkinsonShift * wilkinsonShift + shiftInfo.z;
            if (wilkinsonShift > T(0.f))
            {
                wilkinsonShift = math::p_sqrt(wilkinsonShift);
                if (shiftInfo.y < shiftInfo.x)
                    wilkinsonShift = -wilkinsonShift;
                wilkinsonShift = wilkinsonShift + (shiftInfo.y - shiftInfo.x) / T(2.0);
                wilkinsonShift = shiftInfo.x - shiftInfo.z / wilkinsonShift;
                exshift += wilkinsonShift;
                for (Nd4jLong i = 0; i <= initialIndex; ++i)
                    inputMatrix.t<T>(i, i) -= wilkinsonShift;
                shiftInfo.x = shiftInfo.y = shiftInfo.z = T(0.964); // wilkinson shift after the 30th iteration
            }
        }
    }

    template<typename T>
    void makeHouseholder(NDArray& aThis, NDArray& essential, T& tau, T& beta) {
        auto n = aThis.lengthOf();
        auto tail = aThis({1, n, 0, 0});//derived(), 1, size()-1);
        T tailSqNorm = n == 1 ? T(0.f) : tail.reduceNumber(reduce::SquaredNorm).t<T>(0);
        T c0 = aThis.t<T>(0);
        const T tol = DataTypeUtils::min<T>(); //(std::numeric_limits<RealScalar>::min)();

        if(tailSqNorm <= tol)     {
            tau = T(0.f);
            beta = c0;
            essential.nullify();
        }
        else {
            beta = math::p_sqrt( c0 * c0 + tailSqNorm);
            if (c0 >= T(0.f))
                beta = -beta;
            essential = tail / (c0 - beta);
            tau = (beta - c0) / beta;
        }
    }

/** \internal Compute index im at which Francis QR step starts and the first Householder vector. */
    template<typename T>
    inline void initFrancisQRStep(NDArray const& inputMatrix, Nd4jLong lowerIndex, Nd4jLong upperIndex, const ShiftInfo<T>& shiftInfo, Nd4jLong& francisIndex, ShiftInfo<T>& firstHouseholderVector) {
        for (francisIndex = upperIndex - 2; francisIndex >= lowerIndex; --francisIndex) {
            const T Tmm = inputMatrix.t<T>(francisIndex, francisIndex);
            const T r = shiftInfo.x - Tmm;
            const T s = shiftInfo.y - Tmm;
            firstHouseholderVector.x = (r * s - shiftInfo.z) / inputMatrix.t<T>(francisIndex + 1, francisIndex) + inputMatrix.t<T>(francisIndex, francisIndex + 1);
            firstHouseholderVector.y = inputMatrix.t<T>(francisIndex + 1, francisIndex + 1) - Tmm - r - s;
            firstHouseholderVector.z = inputMatrix.t<T>(francisIndex + 2, francisIndex + 1);
            if (francisIndex == lowerIndex) {
                break;
            }
            const T lhs = inputMatrix.t<T>(francisIndex, francisIndex - 1) * (math::nd4j_abs(firstHouseholderVector.y) + math::nd4j_abs(firstHouseholderVector.z));
            const T rhs = firstHouseholderVector.x * (math::nd4j_abs(inputMatrix.t<T>(francisIndex - 1, francisIndex - 1)) + math::nd4j_abs(Tmm) + abs(inputMatrix.t<T>(francisIndex + 1, francisIndex + 1)));
            if (math::nd4j_abs(lhs) < T(DataTypeUtils::min<T>() * rhs))
                break;
        }
    }

    template <typename T>
    static void applyHouseholderOnTheLeft( NDArray& ioMatrix,  NDArray const& essential, T const& tau) {
        if(ioMatrix.rows() == 1) { // when the vector-row
            ioMatrix *= T(1.f) - tau;
        }
        else if(tau != T(0.f)) {
            auto n = ioMatrix.sizeAt(-1);
            auto m = ioMatrix.sizeAt(-2);
            //Map<typename internal::plain_row_type<PlainObject>::type> tmp(workspace,cols());
            //Block<Derived, EssentialPart::SizeAtCompileTime, Derived::ColsAtCompileTime> bottom(derived(), 1, 0, rows()-1, cols());
            auto bottom = ioMatrix({1, m, 0, n}); // all rows from the second
            auto tmp = NDArrayFactory::create<T>('c', {1, n});
            if(essential.lengthOf() == 1) {
                tmp = essential * bottom;
            }
            else
            MmulHelper::matmul(&essential, &bottom, &tmp, true, false);
            tmp += ioMatrix({0, 1, 0, 0});//this->row(0);
            ioMatrix({0, 1, 0, 0})  -= tau * tmp;
            auto bottomTemp = bottom.ulike();
            if (essential.lengthOf() == 1)
                bottomTemp = essential * tmp;
            else
            MmulHelper::matmul(&essential, &tmp, &bottomTemp, false, false);
            bottom -= tau * bottomTemp;
        }
    }

    template<typename T>
    static void applyHouseholderOnTheRight(NDArray& ioMatrix, const NDArray& essential, const T& tau) {
        auto m = ioMatrix.sizeAt(-2);
        auto n = ioMatrix.sizeAt(-1);

        if(ioMatrix.sizeAt(-1) == 1) {
            ioMatrix *= T(1.f) - tau;
        }
        else if(tau != T(0.f)) {
            auto right = ioMatrix({0, m, 1, n}); // input matrix from the second column //(derived(), 0, 1, rows(), cols()-1);

            auto tmpShape = essential.lengthOf() > 1?ShapeUtils::evalShapeForMatmul(right.getShapeInfo(), essential.getShapeInfo(), false, false):right.getShapeAsVector();
            auto tmp = NDArrayFactory::create<T>('c', tmpShape);//right * essential.transpose();

            if (essential.lengthOf() == 1)
                tmp = right * essential;
            else
            MmulHelper::matmul(&right, &essential, &tmp, false, false);
            auto row = ioMatrix({0, m, 0, 1}).reshape('c', tmpShape);
            tmp += row; // the first column //this->col(0);
            ioMatrix({0, 0, 0, 1}) -= tau * tmp;
            auto tempE = right.ulike();
            if (essential.lengthOf() == 1)
                tempE = tmp * essential;
            else
            MmulHelper::matmul(&tmp, &essential, &tempE, false, true);
            right -= tau * tempE;
        }
    }


/** \internal Perform a Francis QR step involving rows il:iu and columns im:iu. */
    template<typename T>
    inline void performFrancisQRStep(NDArray& ioMatrixT, NDArray& ioMatrixU, Nd4jLong indexLower, Nd4jLong indexMedium, Nd4jLong indexUpper, const ShiftInfo<T>& firstHouseholderVector) {
        const Nd4jLong n = ioMatrixT.sizeAt(-1);

        for (Nd4jLong k = indexMedium; k <= indexUpper - 2; ++k)
        {
            bool firstIteration = (k == indexMedium);

            ShiftInfo<T> v;
            if (firstIteration)
                v = firstHouseholderVector;
            else {
                v.x = ioMatrixT.t<T>(k, k - 1);// = m_matT({k, k + 3, k - 1, k});//.template block<3,1>(k,k-1);
                v.y = ioMatrixT.t<T>(k + 1, k - 1);
                v.z = ioMatrixT.t<T>(k + 2, k - 1);
            }

            T tau, beta;
            NDArray ess('c', {2, 1}, DataTypeUtils::fromT<T>());
            NDArray vArr = NDArrayFactory::create<T>('c', {3, 1}, {v.x, v.y, v.z});
            makeHouseholder<T>(vArr, ess, tau, beta);

            if (beta != T(0.f)) {// if v is not zero
                if (firstIteration && k > indexLower)
                    ioMatrixT.t<T>(k, k - 1) = -ioMatrixT.t<T>(k, k - 1);
                else if (!firstIteration)
                    ioMatrixT.t<T>(k, k - 1) = beta;

                // These Householder transformations form the O(n^3) part of the algorithm
                auto matTKK3 = ioMatrixT({k, k + 3, k, n}); //
                applyHouseholderOnTheLeft<T>(matTKK3, ess, tau);
                auto bottomRow = math::nd4j_min(indexUpper, k + 3);
                auto matTK3 = ioMatrixT({0, bottomRow, k, k + 3});
                applyHouseholderOnTheRight<T>(matTK3, ess, tau);
                auto matT3 = ioMatrixU({0, n, k, k + 3});
                applyHouseholderOnTheRight<T>(matT3, ess, tau);
            }
        }

        auto v = ioMatrixT({indexUpper - 1, indexUpper + 1, indexUpper - 2, indexUpper - 1}).reshape('c', {2,1});//.template block<2,1>(iu-1, iu-2); /// 2x1 block with iu-1 pos
        T tau, beta;
        NDArray ess = NDArrayFactory::create<T>(0.f);
        makeHouseholder(v, ess, tau, beta);
        if (beta != T(0)) { // if v is not zero
            ioMatrixT.t<T>(indexUpper - 1, indexUpper - 2) = beta;
            auto matT1 = ioMatrixT({indexUpper - 1, indexUpper + 1, indexUpper - 1, n});
            applyHouseholderOnTheLeft(matT1, ess, tau);
            auto matTR1 = ioMatrixT({0, indexUpper + 1, indexUpper - 1, indexUpper + 1});
            applyHouseholderOnTheRight(matTR1, ess, tau);
            auto matU1 = ioMatrixU({0, n, indexUpper - 1, indexUpper + 1});
            applyHouseholderOnTheRight(matU1, ess, tau);
        }

        // clean up pollution due to round-off errors
        for (Nd4jLong i = indexMedium + 2; i <= indexUpper; ++i) {
            ioMatrixT.t<T>(i, i - 2) = T(0.f);
            if (i > indexMedium + 2)
                ioMatrixT.t<T>(i, i - 3) = T(0.f);
        }
    }

        /*
     * real schur decomposition algorithm:
     * 1) Reduce input matrix to hessenberg form with housholder transformation
     * 2) Use Francis double shift algorithm to compute decomposition
     * 3) Accumulate transformation matrix QU: A = QHQ^T; H = UTU^T => A = UTU^TQ^T
     *
     * */

    template <typename T>
    void schurDecomposition(sd::LaunchContext* context, NDArray const* input, NDArray* qMatrix, NDArray* tMatrix) {

    }
        /** \internal Computes and returns vector L1 norm of T */
    template<typename T>
    static T computeNormOfT(NDArray const& m_matT) {
        const auto size = m_matT.columns();
        // FIXME to be efficient the following would requires a triangular reduxion code
        // Scalar norm = m_matT.upper().cwiseAbs().sum()
        //               + m_matT.bottomLeftCorner(size-1,size-1).diagonal().cwiseAbs().sum();
        T norm = T(0.f);
        for (auto j = 0; j < size; ++j)
            norm += m_matT({0, math::nd4j_min<Nd4jLong>(size, j + 2), j, j + 1}).reduceNumber(reduce::Norm1).t<T>(0);
        return norm;
    }

    template<typename T>
    void computeFromHessenberg(const NDArray& matrixH, const NDArray& matrixQ, NDArray& m_matT, NDArray& m_matU) {

        m_matT.assign(matrixH);
        m_matU.assign(matrixQ);

        Nd4jULong maxIters = m_matT.rows() * 40;// max count of iterations

        // The matrix m_matT is divided in three parts.
        // Rows 0,...,il-1 are decoupled from the rest because m_matT(il,il-1) is zero.
        // Rows il,...,iu is the part we are working on (the active window).
        // Rows iu+1,...,end are already brought in triangular form.
        Nd4jLong indexUpper = m_matT.sizeAt(-1) - 1;
        Nd4jLong iter = 0;      // iteration count for current eigenvalue
        Nd4jLong totalIter = 0; // iteration count for whole matrix
        T exshift(0.f);   // sum of exceptional shifts
        T norm = computeNormOfT<T>(matrixH);
        if(norm != T(0.f))  { // check for inversibility
            while (indexUpper >= 0) {
                auto indexLower = findSmallerSubdiagonalEntry<T>(&m_matT, indexUpper);
//                nd4j_printf("Upper index is %lld and lower is %lld\n", indexUpper, indexLower);
                // Check for convergence
                if (indexLower == indexUpper) {// One root found
                    m_matT.t<T>(indexUpper, indexUpper) = m_matT.t<T>(indexUpper, indexUpper) + exshift;
                    if (indexUpper > 0)
                        m_matT.t<T>(indexUpper, indexUpper - 1) = T(0.f);
                    indexUpper--;
                    iter = 0;
                }
                else if (indexLower == indexUpper - 1) { // Two roots found - the 2x2 band
                    splitOffTwoRows<T>(&m_matT, &m_matU, indexUpper, exshift);
                    indexUpper -= 2;
                    iter = 0;
                }
                else { // No convergence yet
                    // The firstHouseholderVector vector has to be initialized to something to get rid of a silly GCC warning (-O1 -Wall -DNDEBUG )
                    ShiftInfo<T> firstHouseholderVector = {0}, shiftInfo;
                    computeShift<T>(m_matT, indexUpper, iter, exshift, shiftInfo);
                    iter++;
                    totalIter++;
                    if (totalIter > maxIters) break;
                    Nd4jLong indexMedium;
                    initFrancisQRStep<T>(m_matT, indexLower, indexUpper, shiftInfo, indexMedium, firstHouseholderVector);
                    performFrancisQRStep<T>(m_matT, m_matU, indexLower, indexMedium, indexUpper, firstHouseholderVector);
                }
            }
        }

        if(totalIter > maxIters)
            throw std::runtime_error("schur decomposition: Input matrix does not convergence");
    }


    template <typename T>
    void primitiveSchurDecomposition(sd::LaunchContext* context, NDArray const* input, NDArray* qMatrix, NDArray* tMatrix) {
        tMatrix->assign(input);
        auto k = 0;
        auto resQ = qMatrix->ulike();
        qMatrix->setIdentity();
        auto const kMaxIteration = 40;

//        auto shiftProc = []( T a, T b, T c ) -> T {
//            auto middle = (a - c) / T(2.f);
//            return c - math::nd4j_sign<T,T>(middle) * b * b / (math::nd4j_abs(middle) + math::nd4j_sqrt<T,T>(middle * middle + b * b));
//        };
        auto n = input->sizeAt(-1);
        do {
            auto temp = tMatrix->ulike(); temp.nullify();
            auto tempQ(*qMatrix);
//            auto wilkisonShift = shiftProc(tMatrix->t<T>(n - 2, n - 2), tMatrix->t<T>(n - 1, n - 1), tMatrix->t<T>(n - 2, n - 1));
//            auto tempT(*tMatrix);
//            auto wI = tempT.ulike(); wI.setIdentity(); wI *= wilkisonShift;
//            tempT -= wI;
            helpers::qr(context, tMatrix, &resQ, &temp, false);
            MmulHelper::matmul(&temp, &resQ, tMatrix, false, false);
            tMatrix->printIndexedBuffer("Upper triangle");
//            tMatrix->assign(tempT + wI);
            resQ.printIndexedBuffer("Orthogonal");
            k++;
            nd4j_printf("Iteration %d\n", k)
            ;
            MmulHelper::matmul(&tempQ, &resQ, qMatrix, false, false);

        }
        while (!isDiagonal<T>(&resQ) && k < n * kMaxIteration);
    }

    /*
     * Lemma. the upper triangular matrix with eigen vals has sqrt only when all real eigenvals are positive
     * To check this:
     * */
    template <typename T>
    bool hasSqrt(NDArray const& input) {
        auto matricies =input.allTensorsAlongDimension({-2, -1});
        auto result = true;
        auto n = input.sizeAt(-1);

        for (auto i = 0; i < matricies.size(); ++i) {
            result = math::nd4j_sign<T,T>(input.t<T>(0,0)) > T(0.f);
            if (result) {
                for (auto r = 1; r < n - 1; r++) {
                    if (math::nd4j_sign<T, T>(input.t<T>(r, r)) < T(0.f) &&
                        input.t<T>(r + 1, r) == T(0.f)) // if diagonal element and
                        return false;
                }
            }
            else
                return result;
        }
        return result;
    }

    template <typename T>
    int sqrtMatrixFunctor_(sd::LaunchContext* context, NDArray const* input, NDArray* output) {
        auto pInput = const_cast<NDArray*>(input);
        auto outputQ = pInput->ulike();
        auto outputT = outputQ.ulike();
        auto outputH = outputQ.ulike();
//        input->printIndexedBuffer("Input");
        //outputQ.setIdentity();
        hessenbergReduction<T>(*input, outputH, outputQ);
        MmulHelper::matmul(&outputQ, input, &outputT, true, false); //outputT.printIndexedBuffer("Res");
        MmulHelper::matmul(&outputT, &outputQ, output, false, false); output->printIndexedBuffer("Hessenberg restored");
        outputH.assign(output);
//        MmulHelper::matmul(&outputQ, &outputQ, output, false, true); output->printIndexedBuffer("Should be identity matrix");
          auto outputU = outputQ.ulike(); outputT.nullify();
          outputU.setIdentity();//outputH.setIdentity();
        computeFromHessenberg<T>(outputH, outputU, outputT, outputU);
          //primitiveSchurDecomposition<T>(context, &outputH, &outputU, &outputT);
          outputT.printIndexedBuffer("Triangular after Schur");
          outputU.printIndexedBuffer("Orthogonal after Schur");
//        schurDecomposition<T>(context, input, &outputQ, &outputT);
//        outputQ.printIndexedBuffer("Q matrix");
//        outputT.printIndexedBuffer("T matrix");
//        auto outputT = outputR.ulike();
//        hessenbergReduction<T>(*input, outputT);
//        outputT.printIndexedBuffer("H matrix");
        if (hasSqrt<T>(outputT)) {
            auto outputR = outputT.ulike();
            upperTriangularSqrt<T>(context, &outputT, &outputR);
            // restore to hessenberg reduced
            MmulHelper::matmul(&outputU, &outputR, &outputT, false, false);
            MmulHelper::matmul(&outputT, &outputU, &outputR, false, true);

            // restore to initial
            MmulHelper::matmul(&outputQ, &outputR, &outputT, false, false);
            MmulHelper::matmul(&outputT, &outputQ, output, false, true);

            return Status::OK();
        }
        return Status::CODE(ND4J_STATUS_BAD_INPUT, "helpers::sqrtMatrixFunctor::Cannot retrieve sqrt for given matrix due negative real eighenvals appears.");
    }

    int sqrtMatrixFunctor(sd::LaunchContext* context, NDArray const* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return sqrtMatrixFunctor_, (context, input, output), FLOAT_TYPES);
    }
}
}
}
