/*******************************************************************************
 *
 * Copyright (c) 2019 Konduit K.K.
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
 // @author AbdelRauf 
 //


#include<ops/declarable/helpers/addBias.h>
#include <execution/Threads.h>
#include <type_traits>

namespace nd4j {
	namespace ops {
		namespace helpers {

			//define C macros style for vector intrinsics
#define OPT_DEF_INTRINSICS 1
#define OPT_USE_INNER_COORDS_1 1
//<editor-fold desc="VECTORIZATION_INTRINSICS_STYLES">
#pragma region VECTORIZATION_INTRINSICS_STYLES

#if defined(__GNUC__) 
#define align32 __attribute__((aligned(16)))
#elif defined(_MSC_VER)
#define align32 __declspec(align(32))
#else
#define align32 
#endif 
#if defined(OPT_DEF_INTRINSICS) || defined(OPT_TEMPLATE_INTRINSICS)
#include <immintrin.h>
#endif
#if defined(OPT_DEF_INTRINSICS)

#ifdef __AVX__
#define vf8 __m256
#define vf8_loadu(ptr) _mm256_loadu_ps(ptr)
#define vf8_add(v1,v2) _mm256_add_ps(v1, v2)
#define vf8_storeu(ptr, v) _mm256_storeu_ps(ptr, v)
#else
			//this is where we let compiler figure out our mess and convert it to proper
			//codes or vecorized codes
			//typedef here just to show C style
			typedef struct vec8f {
				float val[8];
			} vf8;
			//in header its still ok with ODR rule 
			extern inline   vf8 vf8_loadu(const void* ptr) {
				//it should be ok memcpy.
				vf8 ret align32;
				memcpy(&ret, ptr, sizeof(vf8));
				return ret;
			}

			extern inline   vf8 vf8_add(vf8 v1, vf8 v2) {
				vf8 ret align32;
				for (size_t i = 0; i < sizeof(vf8) / sizeof(float); i++) {
					ret.val[i] = v1.val[i] + v2.val[i];
				}
				return ret;
			}

			extern inline   void vf8_storeu(void* ptr, vf8 v) {
				memcpy(ptr, &v, sizeof(vf8));
			}
#endif
#elif defined(OPT_TEMPLATE_INTRINSICS)
			//rely on compiler to clean our mess and generate vectorized code
			template <class T>
			class vec {
			private:
				//256bit
				T vals[32 / sizeof(T)];
			public:
				using hold_type = T;
				//other usefull constexpr like count and etc could be defined
				static constexpr size_t count() {
					return 32 / sizeof(T);
				}
				//its redundant inside class but still lets force inline
				FORCEINLINE static vec loadu(const void* ptr) {
					//it should be ok memcpy.
					vec ret align32;
					//printf("default\n");
					memcpy(ret.vals, ptr, sizeof(vals));
					return ret;
				}

				vec operator+ (vec const& v) {
					vec ret align32;
					for (size_t i = 0; i < sizeof(vals) / sizeof(T); i++) {
						ret.vals[i] = vals[i] + v.vals[i];
					}
					return ret;
				}
				void storeu(void* ptr) const {
					memcpy(ptr, vals, sizeof(vals));
				}

			};

#ifdef __AVX__
			template <>
			class vec<float> {
			private:
				__m256 vals;
			public:
				using hold_type = float;

				static constexpr size_t count() {
					return 8;
				}

				FORCEINLINE static vec loadu(const void* ptr) {
					vec ret;
					//printf("avx\n");
					ret.vals = _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));
					return ret;
				}

				FORCEINLINE vec operator+ (vec const& v) {
					vec ret;
					ret.vals = _mm256_add_ps(vals, v.vals);
					return ret;
				}
				FORCEINLINE void storeu(void* ptr) const {
					_mm256_storeu_ps(reinterpret_cast<float*>(ptr), vals);
				}

			};
#endif

#else 
			//nothing lets trust our compiler. its smart enough 
#endif
#pragma endregion
//</editor-fold>


//inner_add function that can be autovectorized
			template <typename T>
			FORCEINLINE  void inner_add(const T* __restrict xx, const T* __restrict yy, T* __restrict zz, size_t N) {
				//our compiler is smart to handle it
				for (uint c = 0; c < N; c++)
					zz[c] = xx[c] + yy[c];

			}

//<editor-fold desc="INNER_ADD_WITH_MANUAL_VECINT_STYLES">
#pragma region INNER_ADD_WITH_MANUAL_VECINT_STYLES

#if defined(OPT_DEF_INTRINSICS) 
			template <>
			FORCEINLINE  void inner_add<float>(const float* __restrict xx, const float* __restrict yy, float* __restrict zz, size_t N) {
				//manual instrinsics 
				size_t nd = N & (-32);
				if (nd >= 32) {
					size_t i = 0;
					vf8 vy0 = vf8_loadu(yy);
					vf8 vy1 = vf8_loadu(yy + 8);
					vf8 vy2 = vf8_loadu(yy + 16);
					vf8 vy3 = vf8_loadu(yy + 24);
					vf8 vx0 = vf8_loadu(xx);
					vf8 vx1 = vf8_loadu(xx + 8);
					vf8 vx2 = vf8_loadu(xx + 16);
					vf8 vx3 = vf8_loadu(xx + 24);
					vf8 vz0 = vf8_add(vx0, vy0);
					vf8 vz1 = vf8_add(vx1, vy1);
					vf8 vz2 = vf8_add(vx2, vy2);
					vf8 vz3 = vf8_add(vx3, vy3);
					for (i = 0; i < nd - 32; i += 32) {

						vy0 = vf8_loadu(yy + i + 32);
						vy1 = vf8_loadu(yy + i + 8 + 32);
						vy2 = vf8_loadu(yy + i + 16 + 32);
						vy3 = vf8_loadu(yy + i + 24 + 32);

						vx0 = vf8_loadu(xx + i + 32);
						vx1 = vf8_loadu(xx + i + 8 + 32);
						vx2 = vf8_loadu(xx + i + 16 + 32);
						vx3 = vf8_loadu(xx + i + 24 + 32);

						vf8_storeu(zz + i, vz0);
						vf8_storeu(zz + i + 8, vz1);
						vf8_storeu(zz + i + 16, vz2);
						vf8_storeu(zz + i + 24, vz3);


						vz0 = vf8_add(vx0, vy0);
						vz1 = vf8_add(vx1, vy1);
						vz2 = vf8_add(vx2, vy2);
						vz3 = vf8_add(vx3, vy3);

					}

					vf8_storeu(zz + i, vz0);
					vf8_storeu(zz + i + 8, vz1);
					vf8_storeu(zz + i + 16, vz2);
					vf8_storeu(zz + i + 24, vz3);

				}
				for (size_t i = nd; i < N; i++)
					zz[i] = xx[i] + yy[i];

			}
#elif defined(OPT_TEMPLATE_INTRINSICS)

			template <>
			FORCEINLINE  void inner_add<float>(const float* __restrict xx, const float* __restrict yy, float* __restrict zz, size_t N) {
				//manual instrinsics 
				using vecf = vec<float>;
				size_t nd = N & (-32);
				if (nd >= 32) {
					size_t i = 0;
					auto vy0 = vecf::loadu(yy);
					auto vy1 = vecf::loadu(yy + 8);
					auto vy2 = vecf::loadu(yy + 16);
					auto vy3 = vecf::loadu(yy + 24);
					auto vx0 = vecf::loadu(xx);
					auto vx1 = vecf::loadu(xx + 8);
					auto vx2 = vecf::loadu(xx + 16);
					auto vx3 = vecf::loadu(xx + 24);
					auto vz0 = vx0 + vy0;
					auto vz1 = vx1 + vy1;
					auto vz2 = vx2 + vy2;
					auto vz3 = vx3 + vy3;
					for (i = 0; i < nd - 32; i += 32) {

						vy0 = vecf::loadu(yy + i + 32);
						vy1 = vecf::loadu(yy + i + 8 + 32);
						vy2 = vecf::loadu(yy + i + 16 + 32);
						vy3 = vecf::loadu(yy + i + 24 + 32);

						vx0 = vecf::loadu(xx + i + 32);
						vx1 = vecf::loadu(xx + i + 8 + 32);
						vx2 = vecf::loadu(xx + i + 16 + 32);
						vx3 = vecf::loadu(xx + i + 24 + 32);

						vz0.storeu(zz + i);
						vz1.storeu(zz + i + 8);
						vz2.storeu(zz + i + 16);
						vz3.storeu(zz + i + 24);


						vz0 = vx0 + vy0;
						vz1 = vx1 + vy1;
						vz2 = vx2 + vy2;
						vz3 = vx3 + vy3;

					}

					vz0.storeu(zz + i);
					vz1.storeu(zz + i + 8);
					vz2.storeu(zz + i + 16);
					vz3.storeu(zz + i + 24);

				}
				for (size_t i = nd; i < N; i++)
					zz[i] = xx[i] + yy[i];

			}

#endif
#pragma endregion
//</editor-fold>



			template<typename T>
			void inner_impl_for_lastC_ews1(const NDArray& x, const NDArray& y, NDArray& out, bool check_strides = false) {

				Nd4jLong coords[MAX_RANK] = {};
				Nd4jLong add_coords[MAX_RANK];
				Nd4jLong* shapeInfo = x.getShapeInfo();
				Nd4jLong rank = shapeInfo[0];

				if (rank < 1) return;

				T* xx_in = x.bufferAsT<T>();
				T* yy_in = y.bufferAsT<T>();
				T* out_in = out.bufferAsT<T>();

				Nd4jLong* y_shapeInfo = y.getShapeInfo();
				Nd4jLong* strides = x.stridesOf();
				Nd4jLong* bases = &(shapeInfo[1]);
				Nd4jLong* y_strides = y.stridesOf();


				size_t t = 1;

				for (size_t i = 0; i < rank; i++) {
					t *= bases[i];
				}

				size_t C = bases[rank - 1];
				assert(C == y_shapeInfo[1] && y_strides[0] == 1);
				bool treat_as_strided = !check_strides;
				if (check_strides) {
					Nd4jLong calc_stride = bases[rank - 1];
					for (int i = rank - 2; i >= 0; i--) {
						bool is_neq = calc_stride != strides[i];
						treat_as_strided &= is_neq;
						if (treat_as_strided) break;
						calc_stride = bases[i] * calc_stride;
					}
				}
				//coords for index 0 is  all 0 zeroes
				//we can manually set 1 for C case here; its  rank-2 should be 1  
				//shape::index2coords(0, shapeInfo, coords);
				shape::index2coords(C, shapeInfo, add_coords);

				size_t offset = 0;
				if (treat_as_strided) {

					//check for rank==4 and rank==5 
					if (rank == 4) {
						for (size_t i = 0; i < t; i += C) {
							inner_add<T>(xx_in + offset, yy_in, out_in + offset, C);
							offset = move_by_coords<3>(bases, strides, coords, add_coords);
						}
					}
					else if (rank == 5) {
						for (size_t i = 0; i < t; i += C) {
							inner_add<T>(xx_in + offset, yy_in, out_in + offset, C);
							offset = move_by_coords<4>(bases, strides, coords, add_coords);
						}
					}
					else if (rank == 1) {
						inner_add<T>(xx_in, yy_in, out_in, C);
					}
					else if (rank == 2) {
						for (size_t i = 0; i < t; i += C) {
							inner_add<T>(xx_in + offset, yy_in, out_in + offset, C);
							offset = move_by_coords<1>(bases, strides, coords, add_coords);
						}
					}
					else if (rank == 3) {
						for (size_t i = 0; i < t; i += C) {
							inner_add<T>(xx_in + offset, yy_in, out_in + offset, C);
							offset = move_by_coords<2>(bases, strides, coords, add_coords);
						}
					}
					else if (rank > 1) {
						for (size_t i = 0; i < t; i += C) {
							inner_add<T>(xx_in + offset, yy_in, out_in + offset, C);
							offset = move_by_coords(bases, strides, coords, add_coords, rank - 1);

						}
					}
				}
				else {
					//std::cout << "treated" << std::endl;

					constexpr size_t MIN_NN = 32;
					T  yyy_buffer[MIN_NN * MIN_NN];
					T* yyy_in;
					size_t OLD_C = C;
					if (C >= MIN_NN) {
						yyy_in = yy_in;
					}
					else {

						size_t NEW_C = C * MIN_NN;
						for (size_t i = 0; i < NEW_C; i += C) {
							//copy to our buffer
							T* pyyy = &(yyy_buffer[i]);
							for (size_t j = 0; j < C; j++) {
								pyyy[j] = yy_in[j];
							}
						}
						//we can increase C here as it is the whole buffer
						//now we got vectorizable buffer
						C = NEW_C;
						yyy_in = yyy_buffer;
					}
					size_t i = 0;
					size_t t_1 = C == OLD_C ? t - 1 : t - C;
					for (; i <= t_1; i += C) {

						inner_add<T>(xx_in, yyy_in, out_in, C);
						xx_in += C;
						out_in += C;
					}

					int tail = t - i;
					if (tail > 0)
						inner_add<T>(xx_in, yyy_in, out_in, tail);
					//std::cout << "__t1 " << t_1 << "__i " << i << " old_C " << OLD_C << " C" << C <<" tail "<<tail<<" t"<<t<< std::endl;


				}


			}
			 
			//////////////////////////////////////////////////////////////////////////
			template <typename X, typename Y>
			static void addBiasE_(const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW,const bool check_strides) {
				const X* x = input.bufferAsT<X>();
				const Y* y = bias.bufferAsT<Y>();
				X* z = output.bufferAsT<X>();
				const bool inOutAreSame = x == z;
				int posOfNonUnityDim;
				bias.isCommonVector(posOfNonUnityDim);
				const Nd4jLong rank =  output.rankOf() ;
				const uint bS = output.sizeAt(0);              // batch size
				
				assert(inOutAreSame == false);
				const uint C = isNCHW ? output.sizeAt(1) : output.sizeAt(3);     // channels
				const uint oH = isNCHW ? output.sizeAt(2) : output.sizeAt(1);     // height
				const uint oW = isNCHW ? output.sizeAt(3) : output.sizeAt(2);     // width 
				const Nd4jLong zStrideC = isNCHW ? output.stridesOf()[1] : output.stridesOf()[3];
				const Nd4jLong zStrideH = isNCHW ? output.stridesOf()[2] : output.stridesOf()[1];
				const Nd4jLong zStrideW = isNCHW ? output.stridesOf()[3] : output.stridesOf()[2];
				const Nd4jLong yStrideC = bias.strideAt(posOfNonUnityDim);
				const Nd4jLong zStrideB = output.strideAt(0);
				const Nd4jLong xStrideB = input.stridesOf()[0]; 
				const Nd4jLong xStrideC = isNCHW ? input.stridesOf()[1] : input.stridesOf()[3];
				const Nd4jLong xStrideH = isNCHW ? input.stridesOf()[2] : input.stridesOf()[1];
				const Nd4jLong xStrideW = isNCHW ? input.stridesOf()[3] : input.stridesOf()[2];
				assert(isNCHW == false);
#ifdef OPT_USE_INNER_COORDS_1
				 if ( zStrideC == 1 && yStrideC == 1 && xStrideC == 1 && 'c' == input.ordering() && std::is_same<X, Y>::value) {
					inner_impl_for_lastC_ews1<X>(input, bias, output, check_strides);
				 }
				 else {
					 //not implemented
					 assert(false);
				 }

#else
				//it is only for rank==4
				assert(output.rankOf() == 4)


				const X* x = input.bufferAsT<X>();
				const Y* y = bias.bufferAsT<Y>();
				X* z = output.bufferAsT<X>();
				const X* yy = (X*)y;
				auto func = PRAGMA_THREADS_FOR_3D{
					for (uint b = start_x; b < stop_x; b++)
						for (uint h = start_y; h < stop_y; h++)
							for (uint w = start_z; w < stop_z; w++) {
								 const X* xx = x + b * xStrideB + h * xStrideH + w * xStrideW;
								 X* zz = z + b * zStrideB + h * zStrideH + w * zStrideW;
								inner_add<X>(xx, yy, zz, C);
					}
				};

				samediff::Threads::parallel_for(func, 0, bS, 1, 0, oH, 1, 0, oW, 1);
#endif
	 
		 
			}

			//////////////////////////////////////////////////////////////////////////
			void addBias_Experimental(graph::Context& block, const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW, const bool check_strides)
			{

				// bias.rankOf() == 1 ? bias : bias.reshape(bias.ordering(), {bias.lengthOf()})
				BUILD_DOUBLE_SELECTOR(input.dataType(), bias.dataType(), addBiasE_, (input, bias, output, isNCHW, check_strides), FLOAT_TYPES, FLOAT_TYPES);
			}


			BUILD_DOUBLE_TEMPLATE(template void addBiasE_, (const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW,const bool check_strides), FLOAT_TYPES, FLOAT_TYPES);

		}
	}
}

