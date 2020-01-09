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
#include <cmath>
#include <execution/ThreadPool.h>
#include <stdexcept>
#include <memory>
namespace nd4j {
	namespace ops {
		namespace helpers {

			//define C macros style for vector intrinsics
#define OPT_TEMPLATE_INTRINSICS 1
#define OPT_USE_INNER_COORDS_1 1
//#define  PRINT_VERBOSE 1



//<editor-fold desc="INNER_ADD_WITH_MANUAL_VECINT_STYLES">
#pragma region INNER_ADD_WITH_MANUAL_VECINT_STYLES

#if defined(__GNUC__) 
#define align32 __attribute__((aligned(32)))
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

			template <typename T>
			FORCEINLINE  void inner_add_inplace( T* __restrict xx, const T* __restrict yy, size_t N) {
				//our compiler is smart to handle it
				for (uint c = 0; c < N; c++)
					xx[c] = xx[c] + yy[c];

			}


			template <typename T>
			FORCEINLINE  void inner_add_inplace_ordinary(T* __restrict xx, const T* __restrict yy, size_t N) {
				 
				for (uint c = 0; c < N; c++)
					xx[c] = xx[c] + yy[c];

			}

			template <typename T>
			FORCEINLINE  void inner_add_ordinary(const T* __restrict xx, const T* __restrict yy, T* __restrict zz, size_t N) {
				//ordinary one will be used in case we dont know 
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

				//prefetchl(&xx[0]);
				//prefetchwl(&zz[0]);
				if (nd >= 32) {
					//prefetchl(&xx[32]); 
					size_t i = 0;
					auto vx0 = vecf::loadu(xx);
					auto vx1 = vecf::loadu(xx + 8);
					auto vx2 = vecf::loadu(xx + 16);
					auto vx3 = vecf::loadu(xx + 24);

					auto vy0 = vecf::loadu(yy);
					auto vy1 = vecf::loadu(yy + 8);
					auto vy2 = vecf::loadu(yy + 16);
					auto vy3 = vecf::loadu(yy + 24);

					auto vz0 = vx0 + vy0;
					auto vz1 = vx1 + vy1;
					auto vz2 = vx2 + vy2;
					auto vz3 = vx3 + vy3;
					for (i = 0; i < nd - 32; i += 32) {
						//prefetchl(&xx[i + 32+32]);
						//prefetchwl(&zz[i + 32]);
						vx0 = vecf::loadu(xx + i + 32);
						vx1 = vecf::loadu(xx + i + 8 + 32);
						vx2 = vecf::loadu(xx + i + 16 + 32);
						vx3 = vecf::loadu(xx + i + 24 + 32);


						vy0 = vecf::loadu(yy + i + 32);
						vy1 = vecf::loadu(yy + i + 8 + 32);
						vy2 = vecf::loadu(yy + i + 16 + 32);
						vy3 = vecf::loadu(yy + i + 24 + 32);



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

#if defined(PRINT_VERBOSE)
 #define doutput(fmt,  ...) nd4j_printf(fmt, __VA_ARGS__) 
#else
#define doutput(fmt,  ...)  
#endif
			int  parallel_for2(FUNC_1D function, int64_t start, int64_t stop, int64_t increment,size_t type_size=sizeof(float), uint32_t req_numThreads = nd4j::Environment::getInstance()->maxMasterThreads()) {
				if (start > stop)
					throw std::runtime_error("Threads::parallel_for got start > stop");


				auto num_elements = (stop - start);
				//this way we preserve increment starts offset
				//so we will adjust considering delta not num_elements
				auto delta = (stop - start) / increment;
				doutput("increment %d \n", increment);
				// in some cases we just fire func as is
				if (delta == 0 || req_numThreads == 1) {
					function(0, start, stop, increment);
					return 1;
				}


				int numThreads = 0;
#if  1
				int adjusted_numThreads = 2;
#else
				int adjusted_numThreads = samediff::ThreadsHelper::numberOfThreads(req_numThreads, (num_elements * sizeof(double))/(200*type_size));
#endif
				if (adjusted_numThreads > delta)
					adjusted_numThreads = delta;

				// shortcut
				if (adjusted_numThreads <= 1) {
					function(0, start, stop, increment);
					return 1;
				}

				//take span as ceil  
				auto spand = std::ceil((double)delta / (double)adjusted_numThreads);
				 
				numThreads = static_cast<int>(std::ceil((double)delta / spand));
				auto span = static_cast<Nd4jLong>(spand);
				doutput("span  %d \n", span);
				auto ticket = samediff::ThreadPool::getInstance()->tryAcquire(numThreads);
				if (ticket != nullptr) {

					//tail_add is additional value of the last part
					//it could be negative or positive
					//we will spread that value across
					auto tail_add = delta - numThreads * span;
					Nd4jLong begin = 0;
					Nd4jLong end = 0;
					doutput("tail_add %d \n", tail_add);
					//we will try enqueu bigger parts first
					decltype(span) span1, span2;
					int last = 0;
					if (tail_add >= 0) {
						//for span == 1  , tail_add is  0 
						last = tail_add;
						span1 = span + 1;
						span2 = span;
					}
					else {
						last = numThreads + tail_add;// -std::abs(tail_add);
						span1 = span;
						span2 = span - 1;
					}
					for (int i = 0; i < last; i++) {
						end = begin + span1 * increment;
						doutput("enque_1 start %d stop %d count:%d \n",begin,end, span1 * increment);
						// putting the task into the queue for a given thread
						ticket->enqueue(i, numThreads, function, begin, end, increment);
						begin = end;
					}
					for (int i = last; i < numThreads - 1; i++) {
						end = begin + span2 * increment;
						doutput("enque_2 start %d stop %d count:%d \n", begin, end, span2 * increment);
						// putting the task into the queue for a given thread
						ticket->enqueue(i, numThreads, function, begin, end, increment);
						begin = end;
					}
					//for last one enqueue last offset as stop
					//we need it in case our ((stop-start) % increment ) > 0
					ticket->enqueue(numThreads - 1, numThreads, function, begin, stop, increment);
					doutput("enque tail: %d \n", stop-begin);
					// block and wait till all threads finished the job
					ticket->waitAndRelease();

					// we tell that parallelism request succeeded
					return numThreads;
				}
				else {
					// if there were no threads available - we'll execute function right within current thread
					function(0, start, stop, increment);

					// we tell that parallelism request declined
					return 1;
				}
			}


			static constexpr size_t MIN_NN = 32;
			static constexpr size_t MIN_NN_K = 4;
			template<typename T>
			void inner_impl_generic_lastC_vect(Nd4jLong* x_shapeInfo, Nd4jLong* z_shapeInfo, T* x, const T* b, T* z, bool inplaceOp, bool sameStride, Nd4jLong num, Nd4jLong C) {
				doutput("inner_impl_generic_lastC_vect %d\n", 0);
				const Nd4jLong rank = x_shapeInfo[0];
				const Nd4jLong* bases = &(x_shapeInfo[1]);
				const Nd4jLong* x_strides = &(x_shapeInfo[rank + 1]);
				const Nd4jLong* z_strides = &(z_shapeInfo[rank + 1]);
				if (sameStride) {

					if (inplaceOp) {

						auto func = PRAGMA_THREADS_FOR{

							Nd4jLong coords_p[MAX_RANK] ;
							shape::index2coords(start, x_shapeInfo, coords_p);
							size_t offset_p = offset_from_coords(x_strides, coords_p, rank);
							for (size_t i = start; i < stop; i += increment) {
									inner_add_inplace<T>(&(x[offset_p]), b,  increment);
									offset_p = inc_by_coords(bases, x_strides, coords_p, offset_p,rank - 1);
							}
						};

						parallel_for2(func, 0, num, C);

					}
					else {
						auto func = PRAGMA_THREADS_FOR{

							Nd4jLong coords_p[MAX_RANK] ;
							shape::index2coords(start, x_shapeInfo, coords_p);
							size_t offset_p = offset_from_coords(x_strides, coords_p, rank);
							for (size_t i = start; i < stop; i += increment) {
								inner_add<T>(&(x[offset_p]), b, &(z[offset_p]), increment);
								offset_p = inc_by_coords(bases, x_strides, coords_p, offset_p,rank - 1);
							}
						};

						parallel_for2(func, 0, num, C);

					}

				}
				else {

					auto func = PRAGMA_THREADS_FOR{

						Nd4jLong coords_p[MAX_RANK] ;
						shape::index2coords(start, x_shapeInfo, coords_p);
						pair_size_t offset;
						offset.first = offset_from_coords(x_strides, coords_p, rank);
						offset.second = offset_from_coords(z_strides, coords_p, rank);
						for (size_t i = start; i < stop; i += increment) {
								inner_add<T>(&(x[ offset.first]), b, &(z[ offset.second]), increment);
								offset = inc_by_coords_zip(bases, x_strides, z_strides, coords_p, offset, rank - 1);
						}
					};

					parallel_for2(func, 0, num, C);


				}

			}

			
			template<typename T, size_t constRank>
			void inner_impl_generic_lastC_vect(Nd4jLong* x_shapeInfo, Nd4jLong* z_shapeInfo, T* x, const T* b, T* z, bool inplaceOp, bool sameStride, Nd4jLong num, Nd4jLong C) {
				doutput("inner_impl_generic_lastC_vect %d\n", 1);
				const Nd4jLong rank = x_shapeInfo[0];
				const Nd4jLong* bases = &(x_shapeInfo[1]);
				const Nd4jLong* x_strides = &(x_shapeInfo[rank + 1]);
				const Nd4jLong* z_strides = &(z_shapeInfo[rank + 1]);
				if (sameStride) {

					if (inplaceOp) {

						auto func = PRAGMA_THREADS_FOR{

							Nd4jLong coords_p[constRank];
							shape::index2coords(start, x_shapeInfo, coords_p);
							size_t offset_p = offset_from_coords(x_strides, coords_p, rank);
							for (size_t i = start; i < stop; i += increment) {
									inner_add_inplace<T>(x + offset_p, b, increment);
									offset_p = inc_by_coords<constRank - 1>(bases, x_strides, coords_p, offset_p);
								}
						};

						parallel_for2(func, 0, num, C);
					}
					else {

						auto func = PRAGMA_THREADS_FOR{

							Nd4jLong coords_p[constRank] ;
							shape::index2coords(start, x_shapeInfo, coords_p);
							size_t offset_p = offset_from_coords(x_strides, coords_p, rank);
							for (size_t i = start; i < stop; i += increment) {
									inner_add<T>(x + offset_p, b, z + offset_p, increment);
									offset_p = inc_by_coords<constRank - 1>(bases, x_strides, coords_p, offset_p);
							}
						};

						parallel_for2(func, 0, num, C);
					}

				}
				else {

					auto func = PRAGMA_THREADS_FOR{

						Nd4jLong coords_p[constRank] ;
						shape::index2coords(start, x_shapeInfo, coords_p);
						pair_size_t offset;
						offset.first = offset_from_coords(x_strides, coords_p, rank);
						offset.second = offset_from_coords(z_strides, coords_p, rank);

						for (size_t i = start; i < stop; i += increment) {
								inner_add<T>(&(x [offset.first]), b, &(z[offset.second]), increment);
								offset = inc_by_coords_zip<constRank - 1>(bases, x_strides, z_strides, coords_p, offset );
							}
					};

					parallel_for2(func, 0, num, C);

				}

			}


			template<typename T>
			void inner_impl_generic_lastC_ordinary(Nd4jLong* x_shapeInfo, Nd4jLong* z_shapeInfo, T* x, const T* b, T* z, bool inplaceOp, bool sameStride, Nd4jLong num, Nd4jLong C) {
				doutput("inner_impl_generic_lastC_ordinary %d\n", 0);
				const Nd4jLong rank = x_shapeInfo[0];
				const Nd4jLong* bases = &(x_shapeInfo[1]);
				const Nd4jLong* x_strides = &(x_shapeInfo[rank + 1]);
				const Nd4jLong* z_strides = &(z_shapeInfo[rank + 1]);
				if (sameStride) {

					if (inplaceOp) {
						auto func = PRAGMA_THREADS_FOR{

							Nd4jLong coords_p[MAX_RANK] ;
							shape::index2coords(start, x_shapeInfo, coords_p);
							size_t offset_p = offset_from_coords(x_strides, coords_p, rank);
							for (size_t i = start; i < stop; i += increment) {
								inner_add_inplace_ordinary(&(x[offset_p]), b, increment);
								offset_p = inc_by_coords(bases, x_strides, coords_p, offset_p,rank - 1);
							}
						};

						parallel_for2(func, 0, num, C);

					}
					else {
						auto func = PRAGMA_THREADS_FOR{

							Nd4jLong coords_p[MAX_RANK] ;
							shape::index2coords(start, x_shapeInfo, coords_p);
							size_t offset_p = offset_from_coords(x_strides, coords_p, rank);
							for (size_t i = start; i < stop; i += increment) {
								inner_add_ordinary(&(x[offset_p]), b, &(z[offset_p]), increment);
								offset_p = inc_by_coords(bases, x_strides, coords_p, offset_p,rank - 1);
							}
						};

						parallel_for2(func, 0, num, C);

					}

				}
				else {

					auto func = PRAGMA_THREADS_FOR{

						Nd4jLong coords_p[MAX_RANK] ;
						shape::index2coords(start, x_shapeInfo, coords_p);
						pair_size_t offset;
						offset.first = offset_from_coords(x_strides, coords_p, rank);
						offset.second = offset_from_coords(z_strides, coords_p, rank);
						for (size_t i = start; i < stop; i += increment) {
							inner_add_ordinary(&(x[offset.first]), b, &(z[offset.second]), increment);
							offset = inc_by_coords_zip(bases, x_strides, z_strides, coords_p, offset, rank - 1);
						}
					};

					parallel_for2(func, 0, num, C);

				}
			}

			template<typename T, size_t constRank>
			void inner_impl_generic_lastC_ordinary(Nd4jLong* x_shapeInfo, Nd4jLong* z_shapeInfo, T* x, const T* b, T* z, bool inplaceOp, bool sameStride, Nd4jLong num, Nd4jLong C) {
				doutput("inner_impl_generic_lastC_ordinary %d\n", 1);
				const  Nd4jLong rank = x_shapeInfo[0];
				const Nd4jLong* bases = &(x_shapeInfo[1]);
				const Nd4jLong* x_strides = &(x_shapeInfo[rank + 1]);
				const Nd4jLong* z_strides = &(z_shapeInfo[rank + 1]);
				if (sameStride) {

					if (inplaceOp) {
						auto func = PRAGMA_THREADS_FOR{

							Nd4jLong coords_p[constRank] ;
							shape::index2coords(start, x_shapeInfo, coords_p);
							size_t offset_p = offset_from_coords(x_strides, coords_p, rank);
							for (size_t i = start; i < stop; i += increment) { 
									inner_add_inplace_ordinary(&(x[offset_p]), b, increment);
									offset_p = inc_by_coords<constRank - 1>(bases, x_strides, coords_p, offset_p);
								}
						};

						parallel_for2(func, 0, num, C);

					}
					else {

						auto func = PRAGMA_THREADS_FOR{

							Nd4jLong coords_p[constRank] ;
							shape::index2coords(start, x_shapeInfo, coords_p);
							size_t offset_p = offset_from_coords(x_strides, coords_p, rank);
							for (size_t i = start; i < stop; i += increment) { 
								inner_add_ordinary(&(x[offset_p]), b, &(z[offset_p]), increment);
								offset_p = inc_by_coords<constRank - 1>(bases, x_strides, coords_p, offset_p);
							}
						};

						parallel_for2(func, 0, num, C);
					}

				}
				else {

					auto func = PRAGMA_THREADS_FOR{

						Nd4jLong coords_p[constRank] ;
						shape::index2coords(start, x_shapeInfo, coords_p);
						pair_size_t offset;
						offset.first = offset_from_coords(x_strides, coords_p, rank);
						offset.second = offset_from_coords(z_strides, coords_p, rank);

						for (size_t i = start; i < stop; i += increment) { 
								inner_add_ordinary(&(x[offset.first]), b, &(z[offset.second]), increment);
								offset = inc_by_coords_zip<constRank - 1>(bases, x_strides, z_strides, coords_p, offset);
							}
					};

					parallel_for2(func, 0, num, C);

				}
			}

			template<typename T>
			void inner_impl_generic_orderC(Nd4jLong* x_shapeInfo, Nd4jLong* z_shapeInfo, T* x, const T* b, T* z, bool inplaceOp, Nd4jLong num) {
				doutput("inner_impl_generic_orderC %d\n", 0);
				const Nd4jLong rank = x_shapeInfo[0];
				const Nd4jLong* bases = &(x_shapeInfo[1]);
				const Nd4jLong* x_strides = &(x_shapeInfo[rank + 1]);
				const Nd4jLong* z_strides = &(z_shapeInfo[rank + 1]);
				//different order
// for          it is better use output order as it has load store 
				Nd4jLong C = bases[rank - 1];
				Nd4jLong x_stride = x_strides[rank - 1];
				Nd4jLong z_stride = z_strides[rank - 1];

				auto func = PRAGMA_THREADS_FOR{

					Nd4jLong coords_p[MAX_RANK] ;
					shape::index2coords(start, z_shapeInfo, coords_p);
					pair_size_t offset;
					//we will skip 1 index so that we are able to iterate
					//this is also achieviable decreasing rank for c order, as it starts from the last
					offset.first = offset_from_coords(x_strides, coords_p, rank);
					offset.second = offset_from_coords(z_strides, coords_p, rank);
					for (size_t i = start; i < stop; i += increment) {
							T* xx = &(x[offset.first]);
							T* zz = &(z[offset.second]);
							for (size_t c = 0; c < increment; c++)
								zz[c * z_stride] = xx[c * x_stride] + b[c];
							offset = inc_by_coords_zip(bases, x_strides, z_strides, coords_p, offset, rank, 1);
							//doutput("%d %d \n", offset.first, offset.second);
					}
				};

				parallel_for2(func, 0, num, C);

			}

			template<typename T, typename T2>
			void inner_impl_generic_orderF(Nd4jLong* x_shapeInfo, Nd4jLong* z_shapeInfo, T* x, const T2* b, T* z, bool inplaceOp, bool needCasting, Nd4jLong num, Nd4jLong yStrideC) {
				doutput("inner_impl_generic_zorderF %d\n", 0);
				const Nd4jLong rank = x_shapeInfo[0];
				const Nd4jLong* bases = &(x_shapeInfo[1]);
				const Nd4jLong* x_strides = &(x_shapeInfo[rank + 1]);
				const Nd4jLong* z_strides = &(z_shapeInfo[rank + 1]);
				Nd4jLong C = bases[0];
				Nd4jLong x_stride = x_strides[0];
				Nd4jLong z_stride = z_strides[0]; 
				auto func = PRAGMA_THREADS_FOR{

					Nd4jLong coords_p[MAX_RANK] ;
				    index2coords_F(start, rank,bases, coords_p);
					pair_size_t offset;
					offset.first = offset_from_coords(x_strides, coords_p, rank);
					offset.second = offset_from_coords(z_strides, coords_p, rank);
					//we want to skip first index to iterate over it 
					for (size_t i = start; i < stop; i += increment) {
						    T* xx = &(x[offset.first]);
					     	T* zz = &(z[offset.second]);
							T yy = static_cast<T>(b[coords_p[rank - 1] * yStrideC]);
							for (size_t c = 0; c < increment; c++)
								zz[c * z_stride] = xx[c * x_stride] + yy;
							offset = inc_by_coords_zip<false>(bases, x_strides, z_strides, coords_p, offset, rank,1);
							//doutput("%d %d \n", offset.first, offset.second);
					}
				};
				parallel_for2(func, 0, num, C);
			}


			/**
			* this is our main optimization which  benefits from everything for numC continous case
			*/
			template<typename T>
			void inner_impl_continous_lastC( T* x, const T* b, T* z, T* b_buffer, bool inplaceOp, Nd4jLong totalNum, Nd4jLong numC) {
				doutput("inner_impl_continous_lastC %d",0);
				const T* b_in;
				size_t OLD_C = numC;

				if (numC > MIN_NN_K*MIN_NN || totalNum < numC * MIN_NN_K){
					//either totalNum was small or sufficient numC
					b_in = b;
				}
				else {
					//our buffer will have MIN_NN * MIN_NN

					size_t NEW_C = numC< MIN_NN ? numC * MIN_NN : numC * MIN_NN/ MIN_NN_K;
					//if we can still multiply lets do
					NEW_C =(NEW_C* MIN_NN_K <= totalNum &&  NEW_C < MIN_NN * MIN_NN / MIN_NN_K) ? MIN_NN_K * NEW_C : NEW_C;

					for (size_t i = 0; i < NEW_C; i += numC) {
						//copy to our buffer
						T* cp = &(b_buffer[i]);
						for (size_t j = 0; j < numC; j++) {
							cp[j] = b[j];
						}
					}
					//we can increase C here as it is the whole buffer
					//now we got vectorizable buffer
					numC = NEW_C;
					b_in = b_buffer;
				}

				if (inplaceOp) {
					//calculate bias_add
					auto func = PRAGMA_THREADS_FOR{
						size_t nums = (stop - start);
						size_t num_inc = nums - nums % increment; 
						size_t offset_p = start;
						for (size_t i = 0; i < num_inc; i += increment) {
							inner_add_inplace<T>(&(x[offset_p]), b_in,  increment);
							offset_p += increment;
						}

						if (nums > num_inc)
							inner_add_inplace<T>(&(x[offset_p]), b_in,  nums - num_inc);
					};

					parallel_for2(func, 0, totalNum, numC);

				}
				else {

					//calculate bias_add
					auto func = PRAGMA_THREADS_FOR{

						size_t nums = (stop - start);
						size_t num_inc = nums - nums % increment;
 
						prefetch_range_rl((char*)b,  1024<increment?1024: increment);
						doutput("start %d stop %d inc %d ::count %d \n",start,stop,increment, nums);
						size_t offset_p = start;
						for (size_t i = 0; i < num_inc; i += increment) { 
							inner_add<T>(&(x[offset_p]), b_in, &(z[offset_p]), increment);
							offset_p += increment;
						}

						if (nums > num_inc)
							inner_add<T>(&(x[offset_p]), b_in, &(z[offset_p]), nums - num_inc);
					};

					parallel_for2(func, 0, totalNum, numC);
				}


			}


			template<typename X, typename Y>
			static
				typename std::enable_if<std::is_same<X, Y>::value, const X*>::type
				flattened_bias(const Y* b_real, X* b_stack, const size_t b_stack_size, std::unique_ptr<X[]>& b_heap, const Nd4jLong num, Nd4jLong yStrideC)
			{
				//for order =='c' we will use last_index_faster methods
				//so our Y will be accessed too often
				// we will buffer it beforehand, 
				//best results when buffer used much , may result bad perf if buffer is used once
				X* b_new = nullptr;
				if (yStrideC != 1) {
					if (num > b_stack_size) {
						b_heap.reset(new X[num]);
						b_new = b_heap.get();
					}
					else {
						b_new = b_stack;
					}
					for (size_t i = 0; i < num; i++) {
						b_new[i] = b_real[i * yStrideC];
					}
				}
				else {
					//no need , just pass normal bias
					return static_cast<const X*>(b_real);
				}

				return const_cast<const X*>(b_new);

			}

			template<typename X, typename Y>
			static
				typename std::enable_if<!std::is_same<X, Y>::value, const X*>::type
				flattened_bias(const Y* b_real, X* b_stack, const size_t b_stack_size, std::unique_ptr<X[]>& b_heap, const Nd4jLong num, Nd4jLong yStrideC)
			{
				//for order =='c' we will use last_index_faster methods
				//so our Y will be accessed too often
				// we will buffer it beforehand, 
				//best results when buffer used much , may result bad perf if buffer is used once
				X* b_new = nullptr;

				if (num > b_stack_size) {
					b_heap.reset(new X[num]);
					b_new = b_heap.get();
				}
				else {
					b_new = b_stack;
				}

				if (yStrideC != 1) {
					for (size_t i = 0; i < num; i++) {
						b_new[i] = static_cast<X>(b_real[i * yStrideC]);
					}
				}
				else {
					for (size_t i = 0; i < num; i++) {
						b_new[i] = static_cast<X>(b_real[i]);
					}
				}

				return const_cast<const X*>(b_new);

			}


			template <typename X, typename Y>
			static void addBiasE_(const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW, const bool check_strides) {
				X* x = input.bufferAsT<X>();
				const Y* b = bias.bufferAsT<Y>();
				X* z = output.bufferAsT<X>();
				Nd4jLong* x_shapeInfo = input.getShapeInfo();
				Nd4jLong* z_shapeInfo = output.getShapeInfo();
				const bool inplaceOp = (x == z);
				constexpr bool needCasting = !std::is_same<X, Y>::value;
				const bool sameOrder = inplaceOp || (input.ordering() == output.ordering());
				bool sameStride = inplaceOp || shape::strideEquals(x_shapeInfo, z_shapeInfo);
				const Nd4jLong* x_strides = input.stridesOf();
				const Nd4jLong* z_strides = output.stridesOf();
				Nd4jLong* x_bases = &(x_shapeInfo[1]);
				bool isContinous = false;
				int posOfNonUnityDim;
				bias.isCommonVector(posOfNonUnityDim);
				const Nd4jLong rank = output.rankOf();
				const Nd4jLong yStrideC = bias.strideAt(posOfNonUnityDim);
#if defined(PRINT_VERBOSE)
				input.printShapeInfo("in");
				output.printShapeInfo("ou");
#endif
				if (sameOrder && sameStride) {

					isContinous = true;
					Nd4jLong calc_stride = 1;
					if (input.ordering() == 'c') {
						for (int i = rank - 1; i >= 0; i--) {
							bool is_eq = (calc_stride == x_strides[i]);
							isContinous &= is_eq;
							if (!isContinous) break;
							calc_stride = x_bases[i] * calc_stride;
						}

					}
					else {
						for (int i = 0; i < rank; i++) {
							bool is_eq = (calc_stride == x_strides[i]);
							isContinous &= is_eq;
							if (!isContinous) break;
							calc_stride = x_bases[i] * calc_stride;
						}

					}


				}//if ( sameOrder && same_stride)

				isContinous = !check_strides;


				if (isNCHW) {


					size_t numNC = 1;
					size_t numHW = 1;

					for (size_t i = 0; i < 2; i++) {
						numNC *= x_bases[i];
					}
					for (size_t i = 2; i < rank; i++) {
						numNC *= x_bases[i];
					}
				}
				else {
					//last C
					constexpr size_t BSIZE1 = 3 * MIN_NN * MIN_NN;
					constexpr size_t BSIZE2 = BSIZE1 + MIN_NN * MIN_NN;
					X  b_stack[BSIZE2] align32;
					std::unique_ptr<X[]> b_heap;
					const X* b_new;
					X* b_extra = nullptr;

					Nd4jLong numC = x_bases[rank - 1];
					Nd4jLong lastStride = x_strides[rank - 1];
					Nd4jLong z_lastStride = z_strides[rank - 1];
					size_t num = 1;

					for (size_t i = 0; i < rank; i++) {
						num *= x_bases[i];
					}
					char order = input.ordering();

					if (order == 'c') {
						size_t b_stack_size = BSIZE2;
						if (isContinous) {
							b_stack_size = BSIZE1;
							b_extra = &(b_stack[BSIZE1]);
						}
						b_new = flattened_bias(b, (X*)b_stack, b_stack_size, b_heap, numC, yStrideC);
					}

					if (sameOrder && lastStride == 1) {
						if (order == 'c') {
							if (isContinous) {
								inner_impl_continous_lastC<X>(x, b_new, z, b_extra, inplaceOp, num, numC);
							}
							else if (numC < vec<X>::count()) {
								// ordinary should be faster for small numC
								if (rank == 2) {
									inner_impl_generic_lastC_ordinary<X, 2>(x_shapeInfo, z_shapeInfo, x, b_new, z, inplaceOp, sameStride, num, numC);
								}
								else if (rank == 3) {
									inner_impl_generic_lastC_ordinary<X, 3>(x_shapeInfo, z_shapeInfo, x, b_new, z, inplaceOp, sameStride, num, numC);
								}
								else if (rank == 4) {
									inner_impl_generic_lastC_ordinary<X, 4>(x_shapeInfo, z_shapeInfo, x, b_new, z, inplaceOp, sameStride, num, numC);
								}
								else if (rank == 5) {
									inner_impl_generic_lastC_ordinary<X, 5>(x_shapeInfo, z_shapeInfo, x, b_new, z, inplaceOp, sameStride, num, numC);
								}
								else {
									inner_impl_generic_lastC_ordinary(x_shapeInfo, z_shapeInfo, x, b_new, z, inplaceOp, sameStride, num, numC);
								}
							}
							else {
								if (rank == 2) {
									inner_impl_generic_lastC_vect<X, 2>(x_shapeInfo, z_shapeInfo, x, b_new, z, inplaceOp, sameStride, num, numC);
								}
								else if (rank == 3) {
									inner_impl_generic_lastC_vect<X, 3>(x_shapeInfo, z_shapeInfo, x, b_new, z, inplaceOp, sameStride, num, numC);
								}
								else if (rank == 4) {
									inner_impl_generic_lastC_vect<X, 4>(x_shapeInfo, z_shapeInfo, x, b_new, z, inplaceOp, sameStride, num, numC);
								}
								else if (rank == 5) {
									inner_impl_generic_lastC_vect<X, 5>(x_shapeInfo, z_shapeInfo, x, b_new, z, inplaceOp, sameStride, num, numC);
								}
								else {
									inner_impl_generic_lastC_vect(x_shapeInfo, z_shapeInfo, x, b_new, z, inplaceOp, sameStride, num, numC);
								}
							}
						}
						else {

						}
					}
					else {
						//full strided ...implementation without order
						//inner_impl_strided(); 
						if (input.ordering()== 'c') {
							inner_impl_generic_orderC<X>(x_shapeInfo, z_shapeInfo, x, b_new, z, inplaceOp, num);
						}
						else {
							inner_impl_generic_orderF<X, Y>(x_shapeInfo, z_shapeInfo, x, b, z, inplaceOp, needCasting, num, yStrideC);
						}
					}

				}

			}


			//////////////////////////////////////////////////////////////////////////
			void addBias_Experimental(graph::Context& block, const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW, const bool check_strides)
			{

				// bias.rankOf() == 1 ? bias : bias.reshape(bias.ordering(), {bias.lengthOf()})
				BUILD_DOUBLE_SELECTOR(input.dataType(), bias.dataType(), addBiasE_, (input, bias, output, isNCHW, check_strides), FLOAT_TYPES, FLOAT_TYPES);
			}


			BUILD_DOUBLE_TEMPLATE(template void addBiasE_, (const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW, const bool check_strides), FLOAT_TYPES, FLOAT_TYPES);

		}
	}
}

