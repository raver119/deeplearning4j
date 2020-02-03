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
//#define OPT_TEMPLATE_INTRINSICS 1
#define OPT_USE_INNER_COORDS_1 1
//#define DEBUG 1
#if defined(DEBUG)
#define  PRINT_VERBOSE 1
#endif


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

				vec(T val) {
					for (size_t i = 0; i < count(); i++) {
						vals[i] = val;
					}
				}

				//other usefull constexpr like count and etc could be defined
				static constexpr size_t count() {
					return sizeof(vals) / sizeof(T);
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
					for (size_t i = 0; i < count(); i++) {
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

				vec(__m256 vf) : vals(vf) {
				}

				vec(float val) {
					vals = _mm256_set1_ps(val);
				}

				static constexpr size_t count() {
					return 8;
				}

				FORCEINLINE static vec loadu(const void* ptr) {

					return vec(_mm256_loadu_ps(reinterpret_cast<const float*>(ptr)));
				}

				FORCEINLINE vec operator+ (vec const& v) {
					vec ret{ _mm256_add_ps(vals, v.vals) };
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


//_add function that can be autovectorized
			template <typename T>
			FORCEINLINE  void _add(const T* __restrict xx, const T* __restrict yy, T* __restrict zz, const Nd4jLong& N) {
				PRAGMA_OMP_SIMD
				for (uint c = 0; c < N; c++)
					zz[c] = xx[c] + yy[c];

			}

			template <typename T>
			FORCEINLINE  void _add_inplace(T* __restrict xx, const T* __restrict yy, const Nd4jLong& N) {
				PRAGMA_OMP_SIMD
				for (uint c = 0; c < N; c++)
					xx[c] = xx[c] + yy[c];

			}

			template <typename T>
			FORCEINLINE  void _add_broadcast_inplace(T* __restrict xx, const T  yy, const Nd4jLong& N) {
				PRAGMA_OMP_SIMD
				for (uint c = 0; c < N; c++)
					xx[c] = xx[c] + yy;

			}


			template <typename T>
			FORCEINLINE  void _add_broadcast(const T* __restrict xx, const T  yy, T* __restrict zz, const Nd4jLong& N) {
				PRAGMA_OMP_SIMD
				for (uint c = 0; c < N; c++)
					zz[c] = xx[c] + yy;

			}

#if defined(OPT_TEMPLATE_INTRINSICS)
			template <typename T>
			FORCEINLINE  void _add_inplace_ordinary(T* __restrict xx, const T* __restrict yy, const Nd4jLong& N) {

				for (uint c = 0; c < N; c++)
					xx[c] = xx[c] + yy[c];

			}

			template <typename T>
			FORCEINLINE  void _add_ordinary(const T* __restrict xx, const T* __restrict yy, T* __restrict zz, const Nd4jLong& N) {
				//ordinary one will be used in case we dont know vectorizable size
				for (uint c = 0; c < N; c++)
					zz[c] = xx[c] + yy[c];

			}

			template <typename T>
			FORCEINLINE  void _add_broadcast_inplace_ordinary(T* __restrict xx, const T  yy, const Nd4jLong& N) {
				//ordinary one will be used in case we dont know vectorizable size
				for (uint c = 0; c < N; c++)
					xx[c] = xx[c] + yy;

			}

			template <typename T>
			FORCEINLINE  void _add_broadcast_ordinary(T* __restrict xx, const T  yy, T* __restrict zz, const Nd4jLong& N) {
				//ordinary one will be used in case we dont know vectorizable size
				for (uint c = 0; c < N; c++)
					zz[c] = xx[c] + yy;

			}
#endif



			//<editor-fold desc="INNER_ADD_WITH_MANUAL_VECINT_STYLES">
#pragma region INNER_ADD_WITH_MANUAL_VECINT_STYLES

#if defined(OPT_DEF_INTRINSICS) 
			template <>
			FORCEINLINE  void _add<float>(const float* __restrict xx, const float* __restrict yy, float* __restrict zz, const Nd4jLong& N) {
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
			FORCEINLINE  void _add<float>(const float* __restrict xx, const float* __restrict yy, float* __restrict zz, const Nd4jLong& N) {
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

			template <>
			FORCEINLINE  void _add_broadcast<float>(const float* __restrict xx, const float yy, float* __restrict zz, const Nd4jLong& N) {
				//manual instrinsics 
				using vecf = vec<float>;
				size_t nd = N & (-32);
				auto vy = vecf(yy);
				//prefetchl(&xx[0]);
				//prefetchwl(&zz[0]);
				if (nd >= 32) {
					//prefetchl(&xx[32]); 
					size_t i = 0;
					auto vx0 = vecf::loadu(xx);
					auto vx1 = vecf::loadu(xx + 8);
					auto vx2 = vecf::loadu(xx + 16);
					auto vx3 = vecf::loadu(xx + 24);


					auto vz0 = vx0 + vy;
					auto vz1 = vx1 + vy;
					auto vz2 = vx2 + vy;
					auto vz3 = vx3 + vy;
					for (i = 0; i < nd - 32; i += 32) {
						//prefetchl(&xx[i + 32+32]);
						//prefetchwl(&zz[i + 32]);
						vx0 = vecf::loadu(xx + i + 32);
						vx1 = vecf::loadu(xx + i + 8 + 32);
						vx2 = vecf::loadu(xx + i + 16 + 32);
						vx3 = vecf::loadu(xx + i + 24 + 32);



						vz0.storeu(zz + i);
						vz1.storeu(zz + i + 8);
						vz2.storeu(zz + i + 16);
						vz3.storeu(zz + i + 24);


						vz0 = vx0 + vy;
						vz1 = vx1 + vy;
						vz2 = vx2 + vy;
						vz3 = vx3 + vy;

					}

					vz0.storeu(zz + i);
					vz1.storeu(zz + i + 8);
					vz2.storeu(zz + i + 16);
					vz3.storeu(zz + i + 24);

				}
				for (size_t i = nd; i < N; i++)
					zz[i] = xx[i] + yy;

			}

#endif
#pragma endregion
			//</editor-fold>

#if defined(PRINT_VERBOSE)
#define print_verbose_output(fmt,  ...) nd4j_printf(fmt, __VA_ARGS__) 
#else
#define print_verbose_output(fmt,  ...)  
#endif
			int  parallel_for2(FUNC_1D function, int64_t start, int64_t stop, int64_t inc, size_t type_size = sizeof(float), uint32_t req_numThreads = nd4j::Environment::getInstance()->maxMasterThreads()) {
				if (start > stop)
					throw std::runtime_error("Threads::parallel_for got start > stop");


				auto num_elements = (stop - start);
				//this way we preserve inc starts offset
				//so we will adjust considering delta not num_elements
				auto delta = (stop - start) / inc;
				print_verbose_output("inc %d \n", inc);
				// in some cases we just fire func as is
				if (delta == 0 || req_numThreads == 1) {
					function(0, start, stop, inc);
					return 1;
				}


				int numThreads = 0;
#if  0
				int adjusted_numThreads = 2;
#else
				int adjusted_numThreads = samediff::ThreadsHelper::numberOfThreads(req_numThreads, (num_elements * sizeof(double)) / (200 * type_size));
#endif
				if (adjusted_numThreads > delta)
					adjusted_numThreads = delta;

				// shortcut
				if (adjusted_numThreads <= 1) {
					function(0, start, stop, inc);
					return 1;
				}

				//take span as ceil  
				auto spand = std::ceil((double)delta / (double)adjusted_numThreads);

				numThreads = static_cast<int>(std::ceil((double)delta / spand));
				auto span = static_cast<Nd4jLong>(spand);
				print_verbose_output("span  %d \n", span);
				auto ticket = samediff::ThreadPool::getInstance()->tryAcquire(numThreads);
				if (ticket != nullptr) {

					//tail_add is additional value of the last part
					//it could be negative or positive
					//we will spread that value across
					auto tail_add = delta - numThreads * span;
					Nd4jLong begin = 0;
					Nd4jLong end = 0;
					print_verbose_output("tail_add %d \n", tail_add);
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
						end = begin + span1 * inc;
						print_verbose_output("enque_1 start %d stop %d count:%d \n", begin, end, span1 * inc);
						// putting the task into the queue for a given thread
						ticket->enqueue(i, numThreads, function, begin, end, inc);
						begin = end;
					}
					for (int i = last; i < numThreads - 1; i++) {
						end = begin + span2 * inc;
						print_verbose_output("enque_2 start %d stop %d count:%d \n", begin, end, span2 * inc);
						// putting the task into the queue for a given thread
						ticket->enqueue(i, numThreads, function, begin, end, inc);
						begin = end;
					}
					//for last one enqueue last offset as stop
					//we need it in case our ((stop-start) % inc ) > 0
					ticket->enqueue(numThreads - 1, numThreads, function, begin, stop, inc);
					print_verbose_output("enque tail: %d \n", stop - begin);
					// block and wait till all threads finished the job
					ticket->waitAndRelease();

					// we tell that parallelism request succeeded
					return numThreads;
				}
				else {
					// if there were no threads available - we'll execute function right within current thread
					function(0, start, stop, inc);

					// we tell that parallelism request declined
					return 1;
				}
			}


			static constexpr size_t MIN_NN = 32;
			static constexpr size_t MIN_NN_K = 4;


			template<typename X, typename Y>
			static typename std::enable_if<std::is_same<X, Y>::value, const X*>::type
				flattened_bias(const Y* b_real, X* b_stack, const size_t b_stack_size, std::unique_ptr<X[]>& b_heap, const Nd4jLong num, Nd4jLong yStrideC)
			{
				print_verbose_output("flattened_bias %d\n", 1);
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
			static typename std::enable_if<!std::is_same<X, Y>::value, const X*>::type
				flattened_bias(const Y* b_real, X* b_stack, const size_t b_stack_size, std::unique_ptr<X[]>& b_heap, const Nd4jLong num, Nd4jLong yStrideC)
			{
				print_verbose_output("flattened_bias %d\n", 2);
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

			template<typename T, size_t constRank>
			static void channel_atTheEnd_stride1_C(const Nd4jLong*& x_strides, const Nd4jLong*& bases, T* x, const T* b, T* z, const bool& inplace, const Nd4jLong& start, const Nd4jLong& stop, const Nd4jLong& inc)
			{

				print_verbose_output("channel_atTheEnd_stride1_C %d\n", 1);

				Nd4jLong coords_p[constRank];

				// (stop-start) % inc == 0 because  we  handled inside partitioning using the channel size
				size_t loop_count = (stop - start) / inc;
				index2coords_C(start, constRank, bases, coords_p);
				size_t offset_p = offset_from_coords(x_strides, coords_p, constRank);

				if (!inplace) {
#if defined(OPT_TEMPLATE_INTRINSICS)
					if (inc >= vec<T>::count()) {
#endif
						for (size_t i = 0; i < loop_count; i++) {
							_add(&(x[offset_p]), b, &(z[offset_p]), inc);
							offset_p = inc_by_coords<constRank - 1>(bases, x_strides, coords_p, offset_p);
						}
#if defined(OPT_TEMPLATE_INTRINSICS)
					}
					else {

						for (size_t i = 0; i < loop_count; i++) {
							_add_ordinary(&(x[offset_p]), b, &(z[offset_p]), inc);
							offset_p = inc_by_coords<constRank - 1>(bases, x_strides, coords_p, offset_p);
						}
					}
#endif

				}
				else {
#if defined(OPT_TEMPLATE_INTRINSICS)
					if (inc >= vec<T>::count()) {
#endif
						for (size_t i = 0; i < loop_count; i++) {
							_add_inplace(&(x[offset_p]), b, inc);
							offset_p = inc_by_coords<constRank - 1>(bases, x_strides, coords_p, offset_p);
						}
#if defined(OPT_TEMPLATE_INTRINSICS)
					}
					else {

						for (size_t i = 0; i < loop_count; i++) {
							_add_inplace_ordinary(&(x[offset_p]), b, inc);
							offset_p = inc_by_coords<constRank - 1>(bases, x_strides, coords_p, offset_p);
						}
					}
#endif
				}
			}


			template<typename T, size_t constRank >
			static void channel_atTheEnd_generic_C(const Nd4jLong* bases, const Nd4jLong* x_strides, const Nd4jLong* z_strides, const bool& inplaceOp, const bool same_stride, const bool same_order, T* x, const T* b, T* z, Nd4jLong start, Nd4jLong stop, Nd4jLong inc) {


				//just ensure that passed sameStride is correct,  because when bases are equal orders matters 

				bool sameOrderStride = same_order && same_stride;

				if (sameOrderStride && x_strides[constRank - 1] == 1) {
					print_verbose_output("channel_atTheEnd_generic_C %d\n", 1);
					channel_atTheEnd_stride1_C<T, constRank>(x_strides, bases, x, b, z, inplaceOp, start, stop, inc);
				}
				else {

					Nd4jLong coords_p[constRank];
					// (stop-start) % inc == 0 because  we  handled inside partitioning using the channel size
					index2coords_C(start, constRank, bases, coords_p);
					size_t loop_count = (stop - start) / inc;
					pair_size_t offset = offset_from_coords(x_strides, z_strides, coords_p, constRank);

					if (same_order && z_strides[constRank - 1] == 1 && x_strides[constRank - 1] == 1) {
						print_verbose_output("channel_atTheEnd_generic_C %d\n", 2);
						/* bases are equal with different strides , but the last one is 1. So we can still vectorize it  */
#if defined(OPT_TEMPLATE_INTRINSICS)
						if (inc >= vec<T>::count()) {
#endif
							for (size_t i = 0; i < loop_count; i++) {
								_add(&(x[offset.first]), b, &(z[offset.second]), inc);
								offset = inc_by_coords_zip<constRank - 1>(bases, x_strides, z_strides, coords_p, offset);
							}
#if defined(OPT_TEMPLATE_INTRINSICS)
						}
						else {
							for (size_t i = 0; i < loop_count; i++) {
								_add_ordinary(&(x[offset.first]), b, &(z[offset.second]), inc);
								offset = inc_by_coords_zip<constRank - 1>(bases, x_strides, z_strides, coords_p, offset);
							}
						}
#endif
					}
					else {
						print_verbose_output("channel_atTheEnd_generic_C %d\n", 3);
						Nd4jLong x_stride = x_strides[constRank - 1];
						Nd4jLong z_stride = z_strides[constRank - 1];
						for (size_t i = 0; i < loop_count; i++) {
							T* xx = &(x[offset.first]);
							T* zz = &(z[offset.second]);
							for (size_t j = 0; j < inc; j++)
								zz[j * z_stride] = xx[j * x_stride] + b[j];
							offset = inc_by_coords_zip<constRank - 1>(bases, x_strides, z_strides, coords_p, offset);
						}
					}
				}

			}

			/**
			* this is our main optimization which  benefits from everything for the continuous last_channel C order case
			* as it is intended for full continous we do not need any rank info
			*/
			template<typename T>
			void channel_atTheEnd_continous_C(T* x, const T* b, T* z, bool inplaceOp, Nd4jLong start, Nd4jLong stop, Nd4jLong inc) {
				print_verbose_output("channel_atTheEnd_continous_C %d\n", 0);

				size_t nums = (stop - start);
				size_t num_inc = nums - nums % inc;
				size_t offset_p = start;
				if (inplaceOp) {
					for (size_t i = 0; i < num_inc; i += inc) {
						_add_inplace<T>(&(x[offset_p]), b, inc);
						offset_p += inc;
					}

					if (nums > num_inc)
						_add_inplace<T>(&(x[offset_p]), b, nums - num_inc);
				}
				else {
					prefetch_range_rl((char*)b, 1024 < inc ? 1024 : inc);
					print_verbose_output("start %d stop %d inc %d ::count %d \n", start, stop, inc, nums);
					size_t offset_p = start;
					for (size_t i = 0; i < num_inc; i += inc) {
						_add<T>(&(x[offset_p]), b, &(z[offset_p]), inc);
						offset_p += inc;
					}

					if (nums > num_inc)
						_add<T>(&(x[offset_p]), b, &(z[offset_p]), nums - num_inc);
				}

			}



			template<typename T, typename T2, size_t constRank, size_t b_index, size_t skip>
			static void channel_generic_stride_skip_F(const Nd4jLong*& x_strides, const Nd4jLong*& bases, T* x, const T2* b, T* z, const bool& inplace, const Nd4jLong yStrideC, const Nd4jLong& start, const Nd4jLong& stop, const Nd4jLong& inc)
			{



				Nd4jLong coords_p[constRank];

				// (stop-start) % inc == 0 because  we  handled inside partitioning using the channel size
				size_t loop_count = (stop - start) / inc;
				index2coords_F(start, constRank, bases, coords_p);
				size_t offset_p = offset_from_coords(x_strides, coords_p, constRank);

				if (!inplace) {
#if defined(OPT_TEMPLATE_INTRINSICS)
					if (inc >= vec<T>::count()) {
						print_verbose_output("channel_generic_stride_skip_F %d %d %d skip %d\n", 1, loop_count, inc, skip);
#endif
						for (size_t i = 0; i < loop_count; i++) {
							T yy = static_cast<T>(b[coords_p[b_index] * yStrideC]);
							//print_verbose_output(" %d %f\n", coords_p[b_index] * yStrideC , yy);
							_add_broadcast(&(x[offset_p]), yy, &(z[offset_p]), inc);
							offset_p = inc_by_coords<constRank, skip, false>(bases, x_strides, coords_p, offset_p);
						}
#if defined(OPT_TEMPLATE_INTRINSICS)
					}
					else {
						print_verbose_output("channel_generic_stride_skip_F %d\n", 2);
						for (size_t i = 0; i < loop_count; i++) {
							T yy = static_cast<T>(b[coords_p[b_index] * yStrideC]);
							_add_broadcast_ordinary(&(x[offset_p]), yy, &(z[offset_p]), inc);
							offset_p = inc_by_coords<constRank, skip, false>(bases, x_strides, coords_p, offset_p);
						}
					}
#endif
				}
				else {
#if defined(OPT_TEMPLATE_INTRINSICS)
					if (inc >= vec<T>::count()) {
#endif
						print_verbose_output("channel_generic_stride_skip_F %d\n", 3);
						for (size_t i = 0; i < loop_count; i++) {
							T yy = static_cast<T>(b[coords_p[b_index] * yStrideC]);
							_add_broadcast_inplace(&(x[offset_p]), yy, inc);
							offset_p = inc_by_coords<constRank, skip, false>(bases, x_strides, coords_p, offset_p);
						}
#if defined(OPT_TEMPLATE_INTRINSICS)
					}
					else {

						print_verbose_output("channel_generic_stride_skip_F %d\n", 4);
						for (size_t i = 0; i < loop_count; i++) {
							T yy = static_cast<T>(b[coords_p[b_index] * yStrideC]);
							_add_broadcast_inplace_ordinary(&(x[offset_p]), yy, inc);
							offset_p = inc_by_coords<constRank, skip, false>(bases, x_strides, coords_p, offset_p);
						}
					}
#endif
				}
			}


			template<typename T, typename T2, size_t constRank, size_t b_index>
			void channel_generic_F(const Nd4jLong* bases, const Nd4jLong* x_strides, const Nd4jLong* z_strides, const bool& inplaceOp, const bool same_stride, const bool same_order, const Nd4jLong yStrideC, T* x, const T2* b, T* z, Nd4jLong start, Nd4jLong stop, Nd4jLong inc) {

				//just ensure that passed sameStride is correct,  because when bases are equal orders matters 

				bool sameOrderStride = same_order && same_stride;

				if (sameOrderStride && x_strides[0] == 1) {
					print_verbose_output("channel_generic_F %d channel index %d\n", 1, b_index);
					channel_generic_stride_skip_F<T, T2, constRank,  b_index, 1>(x_strides, bases, x, b, z, inplaceOp, yStrideC, start, stop, inc);
				}
				else {

					Nd4jLong coords_p[constRank];
					// (stop-start) % inc == 0 because  we  handled inside partitioning using the channel size
					index2coords_F(start, constRank, bases, coords_p);
					size_t loop_count = (stop - start) / inc;
					pair_size_t offset = offset_from_coords(x_strides, z_strides, coords_p, constRank);

					if (same_order && z_strides[0] == 1 && x_strides[0] == 1) {
						print_verbose_output("channel_generic_F %d channel index %d\n", 2, b_index);
#if defined(OPT_TEMPLATE_INTRINSICS)
						if (inc >= vec<T>::count()) {
#endif
							for (size_t i = 0; i < loop_count; i++) {
								T yy = static_cast<T>(b[coords_p[b_index] * yStrideC]);
								_add_broadcast(&(x[offset.first]), yy, &(z[offset.second]), inc);
								offset = inc_by_coords_zip<constRank, 1, false>(bases, x_strides, z_strides, coords_p, offset);
							}
#if defined(OPT_TEMPLATE_INTRINSICS)
						}
						else {
							for (size_t i = 0; i < loop_count; i++) {
								T yy = static_cast<T>(b[coords_p[b_index] * yStrideC]);
								_add_broadcast_ordinary(&(x[offset.first]), yy, &(z[offset.second]), inc);
								offset = inc_by_coords_zip<constRank, 1, false>(bases, x_strides, z_strides, coords_p, offset);
							}
						}
#endif
					}
					else {
						print_verbose_output("channel_generic_F %d channel index %d\n", 3, b_index);
						Nd4jLong x_stride = x_strides[0];
						Nd4jLong z_stride = z_strides[0];
						for (size_t i = 0; i < loop_count; i++) {
							T* xx = &(x[offset.first]);
							T* zz = &(z[offset.second]);
							T yy = static_cast<T>(b[coords_p[b_index] * yStrideC]);
							for (size_t j = 0; j < inc; j++)
								zz[j * z_stride] = xx[j * x_stride] + yy;
							offset = inc_by_coords_zip<constRank, 1, false>(bases, x_strides, z_strides, coords_p, offset);
						}
					}
				}

			}


			template<typename T, typename T2>
			void channel_NC_continous_numHW_C(Nd4jLong rank, const Nd4jLong* bases, const Nd4jLong* x_strides, T* x, const T2* b, T* z, bool inplaceOp, const Nd4jLong yStrideC, Nd4jLong start, Nd4jLong stop, Nd4jLong inc) {
				

				Nd4jLong coords_p[MAX_RANK];

				// (stop-start) % inc == 0 because  we  handled inside partitioning using the channel size
				size_t loop_count = (stop - start) / inc;
				index2coords_C(start, rank, bases, coords_p);
				size_t offset_p = offset_from_coords(x_strides, coords_p, rank);

				//partitioning was done using numHW, so we can increment from rank 2

				if (inplaceOp) {
					print_verbose_output("channel_NC_continous_numHW_C %d\n", 0);
					for (size_t i = 0; i < loop_count; i++) {
						T yy = static_cast<T>(b[coords_p[1] * yStrideC]);
						_add_broadcast_inplace(&(x[offset_p]), yy, inc);
						offset_p = inc_by_coords<2>(bases, x_strides, coords_p, offset_p);
					}
				}
				else {
					if (yStrideC == 1) {
						print_verbose_output("channel_NC_continous_numHW_C %d\n", 2);
						for (size_t i = 0; i < loop_count; i++) {
							T yy = static_cast<T>(b[coords_p[1]]);
							//print_verbose_output("%d %d %f offset %d\n", coords_p[0], coords_p[1], yy,offset_p);
							_add_broadcast(&(x[offset_p]), yy, &(z[offset_p]), inc);
							offset_p = inc_by_coords<2>(bases, x_strides, coords_p, offset_p);
						}
					}
					else {
						print_verbose_output("channel_NC_continous_numHW_C %d\n", 3);
						for (size_t i = 0; i < loop_count; i++) {
							T yy = static_cast<T>(b[coords_p[1] * yStrideC]);
							_add_broadcast(&(x[offset_p]), yy, &(z[offset_p]), inc);
							offset_p = inc_by_coords<2>(bases, x_strides, coords_p, offset_p);
						}
					}
				}

			}

			template<typename T, typename T2, size_t constRank>
			static void channel_NC_stride1_C(const Nd4jLong*& x_strides, const Nd4jLong*& bases, T* x, const T2* b, T* z, const bool& inplace, const Nd4jLong yStrideC, const Nd4jLong& start, const Nd4jLong& stop, const Nd4jLong& inc)
			{

				 

				Nd4jLong coords_p[constRank];

				// (stop-start) % inc == 0 because  we  handled inside partitioning using the channel size
				size_t loop_count = (stop - start) / inc;
				index2coords_C(start, constRank, bases, coords_p);
				size_t offset_p = offset_from_coords(x_strides, coords_p, constRank);

				if (!inplace) {
#if defined(OPT_TEMPLATE_INTRINSICS)
					if (inc >= vec<T>::count()) {
#endif
						print_verbose_output("channel_NC_stride1_C %d %d %d\n", 1,loop_count,inc);

						for (size_t i = 0; i < loop_count; i++) {
							T yy = static_cast<T>(b[coords_p[1] * yStrideC]);
							//print_verbose_output(" %d %f\n", coords_p[1] * yStrideC , yy);
							_add_broadcast(&(x[offset_p]), yy, &(z[offset_p]), inc);
							offset_p = inc_by_coords<constRank - 1>(bases, x_strides, coords_p, offset_p);
						}
#if defined(OPT_TEMPLATE_INTRINSICS)
					}
					else {
					    print_verbose_output("channel_NC_stride1_C %d\n", 2);
						for (size_t i = 0; i < loop_count; i++) {
							T yy = static_cast<T>(b[coords_p[1] * yStrideC]);
							_add_broadcast_ordinary(&(x[offset_p]), yy, &(z[offset_p]), inc);
							offset_p = inc_by_coords<constRank - 1>(bases, x_strides, coords_p, offset_p);
						}
					}
#endif

				}
				else {
#if defined(OPT_TEMPLATE_INTRINSICS)
					if (inc >= vec<T>::count()) {
#endif
						print_verbose_output("channel_NC_stride1_C %d\n", 3);
						for (size_t i = 0; i < loop_count; i++) {
							T yy = static_cast<T>(b[coords_p[1] * yStrideC]);
							_add_broadcast_inplace(&(x[offset_p]), yy, inc);
							offset_p = inc_by_coords<constRank - 1>(bases, x_strides, coords_p, offset_p);
						}
#if defined(OPT_TEMPLATE_INTRINSICS)
					}
					else {

						print_verbose_output("channel_NC_stride1_C %d\n", 4);
						for (size_t i = 0; i < loop_count; i++) {
							T yy = static_cast<T>(b[coords_p[1] * yStrideC]);
							_add_broadcast_inplace_ordinary(&(x[offset_p]), yy, inc);
							offset_p = inc_by_coords<constRank - 1>(bases, x_strides, coords_p, offset_p);
						}
					}
#endif
				}
			}


			template<typename T, typename T2, size_t constRank >
			void channel_NC_generic_C(const Nd4jLong* bases, const Nd4jLong* x_strides, const Nd4jLong* z_strides, const bool& inplaceOp, const bool same_stride, const bool same_order, const Nd4jLong yStrideC, T* x, const T2* b, T* z, Nd4jLong start, Nd4jLong stop, Nd4jLong inc) {
				 
				//just ensure that passed sameStride is correct,  because when bases are equal orders matters 

				bool sameOrderStride = same_order && same_stride;

				if (sameOrderStride && x_strides[constRank - 1] == 1) {
					print_verbose_output("channel_NC_generic_C %d\n", 1);
					channel_NC_stride1_C<T, T2, constRank>(x_strides, bases, x, b, z, inplaceOp, yStrideC, start, stop, inc);
				}
				else {

					Nd4jLong coords_p[constRank];
					// (stop-start) % inc == 0 because  we  handled inside partitioning using the channel size
					index2coords_C(start, constRank, bases, coords_p);
					size_t loop_count = (stop - start) / inc;
					pair_size_t offset = offset_from_coords(x_strides, z_strides, coords_p, constRank);

					if (same_order && z_strides[constRank - 1] == 1 && x_strides[constRank - 1] == 1) {
						print_verbose_output("channel_NC_generic_C %d\n", 2);
						/* bases are equal with different strides , but the last one is 1. So we can still vectorize it  */
#if defined(OPT_TEMPLATE_INTRINSICS)
						if (inc >= vec<T>::count()) {
#endif
							for (size_t i = 0; i < loop_count; i++) {
								T yy = static_cast<T>(b[coords_p[1] * yStrideC]);
								_add_broadcast(&(x[offset.first]), yy, &(z[offset.second]), inc);
								offset = inc_by_coords_zip<constRank - 1>(bases, x_strides, z_strides, coords_p, offset);
							}
#if defined(OPT_TEMPLATE_INTRINSICS)
						}
						else {
							for (size_t i = 0; i < loop_count; i++) {
								T yy = static_cast<T>(b[coords_p[1] * yStrideC]);
								_add_broadcast_ordinary(&(x[offset.first]), yy, &(z[offset.second]), inc);
								offset = inc_by_coords_zip<constRank - 1>(bases, x_strides, z_strides, coords_p, offset);
							}
						}
#endif
					}
					else {
						print_verbose_output("channel_NC_generic_C %d\n", 3);
						Nd4jLong x_stride = x_strides[constRank - 1];
						Nd4jLong z_stride = z_strides[constRank - 1];
						for (size_t i = 0; i < loop_count; i++) {
							T* xx = &(x[offset.first]);
							T* zz = &(z[offset.second]);
							T yy = static_cast<T>(b[coords_p[1] * yStrideC]);
							for (size_t j = 0; j < inc; j++)
								zz[j * z_stride] = xx[j * x_stride] + yy;
							offset = inc_by_coords_zip<constRank - 1>(bases, x_strides, z_strides, coords_p, offset);
						}
					}
				}

			}


			template <typename X, typename Y>
			static void addBiasE_(const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW
#if 1
				, const bool force_non_continuous
#endif			
			) {

				Nd4jLong* x_shapeInfo = input.getShapeInfo();
				Nd4jLong* z_shapeInfo = output.getShapeInfo();

				X* x = input.bufferAsT<X>();
				X* z = output.bufferAsT<X>();
				const Y* b = bias.bufferAsT<Y>();

				const Nd4jLong  rank = x_shapeInfo[0];
				const Nd4jLong* bases = &(x_shapeInfo[1]);
				const Nd4jLong* x_strides = &(x_shapeInfo[rank + 1]);
				const Nd4jLong* z_strides = &(z_shapeInfo[rank + 1]);

				const bool inplaceOp = (x == z);
				constexpr bool NEED_CASTING = !std::is_same<X, Y>::value;
				const bool same_order = inplaceOp || (input.ordering() == output.ordering());
				const bool channel_atTheEnd = !isNCHW;
				const bool same_stride = inplaceOp || shape::strideEquals(x_shapeInfo, z_shapeInfo);
				bool isContinuous = false;

				int posOfNonUnityDim;
				bias.isCommonVector(posOfNonUnityDim);
				const Nd4jLong yStrideC = bias.strideAt(posOfNonUnityDim);
#if defined(PRINT_VERBOSE)
				input.printShapeInfo("in");
				output.printShapeInfo("ou");
 
				//assert for our case
				Nd4jLong channel_size = isNCHW ? bases[1] : bases[rank - 1];
				print_verbose_output("channel %d bias %d\n", channel_size, bias.sizeAt(posOfNonUnityDim));
				assert(bias.sizeAt(posOfNonUnityDim) == channel_size);
#endif
				char order = input.ordering();

				if (same_order && same_stride) {
					isContinuous = check_continuity(order, bases, x_strides, rank);

#if 1
					//this is for testing
					isContinuous = !force_non_continuous;
#endif

				}//if ( sameOrder && same_stride)
				print_verbose_output("isContinuous %d\n", isContinuous);
				bool treat_as_lastC = false;
				//
				if (rank == 2 && isNCHW ) {
					//we believe we better treat it as channel at the end case;
					treat_as_lastC = true;
				}

				if (channel_atTheEnd || treat_as_lastC) {
					//N..HWC case here

					//flattened bias variables
					constexpr size_t BSIZE1 = 3 * MIN_NN * MIN_NN;
					constexpr size_t BSIZE2 = BSIZE1 + MIN_NN * MIN_NN;
					X  flatBias_stack[BSIZE2] align32;
					std::unique_ptr<X[]> flatBias_heap;
					const X* bias_new;
					X* bias_extra = nullptr;

					size_t total_num = 1;

					for (size_t i = 0; i < rank; i++) {
						total_num *= bases[i];
					}


					Nd4jLong inc;
					size_t rank_skip = 1;
					if (order == 'c') {
						size_t b_stack_size = BSIZE2;
						inc = bases[rank - 1];
						if (isContinuous) {
							//for continous we need extra stack memory
							// to create vectorizable bias from small size
							b_stack_size = BSIZE1;
							bias_extra = &(flatBias_stack[BSIZE1]);
						}
						bias_new = flattened_bias(b, (X*)flatBias_stack, b_stack_size, flatBias_heap, inc, yStrideC);

						if (isContinuous && inc < MIN_NN_K * MIN_NN && total_num > inc* MIN_NN_K) {
							//for small size where total_num is sufficient  we need to recreate vectorizable buffer

							size_t old_inc = inc;

							//sizeof bias_extra is MIN_NN * MIN_NN 
							size_t new_inc = inc < MIN_NN ? inc * MIN_NN : inc * MIN_NN / MIN_NN_K;
							//if there is a room then lets multiply
							new_inc = (new_inc * MIN_NN_K <= total_num && new_inc < MIN_NN * MIN_NN / MIN_NN_K) ? MIN_NN_K * new_inc : new_inc;

							for (size_t i = 0; i < new_inc; i += inc) {
								//copy to our buffer
								X* cp = &(bias_extra[i]);
								for (size_t j = 0; j < inc; j++) {
									cp[j] = bias_new[j];
								}
							}
							//vectorizable buffer
							inc = new_inc;
							bias_new = bias_extra;
						}


					}
					else {
						inc = bases[0];
						if (isContinuous) {
							//we can choose other inc and index for that case
							//but for now lets choose all till the last one
							uint32_t req_numThreads = nd4j::Environment::getInstance()->maxMasterThreads();
							isContinuous = false;
							if (rank > 2) {
								if (req_numThreads < 2 || bases[rank - 1] >= req_numThreads  ) {
									inc = total_num / bases[rank - 1];
									isContinuous = true;
									print_verbose_output("inc %d %d\n", bases[0], inc);
									rank_skip = rank - 1;
								}
								else if (rank > 3 && bases[rank - 1] * bases[rank - 2] >= req_numThreads) {
									inc = total_num / bases[rank - 1] / bases[rank - 2]; //for continuous case it is its stride
									rank_skip = rank - 2;
									isContinuous = true;
								}
							}
						}
					}



					FUNC_1D func = [order, isContinuous, rank, x, b, bias_new, z, x_shapeInfo, z_shapeInfo, same_stride, same_order, yStrideC, rank_skip]

					(uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
						const Nd4jLong  rank = x_shapeInfo[0];
						const Nd4jLong* bases = &(x_shapeInfo[1]);
						const Nd4jLong* x_strides = &(x_shapeInfo[rank + 1]);
						const Nd4jLong* z_strides = &(z_shapeInfo[rank + 1]);

						const bool inplaceOp = (x == z);
						if (order == 'c') {
							if (isContinuous) {
								channel_atTheEnd_continous_C(x, bias_new, z, inplaceOp, start, stop, increment);
							}
							// rank is in [2,5]
							else if (rank == 4) {
								channel_atTheEnd_generic_C<X, 4>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, x, bias_new, z, start, stop, increment);

							}
							else if (rank == 5) {
								channel_atTheEnd_generic_C<X, 5>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, x, bias_new, z, start, stop, increment);
							}
							else if (rank == 2) {
								channel_atTheEnd_generic_C<X, 2>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, x, bias_new, z, start, stop, increment);
							}
							else if (rank == 3) {
								channel_atTheEnd_generic_C<X, 3>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, x, bias_new, z, start, stop, increment);
							}
						}
						else {

							//generic F case  
							if (isContinuous) {
								if (rank == 4) {
									if (rank_skip == rank - 2) {
										channel_generic_stride_skip_F<X, Y, 4, 3, 2>(x_strides, bases, x, b, z, inplaceOp, yStrideC, start, stop, increment);
									}
									else {
										channel_generic_stride_skip_F<X, Y, 4, 3, 3>(x_strides, bases, x, b, z, inplaceOp, yStrideC, start, stop, increment);
									}
									
								}
								else if (rank == 5) {
									if (rank_skip == rank - 2) {
										//skip==3
										channel_generic_stride_skip_F<X, Y, 5, 4, 3>(x_strides, bases, x, b, z, inplaceOp, yStrideC, start, stop, increment);
									}
									else {
										channel_generic_stride_skip_F<X, Y, 5, 4, 4>(x_strides, bases, x, b, z, inplaceOp, yStrideC, start, stop, increment);
									}

								} 
								else if (rank == 3) {
									channel_generic_stride_skip_F<X, Y, 3, 2, 2>(x_strides, bases, x, b, z, inplaceOp, yStrideC, start, stop, increment);
								}
							}
							else if (rank == 4) {
								channel_generic_F<X, Y, 4, 3>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, x, b, z, start, stop, increment);
							}
							else if (rank == 5) {
								channel_generic_F<X, Y, 5, 4>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, x, b, z, start, stop, increment);
							}
							else if (rank == 2) {
								channel_generic_F<X, Y, 2, 1>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, x, b, z, start, stop, increment);
							}
							else if (rank == 3) {
								channel_generic_F<X, Y, 3, 2>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, x, b, z, start, stop, increment);
							}
							
						}
					};
					//
					parallel_for2(func, 0, total_num, inc);
				}
				else {
					//NC...HW case here
					size_t numNC = 1;
					size_t numHW = 1;

					for (size_t i = 0; i < 2; i++) {
						numNC *= bases[i];
					}
					for (size_t i = 2; i < rank; i++) {
						numHW *= bases[i];
					}

					Nd4jLong total_num = numNC * numHW;
					Nd4jLong inc  = (order == 'c') ? bases[rank - 1] : bases[0];

					if (order == 'c' && isContinuous) {
						//sometimes last dimension is too big and multithreading could suffer using unfair partitioning
						//so we will do it only when inc is smaller our value or multithreading turned off
						uint32_t req_numThreads = nd4j::Environment::getInstance()->maxMasterThreads();
						if (req_numThreads < 2 || numNC>= req_numThreads || inc <= 2 * 8196  || rank == 3) {
							inc = numHW;
							print_verbose_output("inc %d %d\n", inc,numHW);
						}
						else {
							//treat it as stride1c case
							isContinuous = false;
						}
					}

					FUNC_1D func = [order, isContinuous, rank, x, b, z, x_shapeInfo, z_shapeInfo, same_stride, same_order, yStrideC]

					(uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
						const Nd4jLong  rank = x_shapeInfo[0];
						const Nd4jLong* bases = &(x_shapeInfo[1]);
						const Nd4jLong* x_strides = &(x_shapeInfo[rank + 1]);
						const Nd4jLong* z_strides = &(z_shapeInfo[rank + 1]);

						const bool inplaceOp = (x == z);
						if (order == 'c') {
							if (isContinuous) {

								channel_NC_continous_numHW_C<X, Y>(rank, bases, x_strides, x, b, z, inplaceOp, yStrideC, start, stop, increment);
							}
							// rank is in [3,5]
							else if (rank == 4) {
								channel_NC_generic_C<X, Y, 4>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, x, b, z, start, stop, increment);

							}
							else if (rank == 5) {
								channel_NC_generic_C<X, Y, 5>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, x, b, z, start, stop, increment);
							}
							else if (rank == 3) {
								channel_NC_generic_C<X, Y, 3>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, x, b, z, start, stop, increment);
							}
						}
						else {
							 
							//the same can be applied for NCHW case
							//generic F case 

							//continous case is missing
							if (rank == 4) {
								channel_generic_F<X, Y, 4, 1>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, x, b, z, start, stop, increment);
							}
							else if (rank == 5) {
								channel_generic_F<X, Y, 5, 1>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, x, b, z, start, stop, increment);
							} 
							else if (rank == 3) {
								channel_generic_F<X, Y, 3, 1>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, x, b, z, start, stop, increment);
							}
						}
					};
					//
					parallel_for2(func, 0, total_num, inc);


				}

			}


			//////////////////////////////////////////////////////////////////////////
			void addBias_Experimental(graph::Context& block, const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW, const bool check_strides)
			{
				BUILD_DOUBLE_SELECTOR(input.dataType(), bias.dataType(), addBiasE_, (input, bias, output, isNCHW, check_strides), FLOAT_TYPES, FLOAT_TYPES);
			}


			BUILD_DOUBLE_TEMPLATE(template void addBiasE_, (const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW, const bool check_strides), FLOAT_TYPES, FLOAT_TYPES);

		}
	}
}

