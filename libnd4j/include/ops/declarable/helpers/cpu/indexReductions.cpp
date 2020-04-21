/*******************************************************************************
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

#include <type_traits>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <execution/Threads.h>
#include <execution/ThreadPool.h>
#include <helpers/LoopsCoordsHelper.h>
#include <ops/declarable/helpers/reductions.h>


namespace sd {
	namespace ops {
		namespace helpers {
 
		 
			template<typename X, typename Z, bool Last_Index_Faster = true>
			inline void argMaxInnerReduction(const int rank, const X * buffer, const Nd4jLong * bases, const Nd4jLong * strides, const Nd4jLong & outerLoopStart, const Nd4jLong& outerLoopStop, const Nd4jLong & innerLoopCount, const Nd4jLong & inner_stride, X& max, Z& argMax)
			{

				size_t offset = 0;
				Nd4jLong outerLoopCount = outerLoopStop - outerLoopStart;
				Nd4jLong coords[MAX_RANK] = {};
				Nd4jLong* ptr_coords = (Nd4jLong*)&coords;
				if (outerLoopStart > 0) {
				//	if (Last_Index_Faster) {
						sd::index2coords_C(outerLoopStart, rank-1, bases, ptr_coords);
				/*	}
					else {
						sd::index2coords_F(outerLoopStart, rank-1, &(bases[1]), ptr_coords);
					}*/
					offset = sd::offset_from_coords(strides, ptr_coords, rank);
				}

				Z startIndex = outerLoopStart * innerLoopCount;
				argMax = startIndex;
				max = buffer[offset];
				
				if (inner_stride == 1) {
#if 0
					nd4j_printf("___%s_________%d+\n", __PRETTY_FUNCTION__, 0);
#endif
					for (Z i = 0; i < outerLoopCount; i++) {
						const X *inner_buffer = &(buffer[offset]);
						//typename std::make_signed<Z>::type iArgMax = -1;
						 
							for (Z j = 0; j < innerLoopCount; j++) {
								//nd4j_printf("%f\n", inner_buffer[j]);
								if (inner_buffer[j] > max) {
									max = inner_buffer[j];
									argMax = startIndex + j;
								}

							}

						offset = inc_coords<true>(bases, strides, ptr_coords, offset, rank, 1);
						//if (iArgMax >= 0) argMax = startIndex + iArgMax;
						startIndex += innerLoopCount;
						
					}
				}
				else {
#if 0
					nd4j_printf("___%s_________%d+\n", __PRETTY_FUNCTION__, 1);
#endif
					 
						for (Z i = 0; i < outerLoopCount; i++) {
							const X* inner_buffer = &(buffer[offset]);
							//typename std::make_signed<Z>::type iArgMax = -1;
							 
								for (Z j = 0; j < innerLoopCount; j++) {
									if (inner_buffer[j * inner_stride] > max) {
										max = inner_buffer[j * inner_stride];
										argMax = startIndex + j;
									}

								}

							offset = inc_coords<true>(bases, strides, ptr_coords, offset, rank, 1);

							//offset = inc_coords<Last_Index_Faster>(bases, strides, ptr_coords, offset, rank, 1);
							//if (iArgMax >= 0) argMax = startIndex + iArgMax;
							startIndex += innerLoopCount;
						} 
				}
			}
 

			template<typename X, typename Z>
			FORCEINLINE void argMaxInnerReduction(const X* buffer, const Nd4jLong & loopCount, const Nd4jLong & inner_stride, X& max, Z& argMax)
			{
				argMax = 0;
				max = buffer[0];
				if (inner_stride == 1) {

					if (loopCount >= 256) {

						Nd4jLong loopCount4 = loopCount / 4;
						Nd4jLong loopCountEnd = loopCount4 + (loopCount & 3);

						const X* buffer_1 = buffer + loopCount4;
						const X* buffer_2 = buffer_1 + loopCount4;
						const X* buffer_3 = buffer_2 + loopCount4;
						X max_1 = *buffer_1;
						X max_2 = *buffer_2;
						X max_3 = *buffer_3;
						Z argMax_1 = 0;
						Z argMax_2 = 0;
						Z argMax_3 = 0;

						for (Z j = 0; j < loopCount4; j++) {
							if (buffer[j] > max) {
								max = buffer[j];
								argMax = j;
							}
							if (buffer_1[j] > max_1) {
								max_1 = buffer_1[j];
								argMax_1 = j;
							}
							if (buffer_2[j] > max_2) {
								max_2 = buffer_2[j];
								argMax_2 = j;
							}
							if (buffer_3[j] > max_3) {
								max_3 = buffer_3[j];
								argMax_3 = j;
							} 
						}

						//tail
						for (Z j = loopCount4; j < loopCountEnd; j++) {
							if (buffer_3[j] > max_3) {
								max_3 = buffer_3[j];
								argMax_3 = j;
							}
						}

						//merge
						argMax_1 += loopCount4;
						argMax_2 += 2*loopCount4;
						argMax_3 += 3*loopCount4;

						if (max_1 > max) {
							max = max_1;
							argMax = argMax_1;
						}
						if (max_2 > max) {
							max = max_2;
							argMax = argMax_2;
						}
						if (max_3 > max) {
							max = max_3;
							argMax = argMax_3;
						}


					}
					else {

					 
							for (Z j = 0; j < loopCount; j++) {
								if (buffer[j] > max) {
									max = buffer[j];
									argMax = j;
								}

							}
					}//<256
				}
				else {

					if (loopCount >= 256) {

						Nd4jLong loopCount4 = loopCount / 4;
						Nd4jLong loopCountEnd = loopCount4 + (loopCount & 3);

						const X* buffer_1 = buffer + inner_stride * loopCount4;
						const X* buffer_2 = buffer_1 + inner_stride * loopCount4;
						const X* buffer_3 = buffer_2 + inner_stride * loopCount4;
						X max_1 = *buffer_1;
						X max_2 = *buffer_2;
						X max_3 = *buffer_3;
						Z argMax_1 = 0;
						Z argMax_2 = 0;
						Z argMax_3 = 0;
						Nd4jLong j_offset = 0;
						for (Z j = 0; j < loopCount4; j++) {
							if (buffer[j_offset] > max) {
								max = buffer[j_offset];
								argMax = j;
							}
							if (buffer_1[j_offset] > max_1) {
								max_1 = buffer_1[j_offset];
								argMax_1 = j;
							}
							if (buffer_2[j_offset] > max_2) {
								max_2 = buffer_2[j_offset];
								argMax_2 = j;
							}
							if (buffer_3[j_offset] > max_3) {
								max_3 = buffer_3[j_offset];
								argMax_3 = j;
							}

							j_offset += inner_stride;
						}

						//tail
						for (Z j = loopCount4; j < loopCountEnd; j++) {
							if (buffer_3[j_offset] > max_3) {
								max_3 = buffer_3[j_offset];
								argMax_3 = j;
							}
							j_offset += inner_stride;
						}

						//merge
						argMax_1 += loopCount4;
						argMax_2 += 2 * loopCount4;
						argMax_3 += 3 * loopCount4;

						if (max_1 > max) {
							max = max_1;
							argMax = argMax_1;
						}
						if (max_2 > max) {
							max = max_2;
							argMax = argMax_2;
						}
						if (max_3 > max) {
							max = max_3;
							argMax = argMax_3;
						}


					}
					else {
						for (Z j = 0; j < loopCount; j++) {
							if (*buffer > max) {
								max = *buffer;
								argMax = j;
							}
							buffer += inner_stride;
						}
					}//<256
				}

#if 0			
				nd4j_printf("___%f +\n", max );
#endif
			}


			template<typename X, typename Z>
			FORCEINLINE void argMaxInnerReductionOutBlock4(const X*  buffer , const Nd4jLong outer_stride, const Nd4jLong& loopCount, const Nd4jLong& inner_stride,  Z *output, const Nd4jLong output_stride)
			{

				const X* buffer1 = buffer + outer_stride;
				const X* buffer2 = buffer1 + outer_stride;
				const X* buffer3 = buffer2 + outer_stride; 
				Z argMax = 0; 
				Z argMax1 = 0;
				Z argMax2 = 0;
				Z argMax3 = 0;
				X max = buffer[0];
				X max1 = buffer1[0];
				X max2 = buffer2[0];
				X max3 = buffer3[0];
				Nd4jLong offset = 0;
                PRAGMA_OMP_SIMD
				for (Z j = 0; j < loopCount; j++) {
							if (buffer[offset] > max) {
								max = buffer[offset];
								argMax = j;
							}
							if (buffer1[offset] > max1) {
								max1 = buffer1[offset];
								argMax1 = j;
							}
							if (buffer2[offset] > max2) {
								max2= buffer2[offset];
								argMax2 = j;
							}
							if (buffer3[offset] > max3) {
								max3 = buffer3[offset];
								argMax3 = j;
							} 
							offset += inner_stride; 
				}
				output[0] = argMax;
				output[output_stride] = argMax1;
				output[2*output_stride] = argMax2;
				output[3*output_stride] = argMax3;
#if 0			
				nd4j_printf("___%f__%f__%f__%f+\n",  max,max1,max2,max3);
#endif
			}


			template<typename X, typename Z, bool Last_Index_Faster=true>
			void argMaxCase1Scalar(const  int& second_rank, const Nd4jLong*& inner_bases, const Nd4jLong*& inner_strides, const  X* bufferX, Z* outputZ)
			{
				int maxThreads = sd::Environment::getInstance()->maxMasterThreads(); 
				std::unique_ptr<X[]> maxValues(new X[maxThreads]);
				std::unique_ptr<Z[]> maxIndices(new Z[maxThreads]);
				Nd4jLong inner_total;
				Nd4jLong inner_last=0;
				X* ptrMaxValues  = maxValues.get();
				Z* ptrMaxIndices = maxIndices.get();
				if (second_rank == 1) {
					inner_total=inner_bases[0];
				}else{
					inner_total = getLength<Last_Index_Faster>(inner_bases, second_rank, 1, inner_last);
			    }
				auto func = [ptrMaxValues, ptrMaxIndices, inner_last, second_rank, inner_bases, inner_strides, bufferX](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
#if 0					
					nd4j_printf("___%s_________%d+\n", __PRETTY_FUNCTION__, thread_id);
#endif
					const Nd4jLong inner_stride = Last_Index_Faster ? inner_strides[second_rank - 1] : inner_strides[0];
					
					Z argMax; X max;
					if (second_rank == 1) {
						const Nd4jLong loopTotal = stop - start;
						argMaxInnerReduction(&(bufferX[start*inner_stride]), loopTotal, inner_stride, max, argMax);
						ptrMaxIndices[thread_id] = argMax +start;
					}
					else {
						argMaxInnerReduction<X, Z, Last_Index_Faster>(second_rank, bufferX, inner_bases, inner_strides, start, stop, inner_last, inner_stride, max, argMax);
						ptrMaxIndices[thread_id] = argMax;
					}
					
					ptrMaxValues[thread_id] = max;
				};
#if 0 
				int Count = 0;
				func(0,0, inner_total, 1 );
#else
				int Count = samediff::Threads::parallel_tad(func, 0, inner_total, 1);
#endif
				int arg = 0;
				X max = ptrMaxValues[0];
#if 0
				nd4j_printf("%ld %f\n", ptrMaxIndices[0], ptrMaxValues[0]);
#endif
				for (int i = 1; i < Count; i++) {
#if 0
					nd4j_printf("%ld %f\n", ptrMaxIndices[i], ptrMaxValues[i]);
#endif
					if (ptrMaxValues[i] > max) {
						max = ptrMaxValues[i];
						arg = i;
					}
				};
#if 0
				nd4j_printf("---> %ld %f\n", ptrMaxIndices[arg], ptrMaxValues[arg]);
#endif
				*outputZ = ptrMaxIndices[arg];
			}



			template<typename X, typename Z,bool Last_Index_Faster=true>
			void argMaxCase2(const Nd4jLong* outer_bases, const Nd4jLong* outer_strides, const Nd4jLong output_stride, const  int& second_rank, const Nd4jLong*& inner_bases, const Nd4jLong*& inner_strides,const X* bufferX, Z* outputZ)
			{ 

				//total
				const Nd4jLong total         = outer_bases[0];
				const Nd4jLong outer_stride  = outer_strides[0];
				const Nd4jLong inner_stride  = true ? inner_strides[second_rank - 1] : inner_strides[0];
				auto func = [outer_stride, inner_stride, output_stride, second_rank, inner_bases, inner_strides, bufferX, outputZ](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {

					Nd4jLong loopTotal = stop - start;
					//lambda captures values as immutable , so either we should make our lambda mutable 
					// or just use another local value
					const X*  bufferPtr = &(bufferX[start * outer_stride]);
					Z *outputPtr = outputZ + start * output_stride;

					if (second_rank == 1) {

						const Nd4jLong inner_total = inner_bases[0];
						if (loopTotal >= 4 && (inner_stride > outer_stride || inner_total<256)) {
#if 0
							nd4j_printf("___%s_________%d %d+ looptotal %d innertotal %d\n", __PRETTY_FUNCTION__, thread_id, 0, loopTotal,inner_total);
#endif
							Nd4jLong loopTotal_K = loopTotal / 4;
							Nd4jLong loopTotal_Tail = loopTotal & 3;
							for (Nd4jLong i = 0; i < loopTotal_K; i++) {

								argMaxInnerReductionOutBlock4(bufferPtr, outer_stride, inner_total, inner_stride, outputPtr, output_stride);
								bufferPtr += 4 * outer_stride;
								outputPtr += 4 * output_stride;
							}
							for (Nd4jLong i = 0; i < loopTotal_Tail; i++) {
								Z argMax; X max;
								argMaxInnerReduction(bufferPtr, inner_total, inner_stride, max, argMax);
								*outputPtr = argMax;
								bufferPtr += outer_stride;
								outputPtr += output_stride;
							}
						}
						else {
#if 0
							nd4j_printf("___%s_________%d %d+\n", __PRETTY_FUNCTION__, thread_id, 1);
#endif
							//better do blocking inside inner reduction whenever possible
							for (Nd4jLong i = 0; i < loopTotal; i++) {
								Z argMax; X max;
								argMaxInnerReduction(bufferPtr, inner_total, inner_stride, max, argMax);
								*outputPtr = argMax;
								bufferPtr += outer_stride;
								outputPtr += output_stride;
							}
						}

					}
					else {
#if 0
						nd4j_printf("___%s_________%d %d+\n", __PRETTY_FUNCTION__, thread_id, 2);
#endif
						Nd4jLong inner_last;
						Nd4jLong inner_loop = getLength<true>(inner_bases, second_rank, 1, inner_last);

						for (Nd4jLong i = 0; i < loopTotal; i++) {
							Z argMax; X max;
							argMaxInnerReduction<X, Z, true>(second_rank, bufferPtr, inner_bases, inner_strides, 0, inner_loop, inner_last, inner_stride, max, argMax);
							*outputPtr = argMax;
							bufferPtr += outer_stride;
							outputPtr += output_stride;
						}
					}

				};
#if 0
				func(0, 0, total, 1);
#else
				//
				samediff::Threads::parallel_tad(func, 0, total, 1);
#endif
			}

			template<typename X, typename Z, bool Last_Index_Faster=true>
			void argMaxCase3(const int& first_rank, const Nd4jLong* outer_bases, const Nd4jLong* outer_strides, const Nd4jLong &output_stride, const  int& second_rank, const Nd4jLong*& inner_bases, const Nd4jLong*& inner_strides, X* bufferX, Z* outputZ)
			{
				
				//total
				Nd4jLong total = getLength<Last_Index_Faster>(outer_bases, first_rank);
                Nd4jLong inner_stride  = true ? inner_strides[second_rank - 1]  : inner_strides[0];

				auto func = [first_rank, outer_bases, outer_strides, output_stride, second_rank, inner_bases, inner_strides, inner_stride, bufferX, outputZ](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
#if 0
					nd4j_printf("___%s_________%d\n", __PRETTY_FUNCTION__, thread_id);
#endif
					Nd4jLong outer_coords[MAX_RANK] = {};
					Nd4jLong* ptr_coords = (Nd4jLong*)&outer_coords;
					if (Last_Index_Faster) {
						sd::index2coords_C(start, first_rank, outer_bases, ptr_coords);
					}
					else {
						sd::index2coords_F(start, first_rank, outer_bases, ptr_coords);
					}
					//offset
					size_t offset = sd::offset_from_coords(outer_strides, ptr_coords, first_rank);

					Nd4jLong loopTotal = stop - start;
					//lambda captures values as immutable , so either we should make it mutable or just 
					// use another local value
					Z *outputPtr = &(outputZ[start * output_stride]);

					if (second_rank == 1) {
						const Nd4jLong inner_total = inner_bases[0];
						for (Nd4jLong i = 0; i < loopTotal; i++) {
							Z argMax; X max;
							argMaxInnerReduction(&(bufferX[offset]), inner_total, inner_stride, max, argMax);
							outputPtr[i * output_stride] = argMax;
							offset = inc_coords<Last_Index_Faster>(outer_bases, outer_strides, ptr_coords, offset, first_rank);
						}

					}
					else {
						Nd4jLong inner_last;
						Nd4jLong inner_loop = getLength<true>(inner_bases, second_rank, 1, inner_last);

						for (Nd4jLong i = 0; i < loopTotal; i++) {
							Z argMax; X max;
							argMaxInnerReduction<X, Z, true>(second_rank, &(bufferX[offset]), inner_bases, inner_strides, 0, inner_loop, inner_last, inner_stride, max, argMax);
							outputPtr[i * output_stride] = argMax;
							offset = inc_coords<Last_Index_Faster>(outer_bases, outer_strides, ptr_coords, offset, first_rank);
						}
					}

				};

				//
				samediff::Threads::parallel_tad(func, 0, total, 1);
			}


			template<typename X, typename Z, bool Last_Index_Faster=true>
			void argMaxCase4(const int& first_rank, const Nd4jLong* outer_bases, const Nd4jLong* outer_strides, const Nd4jLong* output_strides, const  int& second_rank, const Nd4jLong*& inner_bases, const Nd4jLong*& inner_strides, X* bufferX, Z* outputZ)
			{
				Nd4jLong total = getLength<Last_Index_Faster>(outer_bases, first_rank);
				Nd4jLong inner_stride = true /*Last_Index_Faster*/? inner_strides[second_rank - 1] : inner_strides[0];

				auto func = [first_rank, outer_bases, outer_strides, output_strides, second_rank, inner_bases, inner_strides, inner_stride, bufferX, outputZ](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
#if 0
					
#endif
					Nd4jLong outer_coords[MAX_RANK] = {};

					Nd4jLong* ptr_coords = (Nd4jLong*)&outer_coords;
					if (Last_Index_Faster) {
						sd::index2coords_C(start, first_rank, outer_bases, ptr_coords);
					}
					else {
						sd::index2coords_F(start, first_rank, outer_bases, ptr_coords);
					}
					//offset
					auto offset = sd::offset_from_coords(outer_strides, output_strides, (const Nd4jLong *)ptr_coords,(const Nd4jLong) first_rank);

					//total 
					Nd4jLong loopTotal = stop - start;

					if (second_rank == 1) {
#if 0
						nd4j_printf("___%s_________+%d\n", __PRETTY_FUNCTION__, 0);
#endif
						const Nd4jLong inner_total = inner_bases[0];
						for (Nd4jLong i = 0; i < loopTotal; i++) {
							Z argMax; X max;
							argMaxInnerReduction(&(bufferX[offset.first]), inner_total, inner_stride, max, argMax);
							outputZ[offset.second] = argMax;
							offset = inc_coords<Last_Index_Faster>(outer_bases, outer_strides, output_strides, ptr_coords, offset, first_rank);
						}

					}
					else {
#if 0
						nd4j_printf("___%s_________+%d\n", __PRETTY_FUNCTION__, 1);
#endif
						Nd4jLong inner_last;
						Nd4jLong inner_loop = getLength<true>(inner_bases, second_rank, 1, inner_last);
						for (Nd4jLong i = 0; i < loopTotal; i++) {
							Z argMax; X max;
							argMaxInnerReduction<X, Z, true>(second_rank, &(bufferX[offset.first]), inner_bases, inner_strides, 0, inner_loop, inner_last, inner_stride, max, argMax);
							outputZ[offset.second] = argMax;
							offset = inc_coords<Last_Index_Faster>(outer_bases, outer_strides, output_strides, ptr_coords, offset, first_rank);
						}
					}

				};
			    
				//
				samediff::Threads::parallel_tad(func, 0, total, 1);
			}



			template<typename X, typename Z>
			void  argMax_(const NDArray& input, NDArray& output, const std::vector<int>& dimensions) {
				

				char input_order = input.ordering();
				bool try_squash_outer = (input_order == output.ordering()) && output.ews() != 0;
				Nd4jLong*  input_shapeInfo  = input.getShapeInfo();
				Nd4jLong* output_shapeInfo  = output.getShapeInfo();
				const Nd4jLong  rank            = input_shapeInfo[0];
				const Nd4jLong* input_bases     = &(input_shapeInfo[1]);
				const Nd4jLong* input_strides   = &(input_shapeInfo[rank + 1]);
				const Nd4jLong  output_rank     = output_shapeInfo[0];
				const Nd4jLong* output_strides  = &(output_shapeInfo[output_rank + 1]);
#if 0
				nd4j_printf("___%s_________+ input_rank %d output_rank %d \n", __PRETTY_FUNCTION__,rank,output_rank);
#endif
				Nd4jLong new_bases[MAX_RANK];
				Nd4jLong new_strides[MAX_RANK];
				int first_begin, first_end, second_begin, second_end;

				//rePartition into two parts based on the selection
				rePartition(input_order, dimensions, rank, input_bases, input_strides, new_bases, new_strides, first_begin, first_end, second_begin, second_end, try_squash_outer, input_order=='c');

				int first_rank = first_end - first_begin; //the first rank can be 0 for scalar cases
				int second_rank = second_end - second_begin;

				X* bufferX = input.bufferAsT<X>();
				Z* outputZ = output.bufferAsT<Z>();
				const Nd4jLong* outer_bases = &(new_bases[first_begin]);
				const Nd4jLong* outer_strides = &(new_strides[first_begin]);
				const Nd4jLong* inner_bases = &(new_bases[second_begin]);
				const Nd4jLong* inner_strides = &(new_strides[second_begin]);
				const Nd4jLong  output_stride = (input_order == 'c') ? output_strides[output_rank - 1] : output_strides[0];

				if (input_order == 'c') {
					if (first_rank == 0) {
						argMaxCase1Scalar<X,Z>(second_rank, inner_bases, inner_strides, bufferX, outputZ);

					}
					else if (/*try_squash_outer &&*/ first_rank == 1) {
						argMaxCase2( outer_bases, outer_strides, output_stride, second_rank, inner_bases, inner_strides, bufferX, outputZ);

					}
					else if (try_squash_outer && first_rank <= output_rank) {
						argMaxCase3<X, Z>(first_rank, outer_bases, outer_strides, output_stride, second_rank, inner_bases, inner_strides, bufferX, outputZ);
					}
					//no squashing was done
					else if (first_rank == output_rank) {
						argMaxCase4<X, Z>(first_rank, outer_bases, outer_strides, output_strides, second_rank, inner_bases, inner_strides, bufferX, outputZ);

					}

				}
				else {
					if (first_rank == 0 ) {
						if (second_rank == 1) {
							argMaxCase1Scalar<X, Z, false>(second_rank, inner_bases, inner_strides, bufferX, outputZ);
						}
						else {
							//we are obliged to find C order index
							argMaxCase1Scalar<X, Z, true>(second_rank, inner_bases, inner_strides, bufferX, outputZ);

						}
						
					}
					else if (/*try_squash_outer &&*/ first_rank == 1) {
						argMaxCase2<X, Z, false>( outer_bases, outer_strides, output_stride, second_rank, inner_bases, inner_strides, bufferX, outputZ);

					}
					else if (try_squash_outer && first_rank <= output_rank) {
						argMaxCase3<X, Z, false>(first_rank, outer_bases, outer_strides, output_stride, second_rank, inner_bases, inner_strides, bufferX, outputZ);
					}
					//no squashing was done
					else if (first_rank == output_rank) {
						argMaxCase4<X, Z, false>(first_rank, outer_bases, outer_strides, output_strides, second_rank, inner_bases, inner_strides, bufferX, outputZ);

					 }
				}
				

			}

			//////////////////////////////////////////////////////////////////////////
			void  argmax(const NDArray& input, NDArray& output, const std::vector<int>& dimensions) {

				BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), argMax_, (input, output, dimensions), FLOAT_TYPES, INDEXING_TYPES);
			}


			BUILD_DOUBLE_TEMPLATE(template void argMax_, (const NDArray& input, NDArray& output, const std::vector<int>& dimensions), FLOAT_TYPES, INDEXING_TYPES);
		}
	}
}
