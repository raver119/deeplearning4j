/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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
// @author Yurii Shyrma, created on 26.02.2018
//

#ifndef LIBND4J_ADDBIAS_H
#define LIBND4J_ADDBIAS_H

#include <ops/declarable/helpers/helpers.h>
#include <graph/Context.h>
#include <type_traits>
namespace nd4j    {
namespace ops     {
namespace helpers {


	void addBias(graph::Context& block, const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW);
   
    void addBias_Experimental(graph::Context& block, const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW
#if 1
		, const bool force_non_continuous
#endif			
	
	);
     
//<editor-fold desc="EXPERIMENTAL_COORDS_HELPERS">
#pragma region EXPERIMENTAL_COORDS_HELPERS
	//ODR RULE:  templates and inline functions
#if defined (_MSC_VER)
 #define likely(x)  (x)
 #define unlikely(x)  (x)
#define prefetch(x)  

#define prefetchw(x)  

#define prefetch_range_r(x ,len) 
#define prefetch_range_w(x ,len) 
#else
#define PREFETCH_STRIDE 64 
#define likely(x) __builtin_expect( (x), 1)	 
#define unlikely(x) __builtin_expect( (x), 0)	
#define prefetch(x) __builtin_prefetch(x,0,1)
#define prefetchw(x) __builtin_prefetch(x,1,1)
#define prefetchl(x) __builtin_prefetch(x,0,3)
#define prefetchwl(x) __builtin_prefetch(x,1,3)
	inline void prefetch_range(char* addr_r, char* addr_w, size_t len)
	{ 
		for (size_t i= 0; i < len; i += PREFETCH_STRIDE) {
			prefetch(&addr_r[i]); 
			prefetchw(&addr_w[i]);
		}
	}
 
inline void prefetch_range_rl(char* addr, size_t len)
 { 
			  char* cp;
		      char* end = addr + len;
		      for (cp = addr; cp < end; cp += PREFETCH_STRIDE)
			                 prefetchl(cp); 
 }
inline void prefetch_range_wl(char* addr, size_t len)
{
	char* cp;
	char* end = addr + len;
	for (cp = addr; cp < end; cp += PREFETCH_STRIDE)
		prefetchwl(cp);
}
#endif

    using pair_size_t = std::pair<size_t, size_t>;

	template<size_t Rank,size_t Index, bool Last_Index_Faster = true>
	constexpr size_t StridesOrderInd() {
		return Last_Index_Faster ? Rank - Index - 1 : Index;
	}

	template<size_t Rank, size_t Index, bool Last_Index_Faster = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 == Index), size_t>::type
		coord_add(const Nd4jLong* bases, const Nd4jLong* strides, Nd4jLong(&coords)[MAX_RANK], const Nd4jLong(&add_coords)[MAX_RANK], Nd4jLong carry, size_t offset) {

		constexpr size_t Ind = StridesOrderInd<Rank, Index, Last_Index_Faster>();
		//add_Ind is always C order
		constexpr size_t base_Ind = StridesOrderInd<Rank, Index, true>();
		Nd4jLong val = coords[Ind] + add_coords[base_Ind] + carry;
		coords[Ind] = val >= bases[Ind] ? val - bases[Ind] : val;
		offset += coords[Ind] * strides[Ind];
		return offset;
	}

	template<size_t Rank, size_t Index, bool Last_Index_Faster = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 != Index), size_t >::type
		coord_add(const Nd4jLong* bases, const Nd4jLong* strides, Nd4jLong(&coords)[MAX_RANK], const Nd4jLong(&add_coords)[MAX_RANK], Nd4jLong carry, size_t offset) {

		constexpr size_t Ind = StridesOrderInd<Rank, Index, Last_Index_Faster>();
		//add_Ind is always C order
		constexpr size_t base_Ind = StridesOrderInd<Rank, Index, true>();
		Nd4jLong val = coords[Ind] + add_coords[base_Ind] + carry;
		coords[Ind] = val >= bases[Ind] ? val - bases[Ind] : val;
		carry = val >= bases[Ind] ? 1 : 0;
		offset += coords[Ind] * strides[Ind];
		return coord_add<Rank, Index + 1, Last_Index_Faster>(bases, strides, coords, add_coords, carry, offset);
	}

	template<size_t Rank, size_t Index=0, bool Last_Index_Faster = true>
	FORCEINLINE size_t move_by_coords(const Nd4jLong* bases, const  Nd4jLong* strides, Nd4jLong(&coords)[MAX_RANK], const Nd4jLong(&add_coords)[MAX_RANK]) {

		return coord_add<Rank,Index,Last_Index_Faster>(bases, strides, coords, add_coords, 0, 0);
	}

	template<bool Last_Index_Faster=true>
	FORCEINLINE size_t move_by_coords(const Nd4jLong* bases, const Nd4jLong* strides, Nd4jLong(&coords)[MAX_RANK], const Nd4jLong(&add_coords)[MAX_RANK], size_t rank) {
		Nd4jLong carry = 0;
		Nd4jLong tmp, val = 0;
		size_t offset = 0; 
			for (int i = rank - 1; i >= 1; i--) {
				val = coords[i] + add_coords[i] + carry;
				if (val >= bases[i]) {
					carry = 1;
					tmp = val - bases[i];
					coords[i] = tmp;
					offset += tmp * strides[i];
				}
				else {
					carry = 0;
					coords[i] = val;
					offset += val * strides[i];
				}
			}

			val = coords[0] + add_coords[0] + carry;
			coords[0] = val >= bases[0] ? val - bases[0] : val;
			offset += val * strides[0]; 
		return offset;
	}

	template<>
	FORCEINLINE size_t move_by_coords<false>(const Nd4jLong* bases, const Nd4jLong* strides, Nd4jLong(&coords)[MAX_RANK], const Nd4jLong(&add_coords)[MAX_RANK], size_t rank) {
		Nd4jLong carry = 0;
		Nd4jLong tmp, val = 0;
		size_t offset = 0;
 
			const Nd4jLong* rev_add_cords = add_coords+rank-1;
			int i = 0;
			for (; i <rank-1 ; i++) {
				val = coords[i] + *rev_add_cords + carry;
				if (val >= bases[i]) {
					carry = 1;
					tmp = val - bases[i];
					coords[i] = tmp;
					offset += tmp * strides[i];
				}
				else {
					carry = 0;
					coords[i] = val;
					offset += val * strides[i];
				}
				--rev_add_cords;
			}

			val = coords[i] + *rev_add_cords + carry;
			coords[i] = val >= bases[i] ? val - bases[i] : val;
			offset += val * strides[i]; 
		return offset;
	}

	INLINEDEF void   index2coords_C(Nd4jLong index, const Nd4jLong rank, const Nd4jLong* bases, Nd4jLong* coords) {
		for (size_t i = rank-1; i > 0; --i) {
			coords[i] = index % bases[i];
			index /= bases[i];
		}
		coords[0] = index;      // last iteration 
	}

	INLINEDEF void   index2coords_F(Nd4jLong index, const Nd4jLong rank, const Nd4jLong *bases, Nd4jLong* coords) {

		for (size_t i = 0; i <rank-1; i++) {
			coords[i ] = index % bases[i];
			index /= bases[i];
		}
		coords[rank-1] = index;      // last iteration
	}

	FORCEINLINE size_t offset_from_coords(const Nd4jLong* strides, const Nd4jLong *coords,const  Nd4jLong &rank) {
		 
		size_t offset = 0; 
		size_t rank_4 = rank & -4;
		for (int i = 0; i <rank_4; i+=4) {
					offset = offset
						+ coords[i] * strides[i]  
						+ coords[i+1] * strides[i+1] 
					    + coords[i+2] * strides[i+2]
					    + coords[i+3] * strides[i+3];
		} 
		for (int i = rank_4; i < rank ; i ++) {
			offset += coords[i] * strides[i];
		}
		return offset;
	}

	FORCEINLINE bool check_continuity(const char order ,const Nd4jLong* bases, const Nd4jLong* x_strides, const Nd4jLong &rank) {
		bool continuous = true;
		Nd4jLong calc_stride = 1;
		if (order == 'c') {
			for (Nd4jLong i = rank - 1; i >= 0; i--) {
				bool is_eq = (calc_stride == x_strides[i]);
				continuous &= is_eq;
				if (!continuous) break;
				calc_stride = bases[i] * calc_stride;
			} 
		}
		else {
			for (Nd4jLong i = 0; i < rank; i++) {
				bool is_eq = (calc_stride == x_strides[i]);
				continuous &= is_eq;
				if (!continuous) break;
				calc_stride = bases[i] * calc_stride;
			}
		}
		return continuous;
	}


	FORCEINLINE pair_size_t offset_from_coords(const Nd4jLong* &x_strides, const Nd4jLong*& z_strides, const Nd4jLong* coords, const Nd4jLong &rank) {

		pair_size_t offset = { 0,0 };
		size_t rank_4 = rank & -4;
		for (int i = 0; i < rank_4; i += 4) {
			offset.first = offset.first
				+ coords[i] * x_strides[i]
				+ coords[i + 1] * x_strides[i + 1]
				+ coords[i + 2] * x_strides[i + 2]
				+ coords[i + 3] * x_strides[i + 3];
			offset.second = offset.second
				+ coords[i] * z_strides[i]
				+ coords[i + 1] * z_strides[i + 1]
				+ coords[i + 2] * z_strides[i + 2]
				+ coords[i + 3] * z_strides[i + 3];
		}
		for (int i = rank_4; i < rank; i++) {
			offset.first += coords[i] * x_strides[i];
			offset.second += coords[i] * z_strides[i];
		}
		return offset;
	}



	template<size_t Rank, size_t Index, bool Last_Index_Faster = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 == Index), size_t>::type
		coord_inc(const Nd4jLong* bases, const Nd4jLong* strides, Nd4jLong* coords, Nd4jLong carry, size_t last_offset,   size_t adjust_stride) {

		constexpr size_t Ind = StridesOrderInd<Rank, Index, Last_Index_Faster>();
		Nd4jLong val = coords[Ind] + carry;
		if (likely(val < bases[Ind])) {
			last_offset +=  strides[Ind] - adjust_stride;
			coords[Ind] = val;
			return last_offset;
		}
		//overflow case should not happen
		coords[Ind] = 0;
		last_offset = 0;// last_offset + strides[Ind] - adjust_stride;
		return last_offset;
	}

	template<size_t Rank, size_t Index, bool Last_Index_Faster = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 != Index), size_t >::type
		coord_inc(const Nd4jLong* bases, const Nd4jLong* strides, Nd4jLong* coords, Nd4jLong carry, size_t last_offset,size_t adjust_stride) {

		constexpr size_t Ind = StridesOrderInd<Rank, Index, Last_Index_Faster>();

		Nd4jLong val = coords[Ind] + carry;
		Nd4jLong tmp;
		if (likely(val < bases[Ind])) {
			last_offset = last_offset +  strides[Ind] - adjust_stride;
			coords[Ind] = val;

		}
		else {
			
			//lets adjust offset
			adjust_stride += coords[Ind] *strides[Ind] ;
			coords[Ind] = 0;
			last_offset = coord_inc<Rank, Index + 1, Last_Index_Faster>(bases, strides, coords, 1, last_offset, adjust_stride);
		}

		return last_offset;

	}

	template<size_t Rank, size_t Index=0, bool Last_Index_Faster = true>
	FORCEINLINE size_t inc_by_coords(const Nd4jLong* bases, const Nd4jLong* strides, Nd4jLong* coords,size_t last_offset) {

		return coord_inc<Rank, Index, Last_Index_Faster>(bases, strides, coords, 1, last_offset,0);
	}

	template<bool Last_Index_Faster = true>
	FORCEINLINE size_t inc_by_coords(const Nd4jLong* bases, const Nd4jLong* strides, Nd4jLong* coords, size_t last_offset, const size_t rank, const size_t skip = 0) {
		Nd4jLong carry = 1;
		Nd4jLong  val = 0;
		size_t adjust = 0;
		for (int i = rank -skip- 1; i >= 0; i--) {
			val = coords[i] + carry;
			if (likely(val < bases[i])) {
				coords[i] = val;
				last_offset += strides[i] - adjust;
				break;
			}
			else { 
				adjust += coords[i] * strides[i];
				coords[i] = 0;
			}
		}

		return last_offset;
	}

	template<>
	FORCEINLINE size_t inc_by_coords<false>(const Nd4jLong* bases, const Nd4jLong* strides, Nd4jLong* coords, size_t last_offset, const size_t rank, const size_t skip ) {
		Nd4jLong carry = 1;
		Nd4jLong  val = 0;
		size_t adjust = 0;

		for (int i = skip; i < rank; i++) {
			val = coords[i] + carry;
			if (likely(val < bases[i])) {
				coords[i] = val;
				last_offset += strides[i] - adjust; 
				break;
			}
			else { 
 
				adjust += coords[i] * strides[i];
				coords[i] = 0;
			}
		}
		return last_offset;
	}


	FORCEINLINE void get_adjusts_for_inc(const Nd4jLong* strides, const Nd4jLong* bases, Nd4jLong* adjusts, size_t rank) {

		size_t offset = 0;
		size_t rank_4 = rank & -4;
		for (int i = 0; i < rank_4; i += 4) {
			adjusts[i] = (bases[i] - 1) * strides[i];
			adjusts[i+1] = (bases[i+2] - 1) * strides[i+2];
			adjusts[i+2] = (bases[i+3] - 1) * strides[i+3];
			adjusts[i+3] = (bases[i+4] - 1) * strides[i+4];
		}
		for (int i = rank_4; i < rank; i++) {
			adjusts[i] = (bases[i] - 1) * strides[i];
		}
		return  ;
	}



	template<bool Last_Index_Faster = true>
	FORCEINLINE size_t inc_by_coords2(const Nd4jLong* bases, const Nd4jLong* strides, const Nd4jLong* adjusts, Nd4jLong* coords, size_t last_offset, const size_t rank, const size_t skip=0 ) {
		Nd4jLong carry = 1;
		Nd4jLong  val = 0;
		size_t adjust = 0;
		for (int i = rank -skip- 1; i >= 0; i--) {
			val = coords[i] + carry;
			if (likely(val < bases[i])) {
				coords[i] = val;
				last_offset += strides[i] - adjust;
				break;
			}
			else { 
				adjust += adjusts[i];
				coords[i] = 0;
			}
		}

		return last_offset;
	}

	template<>
	FORCEINLINE size_t inc_by_coords2<false>(const Nd4jLong* bases, const Nd4jLong* strides, const Nd4jLong* adjusts, Nd4jLong* coords, size_t last_offset, const size_t rank, const size_t skip  ) {
		Nd4jLong carry = 1;
		Nd4jLong  val = 0;
		size_t adjust = 0;

		for (int i = skip; i < rank; i++) {
			val = coords[i] + carry;
			if (likely(val < bases[i])) {
				coords[i] = val;
				last_offset += strides[i] - adjust;
				break;
			}
			else { 
				adjust += adjusts[i];
				coords[i] = 0;
			}
		}
		return last_offset;
	}




	template<size_t Rank, size_t Index, bool Last_Index_Faster = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 == Index), pair_size_t>::type
		coord_inc_zip(const Nd4jLong* bases, const  Nd4jLong* x_strides, const  Nd4jLong* z_strides, Nd4jLong* coords, Nd4jLong carry,
			pair_size_t last_offset, size_t adjust_stride_x, size_t adjust_stride_z) {

		constexpr size_t Ind = StridesOrderInd<Rank, Index, Last_Index_Faster>();
		Nd4jLong val = coords[Ind] + carry;
		if (likely(val < bases[Ind])) {
			last_offset.first += x_strides[Ind] - adjust_stride_x;
			last_offset.second += z_strides[Ind] - adjust_stride_z;
			coords[Ind] = val;
			return last_offset;
		}
		//overflow case should not happen
		coords[Ind] = 0;
		last_offset = {};// last_offset + x_strides[Ind] - adjust_stride;

		return last_offset;
	}

	template<size_t Rank, size_t Index, bool Last_Index_Faster = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 != Index), pair_size_t >::type
		coord_inc_zip(const Nd4jLong* bases, const  Nd4jLong* x_strides, const  Nd4jLong* z_strides, Nd4jLong* coords, Nd4jLong carry,
			pair_size_t last_offset, size_t adjust_stride_x, size_t adjust_stride_z) {

		constexpr size_t Ind = StridesOrderInd<Rank, Index, Last_Index_Faster>();

		Nd4jLong val = coords[Ind] + carry;
		Nd4jLong tmp;
		if (likely(val < bases[Ind])) {
			last_offset.first += x_strides[Ind] - adjust_stride_x;
			last_offset.second += z_strides[Ind] - adjust_stride_z;
			coords[Ind] = val;
			return last_offset;
		}
		//lets adjust offset
		adjust_stride_x += coords[Ind] * x_strides[Ind];
		adjust_stride_z += coords[Ind] * z_strides[Ind];
		coords[Ind] = 0;
		return coord_inc_zip<Rank, Index + 1, Last_Index_Faster>(bases, x_strides, z_strides, coords, 1, last_offset, adjust_stride_x, adjust_stride_z);


	}

	template<size_t Rank, size_t Index =0, bool Last_Index_Faster = true>
	FORCEINLINE pair_size_t inc_by_coords_zip(const Nd4jLong* bases,const Nd4jLong* x_strides,const Nd4jLong* z_strides, Nd4jLong* coords, pair_size_t last_offset) {

		return coord_inc_zip<Rank, Index, Last_Index_Faster>(bases, x_strides, z_strides, coords, 1, last_offset, 0, 0);
	}

	template<bool Last_Index_Faster = true>
	FORCEINLINE pair_size_t inc_by_coords_zip(const Nd4jLong* bases, const Nd4jLong* x_strides, const  Nd4jLong* z_strides, Nd4jLong* coords, pair_size_t last_offset, const size_t rank , const size_t skip =0) {

		Nd4jLong carry = 1;
		Nd4jLong  val = 0;
		size_t adjust_x = 0;
		size_t adjust_z = 0;
		for (int i = rank - skip- 1; i >= 0; i--) {
			val = coords[i] + carry;
			if (likely(val < bases[i])) {
				coords[i] = val;
				last_offset.first += x_strides[i] - adjust_x;
				last_offset.second += z_strides[i] - adjust_z;
				break;
			}
			else {
				adjust_x += coords[i] * x_strides[i];
				adjust_z += coords[i] * z_strides[i];
				coords[i] = 0;
			}
		}

		return last_offset;
	}

	template<>
	FORCEINLINE pair_size_t inc_by_coords_zip<false>(const Nd4jLong* bases, const Nd4jLong* x_strides,const  Nd4jLong* z_strides, Nd4jLong* coords, pair_size_t last_offset, const size_t rank,  const size_t skip  ) {
		Nd4jLong carry = 1;
		Nd4jLong  val = 0;
		size_t adjust_x = 0;
		size_t adjust_z = 0;

		for (int i = skip; i < rank; i++) {
			val = coords[i] + carry;
			if (likely(val < bases[i])) {
				coords[i] = val;
				
				last_offset.first += x_strides[i] - adjust_x;
				last_offset.second += z_strides[i] - adjust_z;
				break;
			}
			else { 
				adjust_x += coords[i] * x_strides[i];
				adjust_z += coords[i] * z_strides[i];
				coords[i] = 0;
			}
		}
		return last_offset;
	}

#pragma endregion
//</editor-fold>
}
}
}


#endif // LIBND4J_ADDBIAS_H
