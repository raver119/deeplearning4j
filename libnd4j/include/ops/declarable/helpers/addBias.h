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
   
    void addBias_Experimental(graph::Context& block, const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW, const bool check_strides);
     
//<editor-fold desc="EXPERIMENTAL_COORDS_HELPERS">
#pragma region EXPERIMENTAL_COORDS_HELPERS
	//ODR RULE:  templates and inline functions
	 
	template<size_t Rank,size_t Index, bool C_Order = true>
	constexpr size_t StridesOrderInd() {
		return C_Order ? Rank - Index - 1 : Index;
	}

	template<size_t Rank, size_t Index, bool C_Order = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 == Index), size_t>::type
		coord_add(Nd4jLong* bases, Nd4jLong* strides, Nd4jLong(&coords)[MAX_RANK], const Nd4jLong(&add_coords)[MAX_RANK], Nd4jLong carry, size_t offset) {

		constexpr size_t Ind = StridesOrderInd<Rank, Index, C_Order>();
		//add_Ind is always C order
		constexpr size_t base_Ind = StridesOrderInd<Rank, Index, true>();
		Nd4jLong val = coords[Ind] + add_coords[base_Ind] + carry;
		coords[Ind] = val >= bases[Ind] ? val - bases[Ind] : val;
		offset += coords[Ind] * strides[Ind];
		return offset;
	}

	template<size_t Rank, size_t Index, bool C_Order = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 != Index), size_t >::type
		coord_add(Nd4jLong* bases, Nd4jLong* strides, Nd4jLong(&coords)[MAX_RANK], const Nd4jLong(&add_coords)[MAX_RANK], Nd4jLong carry, size_t offset) {

		constexpr size_t Ind = StridesOrderInd<Rank, Index, C_Order>();
		//add_Ind is always C order
		constexpr size_t base_Ind = StridesOrderInd<Rank, Index, true>();
		Nd4jLong val = coords[Ind] + add_coords[base_Ind] + carry;
		coords[Ind] = val >= bases[Ind] ? val - bases[Ind] : val;
		carry = val >= bases[Ind] ? 1 : 0;
		offset += coords[Ind] * strides[Ind];
		return coord_add<Rank, Index + 1, C_Order>(bases, strides, coords, add_coords, carry, offset);
	}

	template<size_t Rank, bool C_Order = true> 
	FORCEINLINE size_t move_by_coords(Nd4jLong* bases, Nd4jLong* strides, Nd4jLong(&coords)[MAX_RANK], const Nd4jLong(&add_coords)[MAX_RANK]) {

		return coord_add<Rank,0,C_Order>(bases, strides, coords, add_coords, 0, 0);
	}

	template<bool C_Order=true>
	FORCEINLINE size_t move_by_coords(Nd4jLong* bases, Nd4jLong* strides, Nd4jLong(&coords)[MAX_RANK], const Nd4jLong(&add_coords)[MAX_RANK], size_t rank) {
		Nd4jLong carry = 0;
		Nd4jLong tmp, val = 0;
		size_t offset = 0;
		if (rank > 0) {
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
		}
		return offset;
	}

	template<>
	FORCEINLINE size_t move_by_coords<false>(Nd4jLong* bases, Nd4jLong* strides, Nd4jLong(&coords)[MAX_RANK], const Nd4jLong(&add_coords)[MAX_RANK], size_t rank) {
		Nd4jLong carry = 0;
		Nd4jLong tmp, val = 0;
		size_t offset = 0;
		
		if (rank > 0) {
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
		}
		return offset;
	}


#pragma endregion
//</editor-fold>
}
}
}


#endif // LIBND4J_ADDBIAS_H
