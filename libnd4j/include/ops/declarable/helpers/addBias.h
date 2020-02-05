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
     
#if defined(__GNUC__) 
#define align32 __attribute__((aligned(32)))
#elif defined(_MSC_VER)
#define align32 __declspec(align(32))
#else
#define align32 
#endif 

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
 
}
}
}


#endif // LIBND4J_ADDBIAS_H
