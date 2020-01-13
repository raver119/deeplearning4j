/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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
 // Created by raver119 on 20.11.17.
 //

#include "testlayers.h"
#include <Graph.h>
#include <chrono>
#include <Node.h>
#include <ops/declarable/CustomOperations.h>
#include <graph/profiling/GraphProfilingHelper.h>
#include <type_conversions.h>
#include <helpers/threshold.h>
#include <helpers/MmulHelper.h>
#include <ops/ops.h>
#include <OmpLaunchHelper.h>
#include <GradCheck.h>
#include <ops/declarable/helpers/im2col.h>
#include <Loops.h>
#include <RandomLauncher.h>
#include <ops/declarable/helpers/convolutions.h>

#include <ops/declarable/helpers/addBias.h>
#include <helpers/BenchmarkHelper.h>
#include <ops/declarable/helpers/scatter.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <array>
#include <performance/benchmarking/FullBenchmarkSuit.h>
#include <performance/benchmarking/LightBenchmarkSuit.h>
#include <iostream>
#include <ops/declarable/helpers/legacy_helpers.h>
#include <algorithm>
#include <numeric> 
#include <random>
using namespace nd4j;
using namespace nd4j::graph;

class PlaygroundTests : public testing::Test {
public:
	int numIterations = 100;
	int poolSize = 10;

	PlaygroundTests() {
		printf("\n");
		fflush(stdout);
	}
};
#define BENCH_RELEASE 1
#if defined(BENCH_RELEASE)

int bnch_cases[][4] = {
	{40, 2, 4, 2056},
	{2056, 2, 4, 19 },
	{16, 8, 16, 2056},
	{2056, 8, 16, 15 },
    {32, 7, 7, 96},
    {32, 32, 32, 16},
    {32, 600, 600, 3},
    {32, 112, 112, 32},
    {32, 7, 7, 1024},
	 {2,3,1,32}
};

constexpr int coords_k =   6;
#else
int bnch_cases[][4] = {
	{7,3,5,9},
	 {2,3,1,32},
     {32,32,2,2}
};

constexpr int coords_k = 0;

#endif

#define BENCH_TEST_F_ORDER 1
#define BENCH_TEST_F_ORDER_NC 1
#define BENCH_TEST_C_ORDER 1
#define BENCH_TEST_C_ORDER_NC 1
#define BENCH_TEST_MIXED_ORDERS 1
//#define BENCH_RELEASE 1
#if defined(BENCH_RELEASE)
int outter_loops =    10;
int inner_loops =   10;// 100;
#else
int outter_loops = 1;//  10;
int inner_loops = 1;// 10;// 100;
#endif
#if defined (_MSC_VER)
#define use(x)
#define escape(x)
#define clobber()
#else
template<typename T>
void use(T&& t) {
	__asm__ __volatile__("" :: "g" (t));
}

void escape(void* p) {
	__asm__ __volatile__("" : "+r" (p) :: "memory");
}

void clobber() {
	__asm__ __volatile__("" : : : "memory");
}
#endif
//#define bench_coords 1
template<typename Op,typename... Args>
void time_it(Op op,Nd4jLong totalFlops, Args&&... args) {
	std::vector<double> values;

	size_t n_1 = outter_loops > 1 ? outter_loops - 1 : 1;
	for (int e = 0; e < outter_loops; e++) {
		auto timeStart = std::chrono::system_clock::now();

		for (int i = 0; i < inner_loops; i++)
			op(std::forward<Args>(args)...);

		auto timeEnd = std::chrono::system_clock::now();
		auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
		values.emplace_back((double)elapsed_time / double(inner_loops));
	}

	std::sort(values.begin(), values.end());
	auto sum = std::accumulate(std::begin(values), std::end(values), 0.0);
	auto avg = (double)sum / outter_loops;
	auto sum_sq = std::accumulate(std::begin(values), std::end(values), 0.0, [&avg](int sumsq, int x) { return sumsq + (x - avg) * (x - avg); });

	if (totalFlops > 0) {
		nd4j_printf("Median: %f us\tAvg: %f (sd: %f)\tFlops: %f Mflops\n",
			values[values.size() / 2], avg, sqrt(sum_sq / n_1),
			(double)(totalFlops) / avg);
	} 
	else {
		nd4j_printf("Median: %f us\tAvg: %f (sd: %f)\n",
			values[values.size() / 2], avg, sqrt(sum_sq / n_1));
	}
}

template<typename T>
void fill_random(nd4j::NDArray& arr) {
	Nd4jLong coords[MAX_RANK] = {};
	Nd4jLong add_coords[MAX_RANK] = {}; //zero
	std::random_device rd;
	std::mt19937 gen(rd());
	//for floats
	std::uniform_real_distribution<T> dis((T)0.0,(T) 2.0);
	T* x = arr.bufferAsT<T>(); 
	Nd4jLong* shapeInfo = arr.getShapeInfo();  
	Nd4jLong* strides = arr.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]); 

	add_coords[rank - 1] = 1; 
	size_t t = 1;
	for (size_t i = 0; i < rank; i++) {
		t *= bases[i];
	}
	size_t offset = 0;
	for (size_t i = 0; i < t; i++) {
		x[offset] = dis(gen);
		offset = nd4j::ops::helpers::move_by_coords(bases, strides, coords, add_coords, rank);
	}
}

 
 
void check_correctness_with_experimental( Context& ctx, const NDArray& x, const NDArray& y, NDArray& z,bool isNCHW) {

	auto expected = z.ulike();
	nd4j::ops::helpers::addBias_Experimental(ctx, x, y, expected, isNCHW, true);
	ASSERT_TRUE(expected.equalsTo(z));

}

void check_correctness_with_base(Context& ctx, const NDArray& x, const NDArray& y, NDArray& z, bool isNCHW) {

	auto expected = z.ulike();
	nd4j::ops::helpers::addBias(ctx, x, y, expected, isNCHW);
	ASSERT_TRUE(expected.equalsTo(z));

}
#if defined(BENCH_TEST_C_ORDER)
TEST_F(PlaygroundTests, test_bias_base) {
	 

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('c', { bnch_cases[k][3] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias, total_flops, ctx, x, y, z, false);

		//check for correctness with experimental to see if they are equal
		check_correctness_with_experimental(ctx, x, y, z, false);
	};
}

TEST_F(PlaygroundTests, test_bias_experimental_coords_strided) {

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('c', { bnch_cases[k][3] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, false, true);
 
	};
	 
}

TEST_F(PlaygroundTests, test_bias_experimental_continuous) {

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('c', { bnch_cases[k][3] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, false, false);
		 
	    
		check_correctness_with_experimental(ctx, x, y, z, false);
	};

}


TEST_F(PlaygroundTests, test_bias_experimental_continuous2) {

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<double>('c', { bnch_cases[k][3] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<double>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, false, false);

		
		check_correctness_with_base(ctx, x, y, z, false);
	};

}

#endif
#if defined(BENCH_TEST_C_ORDER_NC)
TEST_F(PlaygroundTests, test_bias_NC_base) {


	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('c', { bnch_cases[k][1] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias, total_flops, ctx, x, y, z, true);

	};
}

TEST_F(PlaygroundTests, test_bias_NC_experimental_coords_strided) {

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('c', { bnch_cases[k][1] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, true, true);
		//check
		check_correctness_with_base(ctx, x, y, z, true);

	};

}

TEST_F(PlaygroundTests, test_bias_NC_experimental_continuous) {

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('c', { bnch_cases[k][1] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, true, false);

		//check
		check_correctness_with_base(ctx, x, y, z, true);
	};

}


TEST_F(PlaygroundTests, test_bias_NC_experimental_continuous2) {

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<double>('c', { bnch_cases[k][1] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<double>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, true, false);

		//check
		check_correctness_with_base(ctx, x, y, z, true);
	};

}

#endif

#if defined(BENCH_TEST_F_ORDER)
TEST_F(PlaygroundTests, test_bias_Fortran_base) {


	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('f', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('f', { bnch_cases[k][3] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias, total_flops, ctx, x, y, z, false);

	};
}

TEST_F(PlaygroundTests, test_bias_Fortran_experimental_coords_strided) {

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('f', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('f', { bnch_cases[k][3] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, false, true);

		check_correctness_with_base(ctx, x, y, z, false);

	};



}

TEST_F(PlaygroundTests, test_bias_Fortran_experimental_continuous) {

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('f', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('f', { bnch_cases[k][3] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, false, false);


		check_correctness_with_base(ctx, x, y, z, false);
	};

}


TEST_F(PlaygroundTests, test_bias_Fortran_experimental_continuous2) {

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('f', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<double>('f', { bnch_cases[k][3] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<double>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, false, false);


		check_correctness_with_base(ctx, x, y, z, false);
	};

}

#endif

#if defined(BENCH_TEST_F_ORDER_NC)
TEST_F(PlaygroundTests, test_bias_Fortran_NC_base) {


	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('f', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('f', { bnch_cases[k][1] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias, total_flops, ctx, x, y, z, true);

	};
}

TEST_F(PlaygroundTests, test_bias_Fortran_NC_experimental_coords_strided) {

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('f', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('f', { bnch_cases[k][1] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, true, true);
		//check
		check_correctness_with_base(ctx, x, y, z, true);

	};

}

TEST_F(PlaygroundTests, test_bias_Fortran_NC_experimental_continuous) {

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('f', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('f', { bnch_cases[k][1] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, true, false);

		//check
		check_correctness_with_base(ctx, x, y, z, true);
	};

}


TEST_F(PlaygroundTests, test_bias_Fortran_NC_experimental_continuous2) {

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('f', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<double>('f', { bnch_cases[k][1] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<double>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, true, false);

		//check
		check_correctness_with_base(ctx, x, y, z, true);
	};

}

#endif

#if defined(BENCH_TEST_MIXED_ORDERS)
TEST_F(PlaygroundTests, test_bias_base_different_order_inF_out_C) {


	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('f', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('c', { bnch_cases[k][3] });
		auto z = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });;

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias, total_flops, ctx, x, y, z, false);

		//check for correctness with base to see if they are equal
		check_correctness_with_base(ctx, x, y, z, false);
	};
}


TEST_F(PlaygroundTests, test_bias_base_different_order_inC_outF) {


	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('c', { bnch_cases[k][3] });
		auto z = NDArrayFactory::create<float>('f', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });;

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias, total_flops, ctx, x, y, z, false);

		//check for correctness with base to see if they are equal
		//check_correctness_with_base(ctx, x, y, z, false);
	};
}

TEST_F(PlaygroundTests, test_bias_different_order_NC_inF_out_C) {


	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('f', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('c', { bnch_cases[k][1] });
		auto z = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });;
		
		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, true,false);

		//check for correctness with base to see if they are equal
		check_correctness_with_base(ctx, x, y, z, true);
	};
}

TEST_F(PlaygroundTests, test_bias_different_order_NC_inC_outF) {


	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('c', { bnch_cases[k][1] });
		auto z = NDArrayFactory::create<float>('f', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });;

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, true, false);

		//check for correctness with base to see if they are equal
		check_correctness_with_base(ctx, x, y, z, true);
	};
}

#endif

#if defined(bench_coords) 

TEST_F(PlaygroundTests, test_coord1) {
	outter_loops = 4;
	inner_loops = 4;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK];
	Nd4jLong add_coords[MAX_RANK]; 
	Nd4jLong* shapeInfo = x.getShapeInfo();
	shape::index2coords(0, shapeInfo, coords);
	shape::index2coords(inc, shapeInfo, add_coords);
	 
	auto ops = [&]() {
		for (size_t i = 0; i < t; i += inc) {
			shape::index2coords(i, shapeInfo, coords);
			//add to prevent code elimination
			use(coords); 
		}
	};
	time_it(ops, 0 );
 }


TEST_F(PlaygroundTests, test_coord2) {
	outter_loops = 4;
	inner_loops = 4;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK];
	Nd4jLong add_coords[MAX_RANK];
	Nd4jLong* shapeInfo = x.getShapeInfo();
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	shape::index2coords(0, shapeInfo, coords);
	shape::index2coords(inc, shapeInfo, add_coords); 
	escape(&rank);
	escape(coords);
	escape(bases); 
	escape(add_coords);
	escape(strides);
	auto ops = [&]() { 
			for (size_t i = 0; i < t; i += inc) {
				nd4j::ops::helpers::move_by_coords(bases, strides, coords, add_coords, rank);
				use(coords); 
				use(i);
		      }
	};
	time_it(ops, 0);
}


TEST_F(PlaygroundTests, test_coord3) {
	outter_loops = 4;
	inner_loops = 4;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK];
	Nd4jLong add_coords[MAX_RANK];
	Nd4jLong* shapeInfo = x.getShapeInfo();
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	shape::index2coords(0, shapeInfo, coords);
	shape::index2coords(inc, shapeInfo, add_coords); 
	escape(&rank);
	escape(coords);
	escape(bases); 
	escape(add_coords);
	escape(strides);
	auto ops = [&]() {
		for (size_t i = 0; i < t; i += inc) {
				nd4j::ops::helpers::move_by_coords<4>(bases, strides,  coords, add_coords);
				use(coords); 
				use(i);
			} 
	};
	time_it(ops, 0);
}

TEST_F(PlaygroundTests, test_coord4_offset) {
	outter_loops = 4;
	inner_loops = 4;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK];
	Nd4jLong add_coords[MAX_RANK];
	Nd4jLong* shapeInfo = x.getShapeInfo();
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	shape::index2coords(0, shapeInfo, coords);
	shape::index2coords(inc, shapeInfo, add_coords); 
	size_t offset = 0;
	escape(&rank);
	escape(coords);
	escape(bases);
	escape(&offset);
	escape(add_coords);
	escape(strides);
	auto ops = [&]() {
		for (size_t i = 0; i < t; i += inc) {
				offset = nd4j::ops::helpers::move_by_coords(bases, strides, coords, add_coords,rank-1);
				use(coords); 
				use(offset);
				use(i);
			} 
	};
	time_it(ops, 0);
}

TEST_F(PlaygroundTests, test_coord5_offset) {
	outter_loops = 4;
	inner_loops = 4;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK];
	Nd4jLong add_coords[MAX_RANK];
	Nd4jLong* shapeInfo = x.getShapeInfo();
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	shape::index2coords(0, shapeInfo, coords);
	shape::index2coords(inc, shapeInfo, add_coords);
	size_t offset = 0;
	escape(&rank);
	escape(coords);
	escape(bases);
	escape(&offset);
	escape(add_coords);
	escape(strides);
	auto ops = [&]() {
		for (size_t i = 0; i < t; i += inc) {
			offset = nd4j::ops::helpers::move_by_coords<3>(bases, strides, coords, add_coords);
			use(coords); 
			use(offset);
			use(i);
		}
	};
	time_it(ops, 0);
}


TEST_F(PlaygroundTests, test_coord6_loop_offset) {
	outter_loops = 4;
	inner_loops = 40;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK];
	Nd4jLong add_coords[MAX_RANK];
	Nd4jLong* shapeInfo = x.getShapeInfo();
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	shape::index2coords(0, shapeInfo, coords);
	shape::index2coords(inc, shapeInfo, add_coords); 
	size_t offset = 0;
	escape(&rank);
	escape(coords); 
	auto ops = [&]() {
		size_t zStrideB = strides[0];
		size_t zStrideH = strides[1];
		size_t zStrideW = strides[2];
		escape(&zStrideB);
		escape(&zStrideH);
		escape(&zStrideW);
		for (uint b = 0; b < bnch_cases[k][0]; b+=inc)
			for (uint h = 0; h < bnch_cases[k][1]; h++)
				for (uint w = 0; w < bnch_cases[k][2]; w++) {
						offset = b * zStrideB + h * zStrideH + w * zStrideW  ;
						use(offset);
						use(h);
						use(w);
						use(b);
					}
	};
	time_it(ops, 0);
}

 
TEST_F(PlaygroundTests, test_coord7_offset) {
	outter_loops = 4;
	inner_loops = 40;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK]; 
	Nd4jLong* shapeInfo = x.getShapeInfo();
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	shape::index2coords(0, shapeInfo, coords);  
	size_t offset = 0; 
	escape(&rank);
	escape(coords);
	escape(bases);
	escape(&offset); 
	escape(strides);
	auto ops = [&]() {
		for (size_t i = 0; i < t; i+=inc) {
			 
			offset = nd4j::ops::helpers::inc_by_coords<3>(bases, strides, coords, offset); 
			use(coords); 
			use(offset);
			use(i);
		}
	};
	time_it(ops, 0);
}


TEST_F(PlaygroundTests, test_coord8_offset) {
	outter_loops = 4;
	inner_loops = 40;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK]; 
	Nd4jLong* shapeInfo = x.getShapeInfo();
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	shape::index2coords(0, shapeInfo, coords);  
	size_t offset = 0; 
	escape(&rank);
	escape(coords);
	escape(bases);
	escape(&offset);
	escape(strides);
	auto ops = [&]() {
		for (size_t i = 0; i < t; i+=inc) {

			offset = nd4j::ops::helpers::inc_by_coords(bases, strides, coords, offset,rank-1);
			use(coords); 
			use(offset);  
			use(i);
		}
	};
	time_it(ops, 0);
}


TEST_F(PlaygroundTests, test_coord9_offset) {
	outter_loops = 4;
	inner_loops = 40;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK];
	Nd4jLong adjusts[MAX_RANK];
	Nd4jLong* shapeInfo = x.getShapeInfo();
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	shape::index2coords(0, shapeInfo, coords); 
	nd4j::ops::helpers::get_adjusts_for_inc(strides, bases, adjusts, rank - 1);
	  size_t offset = 0; 
	  escape(&rank);
	  escape(coords);
	  escape(bases);
	  escape(&offset);
	  escape(adjusts);
	  escape(strides);
	auto ops = [&]() {
		for (size_t i = 0; i < t; i += inc) {

			offset = nd4j::ops::helpers::inc_by_coords2(bases, strides, adjusts, coords, offset, rank - 1);
			use(coords);
			use(offset);  
			use(i);
		}
	};
	time_it(ops, 0);
}

TEST_F(PlaygroundTests, test_coord_correctness) {
	constexpr int k = 0;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = 1;// bnch_cases[k][3];


	Nd4jLong coords[MAX_RANK];
	Nd4jLong coords2[MAX_RANK];
	Nd4jLong coords3[MAX_RANK];
	Nd4jLong coords4[MAX_RANK] = {};
	Nd4jLong coords5[MAX_RANK] = {};
	Nd4jLong add_coords[MAX_RANK]; 
	Nd4jLong* shapeInfo = x.getShapeInfo();
	shape::index2coords(0, shapeInfo, coords2);
	shape::index2coords(0, shapeInfo, coords3);
	shape::index2coords(inc, shapeInfo, add_coords);
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	size_t offset = 0;
	size_t offset_uni = 0;
	for (size_t i = 0; i < t; i += inc) {

		shape::index2coords(i, shapeInfo, coords);

		for (int j = 0; j < rank; j++) {
			ASSERT_TRUE(coords2[j] == coords3[j]);
			ASSERT_TRUE(coords[j] == coords2[j]);
		}
		//increment should be last
		nd4j::ops::helpers::move_by_coords(bases,strides, coords2, add_coords, rank);
		nd4j::ops::helpers::move_by_coords<4>(bases,strides, coords3, add_coords);
		ASSERT_EQ(offset, i);
		offset = nd4j::ops::helpers::inc_by_coords<4>(bases, strides, coords4, offset);
		ASSERT_EQ(offset_uni, i);
		offset_uni = nd4j::ops::helpers::inc_by_coords(bases, strides, coords5, offset_uni, rank);
	}



}

#endif

/*
TEST_F(PlaygroundTests, test_s_1) {
	constexpr int ib = 2;
	constexpr int ih = 3;
	constexpr int iw = 5;
	constexpr int ic = 4;
	auto x = NDArrayFactory::create<float>('c', { ib, ih, iw, ic });

	Nd4jLong coords[4];

	for (int index = 1; index < ib * ih * iw * ic; index++) {

		shape::index2coords(index, x.getShapeInfo(), coords);
		 nd4j_printf("%lld:  %lld  %lld  %lld  %lld  \n",index, coords[0],coords[1],coords[2],coords[3]);
	}

}
**/
/*
TEST_F(PlaygroundTests, test_s_1) {
	auto x0 = NDArrayFactory::create<float>('c', {32, 7, 7, 176});
	auto x1 = x0.ulike();
	auto x2 = x0.ulike();
	auto x3 = x0.ulike();
	auto x4 = x0.ulike();
	auto x5 = x0.ulike();

	auto y = NDArrayFactory::create<int >(3);
	auto z = NDArrayFactory::create<float>('c', {32, 7, 7, 1056});

	Context ctx(1);
	ctx.setInputArray(0, &x0);
	ctx.setInputArray(1, &x1);
	ctx.setInputArray(2, &x2);
	ctx.setInputArray(3, &x3);
	ctx.setInputArray(4, &x4);
	ctx.setInputArray(5, &x5);

	ctx.setInputArray(6, &y);
	ctx.setOutputArray(0, &z);
	ctx.setBArguments({true});

	std::vector<Nd4jLong> values;

	nd4j::ops::concat op;
	op.execute(&ctx);

	for (int e = 0; e < 1000; e++) {
		auto timeStart = std::chrono::system_clock::now();

		op.execute(&ctx);

		auto timeEnd = std::chrono::system_clock::now();
		auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
		values.emplace_back(outerTime);
	}


	std::sort(values.begin(), values.end());

	nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);
}
*/

/*
TEST_F(PlaygroundTests, test_s_1) {
	auto t = ::runLightBenchmarkSuit(true);
	delete[] t;
}

TEST_F(PlaygroundTests, test_s_2) {
	std::atomic<int> s;
	s = 0;
	auto func = PRAGMA_THREADS_FOR {
		s++;
	};

	samediff::Threads::parallel_for(func, 0, 8192, 1, 4);
	std::vector<Nd4jLong> values;

	for (int e = 0; e < 100000; e++) {
		s = 0;

		auto timeStart = std::chrono::system_clock::now();
		//samediff::Threads::parallel_for(func, 0, 8192, 1, 4);
		PRAGMA_OMP_PARALLEL_THREADS(4) {
			s++;
		}

		auto timeEnd = std::chrono::system_clock::now();
		auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds> (timeEnd - timeStart).count();
		values.emplace_back(outerTime);
	};
	std::sort(values.begin(), values.end());

	nd4j_printf("Time: %lld;\n", values[values.size() / 2]);
}
 */
 /*
 TEST_F(PlaygroundTests, test_s_4) {
	 std::atomic<float> f;
	 std::atomic<int> s;
	 std::vector<Nd4jLong> valuesX, valuesY;
	 int iterations = 1000;
	 s = 0;
	 auto func = PRAGMA_THREADS_FOR {
		 s++;
	 };

	 samediff::Threads::parallel_for(func, 0, 8192, 1, 4);

	 ////////

	 auto x = NDArrayFactory::create<float>('c', {32, 3, 256, 256});
	 auto z = NDArrayFactory::create<float>('c', {32, 3, 256, 256});
	 x.linspace(1.0);

	 auto xs0 = x.sizeAt(0);
	 auto xs1 = x.sizeAt(1);
	 auto xs2 = x.sizeAt(2);
	 auto xs3 = x.sizeAt(3);

	 auto buffer = x.bufferAsT<float>();
	 auto zbuffer = z.bufferAsT<float>();

	 for (int e = 0; e < iterations; e++) {
		 auto timeStart = std::chrono::system_clock::now();
		 PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
		 for (int i = 0; i < xs0; i++) {
			 for (int j = 0; j < xs1; j++) {
				 auto thread_id = omp_get_thread_num();
				 for (int k = 0; k < xs2; k++) {
					 for (int l = 0; l < xs3; l++) {
						 zbuffer[thread_id] += buffer[i * j + (k*l)] * 2.5f;
					 }
				 }
			 }
		 }
		 auto timeEnd = std::chrono::system_clock::now();
		 auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
		 valuesX.emplace_back(outerTime);
	 }


	 for (int e = 0; e < iterations; e++) {
		 auto timeStart = std::chrono::system_clock::now();
		 auto f2d = PRAGMA_THREADS_FOR_2D {
			 for (auto i = start_x; i < stop_x; i++) {
				 for (auto j = start_y; j < stop_y; j++) {

					 for (auto k = 0; k < xs2; k++) {
						 for (auto l = 0; l < xs3; l++) {
							 zbuffer[thread_id] += buffer[i * j + (k * l)] * 2.5f;
						 }
					 }
				 }
			 }
		 };
		 samediff::Threads::parallel_for(f2d, 0, xs0, 1, 0, xs1, 1);

		 auto timeEnd = std::chrono::system_clock::now();
		 auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
		 valuesY.emplace_back(outerTime);
	 }

	 if (valuesX.size() > 0) {
		 std::sort(valuesX.begin(), valuesX.end());
		 nd4j_printf("OpenMP time: %lld; Min: %lld; Max: %lld;\n", valuesX[valuesX.size() / 2], valuesX[0], valuesX[valuesX.size() - 1]);
	 }

	 if (valuesY.size() > 0) {
		 std::sort(valuesY.begin(), valuesY.end());
		 nd4j_printf("Threads time: %lld; Min: %lld; Max: %lld;\n", valuesY[valuesY.size() / 2], valuesY[0], valuesY[valuesY.size() - 1]);
	 }

	 nd4j_printf("Sum: %f\n", z.sumNumber().e<float>(0));
 }


 TEST_F(PlaygroundTests, test_s_5) {
	 auto x = NDArrayFactory::create<float>('c', {32, 1, 28, 28});

	 std::vector<Nd4jLong> values;
	 auto iterations = 100;

	 auto startX = 0;
	 auto stopX = x.sizeAt(0);
	 auto incX = 1;
	 auto startY = 0;
	 auto stopY = x.sizeAt(1);
	 auto incY = 1;
	 auto numThreads = 4;

	 // number of elements per loop
	 auto delta_x = (stopX - startX);
	 auto delta_y = (stopY - startY);

	 // number of iterations per loop
	 auto itersX = delta_x / incX;
	 auto itersY = delta_y / incY;

	 for (int e = 0; e < iterations; e++) {
		 auto timeStart = std::chrono::system_clock::now();

		 // picking best fit here
		 auto splitLoop = samediff::ThreadsHelper::pickLoop2d(numThreads, itersX, itersY);
		 auto span = samediff::Span2::build(splitLoop, 0, numThreads, startX, stopX, incX, startY, stopY, incY);

		 auto timeEnd = std::chrono::system_clock::now();
		 auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
		 values.emplace_back(outerTime);
	 }

	 std::sort(values.begin(), values.end());

	 nd4j_printf("Calculations time: [Median: %lld; Min: %lld; Max: %lld;]\n", values[values.size() / 2], values[0], values[values.size()-1]);
 }


 TEST_F(PlaygroundTests, test_s_6) {
	 auto x = NDArrayFactory::create<float>('c', {1024 * 1024 * 64});
	 auto buffer = x.bufferAsT<float>();
	 auto len = x.lengthOf();
	 std::vector<Nd4jLong> values;
	 auto iterations = 1000;

	 for (int i = 0; i < iterations; i++) {
		 auto timeStart = std::chrono::system_clock::now();

		 // picking best fit here
		 for (int e = 0; e < len; e++) {
			 buffer[e] = (buffer[e] + 1.72f) * 3.17f - 0.0012f;
		 }

		 auto timeEnd = std::chrono::system_clock::now();
		 auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
		 values.emplace_back(outerTime);
	 }

	 std::sort(values.begin(), values.end());

	 nd4j_printf("Calculations time: [Median: %lld; Min: %lld; Max: %lld;]\n", values[values.size() / 2], values[0], values[values.size()-1]);
 }


 TEST_F(PlaygroundTests, test_s_3) {
	 std::atomic<int> s;
	 s = 0;
	 auto func = PRAGMA_THREADS_FOR {
		 s++;
	 };

	 for (int e = 0; e < 10000; e++) {

		 samediff::Threads::parallel_for(func, 0, 8192, 1, 4);
	 }
 }
  */

  /*
  TEST_F(PlaygroundTests, test_relubp_1) {
	  auto x = NDArrayFactory::create<float>('c', {128, 64, 224, 224});
	  auto y = x.ulike();
	  auto z = x.ulike();
	  RandomGenerator rng(119, 120);
	  RandomLauncher::fillUniform(LaunchContext::defaultContext(), rng, &x, -1.0, 1.0);
	  RandomLauncher::fillUniform(LaunchContext::defaultContext(), rng, &y, -1.0, 1.0);

	  int iterations = 10;

	  auto timeStart = std::chrono::system_clock::now();
	  for (int e = 0; e < iterations; e++)
		  ops::helpers::reluDerivative(LaunchContext::defaultContext(), &x, &y, &z);
	  auto timeEnd = std::chrono::system_clock::now();

	  auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
	  auto time = (Nd4jLong) outerTime / iterations;
	  auto bw = (1000000L * (float) (x.lengthOf() * x.sizeOfT()) / time) / 1024 / 1024 / 1024;

	  nd4j_printf("Time: %lld; BW: %f GB/s\n", time, bw);
  }

  //////////////////////////////////////////////////////////////////////
  TEST_F(PlaygroundTests, my) {

	  int bS=8, iD=32,iH=32,iW=32,  iC=128,  kD=2,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=2,dH=2,dW=2;
	  int       oD,oH,oW;

	  nd4j::ops::ConvolutionUtils::calcOutSizeDeconv3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, 0);

	  printf("!!%i, %i, %i\n", oD,oH,oW);

	  NDArray col('c', {bS, iC, kD, kH, kW, iD, iH, iW}, nd4j::DataType::DOUBLE);
	  NDArray vol('c', {bS, iC, oD, oH, oW}, nd4j::DataType::DOUBLE);

	  col = 3.77;
	  vol = -10.33;

	  auto variableSpace = new VariableSpace();
	  auto block = new Context(1, variableSpace, false);  // not-in-place

	  auto timeStart = std::chrono::system_clock::now();
	  nd4j::ops::ConvolutionUtils::col2vol(*block, col, vol, sD, sH, sW, pD, pH, pW, dD, dH, dW);
	  auto timeEnd = std::chrono::system_clock::now();
	  auto time = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

	  printf("time: %i \n", time);

	  delete block;
	  delete variableSpace;
  }

  TEST_F(PlaygroundTests, my) {

	  int bS=32, iD=32,iH=64,iW=64,  iC=128,  kD=2,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=2,dH=2,dW=2;
	  int       oD,oH,oW;

	  // nd4j::ops::ConvolutionUtils::calcOutSizeDeconv3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, 0);
	  nd4j::ops::ConvolutionUtils::calcOutSizeDeconv2D(oH, oW, kH, kW, sH, sW, pH, pW,dH, dW, iH, iW, 0);

	  printf("!!%i, %i, %i\n", oD,oH,oW);

	  // NDArray col('c', {bS, iC, kD, kH, kW, iD, iH, iW}, nd4j::DataType::DOUBLE);
	  // NDArray vol('c', {bS, iC, oD, oH, oW}, nd4j::DataType::DOUBLE);
	  NDArray col('c', {bS, iC, kH, kW, iH, iW}, nd4j::DataType::DOUBLE);
	  NDArray im('c', {bS, iC, oH, oW}, nd4j::DataType::DOUBLE);

	  col = 3.77;
	  // vol = -10.33;
	  im = -10.33;

	  auto variableSpace = new VariableSpace();
	  auto block = new Context(1, variableSpace, false);  // not-in-place

	  auto timeStart = std::chrono::system_clock::now();
	  // nd4j::ops::ConvolutionUtils::col2vol(*block, col, vol, sD, sH, sW, pD, pH, pW, dD, dH, dW);
	  nd4j::ops::helpers::col2im(*col.getContext(), col, im, sH, sW, pH, pW, iH, iW, dH, dW);
	  auto timeEnd = std::chrono::system_clock::now();
	  auto time = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

	  printf("time: %i \n", time);

	  delete block;
	  delete variableSpace;
  }

  */
  /*
  TEST_F(PlaygroundTests, my) {

	  NDArray a('c', { 2,3,4 }, nd4j::DataType::DOUBLE);
	  a({ 0,0, 0,1, 0,1 }).printShapeInfo();
	  a({ 0,1, 0,0, 0,1 }).printShapeInfo();
	  a({ 0,0, 0,1, 0,1 }).printShapeInfo();

  }
  **/