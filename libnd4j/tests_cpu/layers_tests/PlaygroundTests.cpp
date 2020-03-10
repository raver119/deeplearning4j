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
#include <graph/Graph.h>
#include <chrono>
#include <graph/Node.h>
#include <ops/declarable/CustomOperations.h>
#include <graph/profiling/GraphProfilingHelper.h>
#include <loops/type_conversions.h>
#include <helpers/threshold.h>
#include <helpers/MmulHelper.h>
#include <ops/ops.h>
#include <helpers/OmpLaunchHelper.h>
#include <helpers/GradCheck.h>
#include <ops/declarable/helpers/im2col.h>
#include <helpers/Loops.h>
#include <helpers/RandomLauncher.h>
#include <ops/declarable/helpers/convolutions.h>

#include <helpers/BenchmarkHelper.h>
#include <ops/declarable/helpers/scatter.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <array>
#include <performance/benchmarking/FullBenchmarkSuit.h>
#include <performance/benchmarking/LightBenchmarkSuit.h>

#include <ops/declarable/helpers/legacy_helpers.h>
#include <ops/declarable/helpers/addBias.h>
#include <algorithm>
#include <numeric> 
#include <random>
#include <helpers/LoopsCoordsHelper.h>





using namespace sd;
using namespace sd::graph;

class PlaygroundTests : public testing::Test {
public:
	int numIterations = 3;
	int poolSize = 10;

	PlaygroundTests() {
	}
};


class CorrectnessTests : public testing::Test {
public:
	CorrectnessTests() {
	}
};
#if 0
constexpr int outter_loops = 1;// 10;
constexpr int inner_loops = 1;// 10;
#else
constexpr int outter_loops = 2;
constexpr int inner_loops = 2;
#endif
#define CHECK_CORRECTNESS 1
template<typename Op, typename... Args>
void time_it(Op op, double totalFlops, Args&&... args) {
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
	double avg = (double)sum / outter_loops;
	auto sum_sq = std::accumulate(std::begin(values), std::end(values), 0.0, [&avg](int sumsq, int x) { return sumsq + (x - avg) * (x - avg); });

	if (totalFlops > 0.0) {
		nd4j_printf("Median: %f us\tAvg: %f (sd: %f)\tFlops: %f Mflops\n",
			values[values.size() / 2], avg, sqrt(sum_sq / n_1),
			(totalFlops) / avg);
	}
	else {
		nd4j_printf("Median: %f us\tAvg: %f (sd: %f)\n",
			values[values.size() / 2], avg, sqrt(sum_sq / n_1));
	}
}

template<typename T>
void fill_matrice_lastC(sd::NDArray& arr, sd::NDArray* fill = nullptr, bool zero = false) {
	Nd4jLong coords[MAX_RANK] = {};
	constexpr int last_ranks = 2;
	std::random_device rd;
	std::mt19937 gen(rd());
	//for floats
	std::uniform_real_distribution<T> dis((T)0.0, (T)2.0);
	T* x = arr.bufferAsT<T>();
	Nd4jLong* shapeInfo = arr.getShapeInfo();
	Nd4jLong* strides = arr.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);

	if (rank < last_ranks) {
		return;
	}
	bool random = (fill == nullptr);
	size_t t = 1;
	const int r = rank - last_ranks;
	size_t M = bases[rank - 2];
	size_t stride_m = strides[rank - 2];
	size_t N = bases[rank - 1];
	size_t stride_n = strides[rank - 1];
	for (size_t i = 0; i < rank - last_ranks; i++) {
		t *= bases[i];
	}
	size_t offset = 0;
	if (random) {
		//first matrix is random
		//next ones copy
		if (zero) {
			for (size_t i = 0; i < t; i++) {
				//1 time
				for (size_t j = 0; j < M; j++) {
					for (size_t n_i = 0; n_i < N; n_i++) {
						x[offset + j * stride_m + n_i * stride_n] = 0;
					}
				}
				offset = sd::inc_coords(bases, strides, coords, offset, r);
			}
		}
		else {
			for (size_t i = 0; i < 1; i++) {
				//1 time
				for (size_t j = 0; j < M; j++) {
					for (size_t n_i = 0; n_i < N; n_i++) {
						x[j * stride_m + n_i * stride_n] = dis(gen);
						//nd4j_printf("%lf  %ld  %ld %ld\n", x[j * stride_n + n_i], stride_n,j,n_i);
					}
				}
				offset = sd::inc_coords(bases, strides, coords, offset, r);
			}


			for (size_t i = 1; i < t; i++) {
				//copy_from the first
				for (size_t j = 0; j < M; j++) {
					for (size_t n_i = 0; n_i < N; n_i++) {
						x[offset + j * stride_m + n_i * stride_n] = x[j * stride_m + n_i * stride_n];
					}
				}
				offset = sd::inc_coords(bases, strides, coords, offset, r);
			}
		}
	}
	else {
		auto fill_buffer = fill->bufferAsT<float>();
		for (size_t i = 0; i < t; i++) {
			//m*n
			auto fill_stride_m = fill->stridesOf()[0];
			auto fill_strinde_n = fill->stridesOf()[1];
			for (size_t j = 0; j < M; j++) {
				for (size_t n_i = 0; n_i < N; n_i++) {
					x[offset + j * stride_m + n_i * stride_n] = fill_buffer[j * fill_stride_m + n_i * fill_strinde_n];
				}
			}
			offset = sd::inc_coords(bases, strides, coords, offset, r);
		}

	}
}

int bnch_cases[][5] = {
	//{1,1,4,5,2},
	{1,3,4,5,2},
	{1,1,32,64,64},
	{1,1,64,64,64},
	{1,1,64,256,64},
	{4*32,128,16,16,16},
	{1,4 * 32*128,16,16,16},
	{32,128,64,64,64}, 
	{1,4096,64,64,64},
	{32,128,64,128,64},
	{1,4096,64,128,64},
	{32,128,128,64,128},
	{1,4096,128,64,128},
	{32,128,128,128,128},
	{1,4096,128,128,128},
	{32,128,64,256,64},
	{1,4096,64,256,64},
	{32,128,32,64,64},
	{1,4096,32,64,64}, 
	//{4,32,512,512,512},
	//{1,4*32,512,512,512}

};
#if 0
TEST_F(PlaygroundTests, TestMatmMul_BNCH1) {

	char out_orders[] = { 'c', 'f' };

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		int batch_k1 = bnch_cases[k][0];
		int batch_k2 = bnch_cases[k][1];
		int M = bnch_cases[k][2];
		int K = bnch_cases[k][3];
		int N = bnch_cases[k][4];
		double alpha = 1.0;
		double beta = 1.0;
		for (auto out_order : out_orders) {
			auto a = NDArrayFactory::create<float>('c', { batch_k1,batch_k2, M,K });
			auto b = NDArrayFactory::create<float>('c', { batch_k1,batch_k2, K,N });
			auto c_ref = NDArrayFactory::create<float>(out_order, { batch_k1,batch_k2, M,N });
			auto c_actual = NDArrayFactory::create<float>(out_order, { batch_k1,batch_k2, M,N });
			fill_matrice_lastC<float>(a, nullptr);
			fill_matrice_lastC<float>(b, nullptr);
			fill_matrice_lastC<float>(c_actual, nullptr, true);

			auto a_mk = a.subarray({ NDIndex::point(0) , NDIndex::point(0)  ,NDIndex::all(), NDIndex::all() });
			auto b_kn = b.subarray({ NDIndex::point(0) , NDIndex::point(0)  ,NDIndex::all(), NDIndex::all() });

			a_mk.reshapei({ M,K });
			b_kn.reshapei({ K,N });

			auto ptr_c_mn = MmulHelper::mmulMxM(&a_mk, &b_kn, nullptr, alpha, beta, out_order);
			fill_matrice_lastC<float>(c_ref, ptr_c_mn);


			//check performance
			double total_FlOps = batch_k1 * batch_k2 * 2.0 * M * K * N;
			nd4j_printf("out_order %c batch_1 %ld batch_2 %ld M %ld K %ld N %ld:::: ", out_order, bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3], bnch_cases[k][4]);

			time_it(MmulHelper::mmulNxN, total_FlOps, &a, &b, &c_actual, alpha, beta, out_order);


#if CHECK_CORRECTNESS
			fill_matrice_lastC<float>(c_actual, nullptr, true);

			MmulHelper::mmulNxN(&a, &b, &c_actual, alpha, beta, out_order);

#if 0
			fill_matrice_lastC<float>(c_actual, nullptr, true);
			MmulHelper::mmulNxN_2(&a, &b, &c_actual, alpha, beta, out_order);
			a_mk.printShapeInfo("smallA_MK");
			a_mk.printShapeInfo("smallB_KN");
			ptr_c_mn->printShapeInfo("smallC_MN");
				a_mk.printIndexedBuffer("smallA_MK");
			b_kn.printIndexedBuffer("smallB_KN");
			ptr_c_mn->printIndexedBuffer("smallC_MN");
			a.printShapeInfo("A");
			b.printShapeInfo("B");
			c_ref.printShapeInfo("C REFERENCE");
			c_actual.printShapeInfo("C");
			a.printIndexedBuffer("A");
			b.printIndexedBuffer("B");
			c_ref.printIndexedBuffer("C REFERENCE");
			c_actual.printIndexedBuffer("C ACTUAL");


#endif
#endif
			delete ptr_c_mn;
#if CHECK_CORRECTNESS
			ASSERT_TRUE(c_ref.isSameShape(&c_actual));
			ASSERT_TRUE(c_ref.equalsTo(&c_actual));
#endif 
		}//out_order
	}
}
#endif
#if 1

TEST_F(PlaygroundTests, TestMatmMul_BNCH2) {

	char out_orders[] = { 'c', 'f' };

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		int batch_k1 = bnch_cases[k][0];
		int batch_k2 = bnch_cases[k][1];
		int M = bnch_cases[k][2];
		int K = bnch_cases[k][3];
		int N = bnch_cases[k][4];
		double alpha = 1.0;
		double beta = 1.0;
		for (auto out_order : out_orders) {
			auto a = NDArrayFactory::create<float>('c', { batch_k1,batch_k2, M,K });
			auto b = NDArrayFactory::create<float>('c', { batch_k1,batch_k2, K,N });
			auto c_ref = NDArrayFactory::create<float>(out_order, { batch_k1,batch_k2, M,N });
			auto c_actual = NDArrayFactory::create<float>(out_order, { batch_k1,batch_k2, M,N });
			fill_matrice_lastC<float>(a, nullptr);
			fill_matrice_lastC<float>(b, nullptr);
			fill_matrice_lastC<float>(c_actual, nullptr, true);

			auto a_mk = a.subarray({ NDIndex::point(0) , NDIndex::point(0)  ,NDIndex::all(), NDIndex::all() });
			auto b_kn = b.subarray({ NDIndex::point(0) , NDIndex::point(0)  ,NDIndex::all(), NDIndex::all() });

			a_mk.reshapei({ M,K });
			b_kn.reshapei({ K,N });

			auto ptr_c_mn = MmulHelper::mmulMxM(&a_mk, &b_kn, nullptr, alpha, beta, out_order);
			fill_matrice_lastC<float>(c_ref, ptr_c_mn);


			//check performance
			double total_FlOps = batch_k1 * batch_k2 * 2.0 * M * K * N;
			nd4j_printf("out_order %c batch_1 %ld batch_2 %ld M %ld K %ld N %ld:::: ", out_order, bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3], bnch_cases[k][4]);
			time_it(MmulHelper::mmulNxN_2, total_FlOps, &a, &b, &c_actual, alpha, beta, out_order);

#if CHECK_CORRECTNESS
			fill_matrice_lastC<float>(c_actual, nullptr, true);

			MmulHelper::mmulNxN_2(&a, &b, &c_actual, alpha, beta, out_order);

#if 0
			fill_matrice_lastC<float>(c_actual, nullptr, true);
			MmulHelper::mmulNxN_2(&a, &b, &c_actual, alpha, beta, out_order);
			a_mk.printShapeInfo("smallA_MK");
			a_mk.printShapeInfo("smallB_KN");
			ptr_c_mn->printShapeInfo("smallC_MN0");
				a_mk.printIndexedBuffer("smallA_MK");
			b_kn.printIndexedBuffer("smallB_KN");
			ptr_c_mn->printIndexedBuffer("smallC_MN");
			a.printShapeInfo("A");
			b.printShapeInfo("B");
			c_ref.printShapeInfo("C REFERENCE");
			c_actual.printShapeInfo("C");
			a.printIndexedBuffer("A");
			b.printIndexedBuffer("B");
			c_ref.printIndexedBuffer("C REFERENCE");
			c_actual.printIndexedBuffer("C ACTUAL");


#endif
#endif
			delete ptr_c_mn;
#if CHECK_CORRECTNESS
			ASSERT_TRUE(c_ref.isSameShape(&c_actual));
			ASSERT_TRUE(c_ref.equalsTo(&c_actual));
#endif 
		}//out_order
	}
}

#endif
TEST_F(PlaygroundTests, TestMatmMul_BNCH3) {

	char out_orders[] = { 'c', 'f' };

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		int batch_k1 = bnch_cases[k][0];
		int batch_k2 = bnch_cases[k][1];
		int M = bnch_cases[k][2];
		int K = bnch_cases[k][3];
		int N = bnch_cases[k][4];
		double alpha = 1.0;
		double beta = 0.0;
		for (auto out_order : out_orders) {
			auto a = NDArrayFactory::create<float>('c', { batch_k1,batch_k2, M,K });
			auto b = NDArrayFactory::create<float>('c', { batch_k1,batch_k2, K,N });
			auto c_ref = NDArrayFactory::create<float>(out_order, { batch_k1,batch_k2, M,N });
			auto c_actual = NDArrayFactory::create<float>(out_order, { batch_k1,batch_k2, M,N });
			fill_matrice_lastC<float>(a, nullptr);
			fill_matrice_lastC<float>(b, nullptr);
			fill_matrice_lastC<float>(c_actual, nullptr, true);
			auto a_mk = a.subarray({ NDIndex::point(0) , NDIndex::point(0)  ,NDIndex::all(), NDIndex::all() });
			auto b_kn = b.subarray({ NDIndex::point(0) , NDIndex::point(0)  ,NDIndex::all(), NDIndex::all() });

			a_mk.reshapei({ M,K });
			b_kn.reshapei({ K,N });

			auto ptr_c_mn = MmulHelper::mmulMxM(&a_mk, &b_kn, nullptr, alpha, beta, out_order);
			fill_matrice_lastC<float>(c_ref, ptr_c_mn);


			//check performance
			double total_FlOps = batch_k1 * batch_k2 * 2.0 * M * K * N;
			nd4j_printf("out_order %c batch_1 %ld batch_2 %ld M %ld K %ld N %ld:::: ", out_order, bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3], bnch_cases[k][4]);
			time_it(MmulHelper::mmulNxN_3, total_FlOps, &a, &b, &c_actual, alpha, beta, out_order);


			//check_correctness
			fill_matrice_lastC<float>(c_actual, nullptr, true);

			MmulHelper::mmulNxN_3(&a, &b, &c_actual, alpha, beta, out_order);
#if 0

			a_mk.printShapeInfo("smallA_MK");
			a_mk.printShapeInfo("smallB_KN");
			ptr_c_mn->printShapeInfo("smallC_MN");
			a_mk.printIndexedBuffer("smallA_MK");
			b_kn.printIndexedBuffer("smallB_KN");
			ptr_c_mn->printIndexedBuffer("smallC_MN");
			a.printShapeInfo("A");
			b.printShapeInfo("B");
			c_ref.printShapeInfo("C REFERENCE");
			c_actual.printShapeInfo("C");
			a.printIndexedBuffer("A");
			b.printIndexedBuffer("B");
			c_ref.printIndexedBuffer("C REFERENCE");
			c_actual.printIndexedBuffer("C ACTUAL");


#endif
			delete ptr_c_mn;
#if CHECK_CORRECTNESS
			ASSERT_TRUE(c_ref.isSameShape(&c_actual));
			ASSERT_TRUE(c_ref.equalsTo(&c_actual));
#endif

		}//out_order
	}
}
//#define batch_gemm_test 1
//#define xmmul 1
#if xmmul

TEST_F(CorrectnessTests, TestMMulMultiDim) {
	const int bS = 2;
	const int K = 3;
	const int N = 4;

	auto input = NDArrayFactory::create<double>('c', { bS,  K, N });
	auto weights = NDArrayFactory::create<double>('c', { 3 * K, K });
	auto expected = NDArrayFactory::create<double>('c', { bS,  3 * K, N }, { 38,   44,   50,   56, 83,   98,  113,  128, 128,  152,  176,  200, 173,  206,  239,  272, 218,  260,  302,  344, 263,  314,  365,  416, 308,  368,  428,  488, 353,  422,  491,  560, 398,  476,  554,  632, 110,  116,  122,  128, 263,  278,  293,  308, 416,  440,  464,  488, 569,  602,  635,  668, 722,  764,  806,  848, 875,  926,  977, 1028, 1028, 1088, 1148, 1208, 1181, 1250, 1319, 1388, 1334, 1412, 1490, 1568 });

	input.linspace(1);
	weights.linspace(1);

	auto result = MmulHelper::mmul(&weights, &input, nullptr, 1., 0.);
	//  result must have such shape   [bS x 3K x N]

	ASSERT_TRUE(result->isSameShape(&expected));

	//result->printShapeInfo("result shape");
	// result->printBuffer("result buffer");
	ASSERT_TRUE(result->equalsTo(&expected));
	delete result;
}



TEST_F(CorrectnessTests, TestMmulHelper_ND_1) {
	Nd4jLong _expS[] = { 3, 2, 3, 3, 9, 3, 1, 8192, 1, 99 };
	float _expB[] = { 70.f, 80.f, 90.f, 158.f, 184.f, 210.f, 246.f, 288.f, 330.f, 1030.f, 1088.f, 1146.f, 1310.f, 1384.f, 1458.f, 1590.f, 1680.f, 1770.f };

	auto a = NDArrayFactory::create<float>('c', { 2, 3, 4 });
	for (int e = 0; e < a.lengthOf(); e++)
		a.p(e, e + 1);

	auto b = NDArrayFactory::create<float>('c', { 2, 4, 3 });
	for (int e = 0; e < b.lengthOf(); e++)
		b.p(e, e + 1);

	sd::NDArray exp(_expB, _expS);
	auto c = MmulHelper::mmul(&a, &b);

	ASSERT_TRUE(exp.isSameShape(c));
	ASSERT_TRUE(exp.equalsTo(c));

	delete c;
}




TEST_F(CorrectnessTests, TestMatmMul_Again_1) {
	auto a = NDArrayFactory::create<float>('c', { 3, 4, 1 });
	auto b = NDArrayFactory::create<float>('c', { 3, 1, 5 });

	a.linspace(1);
	b.linspace(1);

	float _expB[] = { 1.f,    2.f,    3.f,    4.f,    5.f,    2.f,    4.f,    6.f,    8.f,   10.f,    3.f,    6.f,    9.f,   12.f,   15.f,    4.f,    8.f,   12.f,   16.f,   20.f,   30.f,   35.f,   40.f,   45.f,    50.f,   36.f,   42.f,   48.f,   54.f,   60.f,   42.f,   49.f,   56.f,   63.f,   70.f,   48.f,    56.f,   64.f,   72.f,   80.f,   99.f,  108.f,  117.f,  126.f,  135.f,  110.f,  120.f,  130.f,    140.f,  150.f,  121.f,  132.f,  143.f,  154.f,  165.f,  132.f,  144.f,  156.f,  168.f,  180.f };
	Nd4jLong _expS[] = { 3, 3, 4, 5, 20, 5, 1, 8192, 1, 99 };
	sd::NDArray c(_expB, _expS, sd::LaunchContext::defaultContext(), false);

	//auto c_ = MmulHelper::mmul(&a, &b);
	auto c_ = MmulHelper::mmulNxN(&a, &b, nullptr);

	ASSERT_TRUE(c.isSameShape(c_));
	ASSERT_TRUE(c.equalsTo(c_));

	delete c_;
}

//////////////////////////////////////////////////////////////////////
TEST_F(CorrectnessTests, TestMatmMul_Again_2) {
	auto a = NDArrayFactory::create<double>('c', { 2, 5, 4 });
	auto b = NDArrayFactory::create<double>('c', { 2, 4, 1 });

	a.linspace(1);
	b.linspace(1);

	double _expB[] = { 30.f,    70.f,   110.f,   150.f,   190.f,   590.f,   694.f,   798.f,   902.f,  1006.f };
	Nd4jLong _expS[] = { 3, 2, 5, 1, 5, 1, 1, 16384, 1, 99 };
	sd::NDArray c(_expB, _expS);

	auto c_ = MmulHelper::mmul(&a, &b);

	ASSERT_TRUE(c.isSameShape(c_));

	ASSERT_TRUE(c.equalsTo(c_));

	delete c_;
}



#endif
#if batch_gemm_test

TEST_F(CorrectnessTests, Test_Batched_Gemm_1) {
	auto a = NDArrayFactory::create<double>('c', { 1, 3 }, { 1, 1, 1 });
	auto b = NDArrayFactory::create<double>('c', { 1, 3 }, { 0, 0, 0 });
	auto x = NDArrayFactory::create<double>('f', { 3, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
	auto y = NDArrayFactory::create<double>('f', { 3, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });

	auto exp = MmulHelper::mmul(&x, &y);

	sd::ops::batched_gemm op;
	auto result = op.evaluate({ &a, &b, &x, &x, &x, &y, &y, &y }, {}, { 111, 111, 3, 3, 3, 3, 3, 3, 3 });
	ASSERT_EQ(ND4J_STATUS_OK, result->status());

	ASSERT_EQ(3, result->size());

	for (int e = 0; e < 3; e++) {
		auto z = result->at(e);

		//        exp->printIndexedBuffer("e");
		//        z->printIndexedBuffer("z");

		ASSERT_TRUE(exp->isSameShape(z));
		ASSERT_TRUE(exp->equalsTo(z));
	}

	delete exp;
	delete result;
}

TEST_F(CorrectnessTests, Test_Batched_Gemm_2) {
	auto a = NDArrayFactory::create<double>('c', { 1, 3 }, { 1, 1, 1 });
	auto b = NDArrayFactory::create<double>('c', { 1, 3 }, { 0, 0, 0 });
	auto x = NDArrayFactory::create<double>('c', { 3, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
	auto y = NDArrayFactory::create<double>('c', { 3, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });

	auto exp = MmulHelper::mmul(&x, &y);

	sd::ops::batched_gemm op;
	auto result = op.evaluate({ &a, &b, &x, &x, &x, &y, &y, &y }, {}, { 112, 112, 3, 3, 3, 3, 3, 3, 3 });
	ASSERT_EQ(ND4J_STATUS_OK, result->status());

	ASSERT_EQ(3, result->size());

	for (int e = 0; e < 3; e++) {
		auto z = result->at(e);

		//exp->printIndexedBuffer("e");
		//z->printIndexedBuffer("z");

		ASSERT_TRUE(exp->isSameShape(z));
		ASSERT_TRUE(exp->equalsTo(z));
	}

	delete exp;
	delete result;
}

TEST_F(CorrectnessTests, Test_Batched_Gemm_3) {
	auto a = NDArrayFactory::create<double>('c', { 1, 3 }, { 1, 1, 1 });
	auto b = NDArrayFactory::create<double>('c', { 1, 3 }, { 0, 0, 0 });
	auto x = NDArrayFactory::create<double>('c', { 3, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
	auto y = NDArrayFactory::create<double>('f', { 3, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });

	auto exp = MmulHelper::mmul(&x, &y);

	sd::ops::batched_gemm op;
	auto result = op.evaluate({ &a, &b, &x, &x, &x, &y, &y, &y }, {}, { 112, 111, 3, 3, 3, 3, 3, 3, 3 });
	ASSERT_EQ(ND4J_STATUS_OK, result->status());

	ASSERT_EQ(3, result->size());

	for (int e = 0; e < 3; e++) {
		auto z = result->at(e);

		//        exp->printIndexedBuffer("e");
		//        z->printIndexedBuffer("z");

		ASSERT_TRUE(exp->isSameShape(z));
		ASSERT_TRUE(exp->equalsTo(z));
	}

	delete exp;
	delete result;
}

TEST_F(CorrectnessTests, Test_Batched_Gemm_4) {
	auto a = NDArrayFactory::create<double>('c', { 1, 3 }, { 1, 1, 1 });
	auto b = NDArrayFactory::create<double>('c', { 1, 3 }, { 0, 0, 0 });
	auto x = NDArrayFactory::create<double>('f', { 5, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 });
	auto y = NDArrayFactory::create<double>('f', { 3, 4 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });

	auto exp = MmulHelper::mmul(&x, &y);

	sd::ops::batched_gemm op;
	auto result = op.evaluate({ &a, &b, &x, &x, &x, &y, &y, &y }, {}, { 111, 111, 5, 4, 3, 5, 3, 5, 3 });
	ASSERT_EQ(ND4J_STATUS_OK, result->status());

	ASSERT_EQ(3, result->size());

	for (int e = 0; e < 3; e++) {
		auto z = result->at(e);

		//exp->printIndexedBuffer("e");
		//z->printIndexedBuffer("z");

		ASSERT_TRUE(exp->isSameShape(z));
		ASSERT_TRUE(exp->equalsTo(z));
	}

	delete exp;
	delete result;
}

TEST_F(CorrectnessTests, Test_Batched_Gemm_5) {
	auto a = NDArrayFactory::create<double>('c', { 1, 3 }, { 1, 1, 1 });
	auto b = NDArrayFactory::create<double>('c', { 1, 3 }, { 0, 0, 0 });
	auto x = NDArrayFactory::create<double>('c', { 5, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 });
	auto y = NDArrayFactory::create<double>('c', { 3, 4 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });

	auto exp = MmulHelper::mmul(&x, &y);

	sd::ops::batched_gemm op;
	auto result = op.evaluate({ &a, &b, &x, &x, &x, &y, &y, &y }, {}, { 112, 112, 5, 4, 3, 3, 4, 5, 3 });
	ASSERT_EQ(ND4J_STATUS_OK, result->status());

	ASSERT_EQ(3, result->size());

	for (int e = 0; e < 3; e++) {
		auto z = result->at(e);

		//exp->printIndexedBuffer("e");
		//z->printIndexedBuffer("z");

		ASSERT_TRUE(exp->isSameShape(z));
		ASSERT_TRUE(exp->equalsTo(z));
	}

	delete exp;
	delete result;
}


TEST_F(CorrectnessTests, Test_Batched_Gemm_6) {
	auto a = NDArrayFactory::create<double>('c', { 1, 3 }, { 1, 1, 1 });
	auto b = NDArrayFactory::create<double>('c', { 1, 3 }, { 0, 0, 0 });
	auto x = NDArrayFactory::create<double>('f', { 2, 5 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
	auto y = NDArrayFactory::create<double>('f', { 5, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 });

	auto exp = MmulHelper::mmul(&x, &y);

	sd::ops::batched_gemm op;
	auto result = op.evaluate({ &a, &b, &x, &x, &x, &y, &y, &y }, {}, { 111, 111, 2, 3, 5, 2, 5, 2, 3 });
	ASSERT_EQ(ND4J_STATUS_OK, result->status());

	ASSERT_EQ(3, result->size());

	for (int e = 0; e < 3; e++) {
		auto z = result->at(e);

		//exp->printIndexedBuffer("e");
		//z->printIndexedBuffer("z");

		ASSERT_TRUE(exp->isSameShape(z));
		ASSERT_TRUE(exp->equalsTo(z));
	}

	delete exp;
	delete result;
}

TEST_F(CorrectnessTests, Test_Batched_Gemm_7) {
	auto a = NDArrayFactory::create<double>('c', { 1, 3 }, { 1, 1, 1 });
	auto b = NDArrayFactory::create<double>('c', { 1, 3 }, { 0, 0, 0 });
	auto x = NDArrayFactory::create<double>('c', { 2, 5 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
	auto y = NDArrayFactory::create<double>('c', { 5, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 });

	auto exp = MmulHelper::mmul(&x, &y);

	// exp->printShapeInfo("exp shape");

	sd::ops::batched_gemm op;
	auto result = op.evaluate({ &a, &b, &x, &x, &x, &y, &y, &y }, {}, { 112, 112, 2, 3, 5, 5, 3, 2, 3 });
	ASSERT_EQ(ND4J_STATUS_OK, result->status());

	ASSERT_EQ(3, result->size());

	for (int e = 0; e < 3; e++) {
		auto z = result->at(e);

		//exp->printIndexedBuffer("e");
		//z->printIndexedBuffer("z");

		ASSERT_TRUE(exp->isSameShape(z));
		ASSERT_TRUE(exp->equalsTo(z));
	}

	delete exp;
	delete result;
}

TEST_F(CorrectnessTests, Test_Batched_Gemm_Validation_1) {
	auto a = NDArrayFactory::create<float>('c', { 1, 3 }, { 1.f, 1.f, 1.f });
	auto b = NDArrayFactory::create<double>('c', { 1, 3 }, { 0.f, 0.f, 0.f });
	auto x = NDArrayFactory::create<float16>('c', { 2, 5 }, { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f });
	auto y = NDArrayFactory::create<float>('c', { 5, 3 }, { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f });

	sd::ops::batched_gemm op;
	try {
		auto result = op.evaluate({ &a, &b, &x, &x, &x, &y, &y, &y }, {}, { 112, 112, 2, 3, 5, 5, 3, 2, 3 });
		delete result;
		ASSERT_TRUE(false);
	}
	catch (std::invalid_argument& e) {
		//
	}
}

TEST_F(CorrectnessTests, Test_Batched_Gemm_Validation_2) {
	auto a = NDArrayFactory::create<float>('c', { 1, 3 }, { 1.f, 1.f, 1.f });
	auto b = NDArrayFactory::create<double>('c', { 1, 3 }, { 0.f, 0.f, 0.f });
	auto x = NDArrayFactory::create<float>('c', { 2, 5 }, { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f });
	auto y = NDArrayFactory::create<float>('c', { 5, 3 }, { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f });

	auto z = NDArrayFactory::create<double>('c', { 2, 3 });

	sd::ops::batched_gemm op;
	try {
		auto result = op.execute({ &a, &b, &x, &x, &x, &y, &y, &y }, { &z }, {}, { 112, 112, 2, 3, 5, 5, 3, 2, 3 }, {});
		ASSERT_TRUE(false);
	}
	catch (std::invalid_argument& e) {
		//
	}
}

#endif


#if 0

TEST_F(PlaygroundTests, test_avx) {
	nd4j_printf("Optimal level: %i; Binary level: %i;\n", ::optimalLevel(), ::binaryLevel());
}

TEST_F(PlaygroundTests, test_split_1) {
	auto axis = NDArrayFactory::create<int>(1);
	auto array = NDArrayFactory::create<double>('c', { 1, 512 });

	auto outA = NDArrayFactory::create<double>('c', { 1, 128 });
	auto outB = outA.ulike();
	auto outC = outA.ulike();
	auto outD = outA.ulike();

	sd::ops::split op;

	auto timeStart = std::chrono::system_clock::now();

	auto result = op.execute({ &axis, &array }, { &outA, &outB, &outC, &outD }, { 4 });

	auto timeEnd = std::chrono::system_clock::now();
	auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
	nd4j_printf("Split time: %lld us;\n", outerTime);

	ASSERT_EQ(Status::OK(), result);
}

TEST_F(PlaygroundTests, test_concat_1) {
	auto t = NDArrayFactory::create<double>('c', { 1, 28 });
	auto u = NDArrayFactory::create<double>('c', { 1, 128 });
	auto v = NDArrayFactory::create<int>(1);
	auto z = NDArrayFactory::create<double>('c', { 1, 156 });

	sd::ops::concat op;
	auto timeStart = std::chrono::system_clock::now();

	auto status = op.execute({ &t, &u, &v }, { &z }, { true });

	auto timeEnd = std::chrono::system_clock::now();
	auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();

	nd4j_printf("Concat time: %lld\n", outerTime);

	ASSERT_EQ(Status::OK(), status);
}

TEST_F(PlaygroundTests, test_gather_1) {
	// this test will run ONLY if this model exists
	if (sd::graph::getFileSize("/home/raver119/Downloads/Bert_minimal_model/bert_minimal_model.fb") < 0)
		return;

	auto x = NDArrayFactory::create<float>('c', { 30522, 768 });
	auto y = NDArrayFactory::fromNpyFile("/home/raver119/Downloads/Bert_minimal_model/bert_minimal_input_IteratorGetNext.numpy");
	auto z = NDArrayFactory::create<float>('c', { 4, 128, 768 });

	x.linspace(1.0f, 0.3f);


	sd::ops::gather op;
	auto timeStart = std::chrono::system_clock::now();

	auto status = op.execute({ &x, &y }, { &z });

	auto timeEnd = std::chrono::system_clock::now();
	auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();

	nd4j_printf("Gather time: %lld\n", outerTime);
}

TEST_F(PlaygroundTests, test_matmul_1) {
	auto x = NDArrayFactory::create<float>('c', { 512, 768 });
	auto y = NDArrayFactory::create<float>('c', { 768, 768 });
	auto z = NDArrayFactory::create<float>('c', { 512, 768 });

	x.linspace(1.0f, 0.3f);
	y.linspace(1.0f, 0.2f);

	sd::ops::matmul op;
	auto timeStart = std::chrono::system_clock::now();

	auto status = op.execute({ &x, &y }, { &z }, { 0, 0 });

	auto timeEnd = std::chrono::system_clock::now();
	auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();

	nd4j_printf("Time: %lld\n", outerTime);
}

TEST_F(PlaygroundTests, test_matmul_2) {
	auto x1 = NDArrayFactory::create<float>('c', { 4, 12, 128, 128 });
	auto y1 = NDArrayFactory::create<float>('c', { 4, 12, 128, 64 });
	auto z1 = NDArrayFactory::create<float>('c', { 4, 12, 128, 64 });

	auto x2 = NDArrayFactory::create<float>('c', { 512, 768 });
	auto y2 = NDArrayFactory::create<float>('c', { 768, 768 });
	auto z2 = NDArrayFactory::create<float>('c', { 512, 768 });

	sd::ops::matmul op;
	auto timeStart1 = std::chrono::system_clock::now();

	op.execute({ &x1, &y1 }, { &z1 }, { 0, 0 });

	auto timeStart2 = std::chrono::system_clock::now();

	op.execute({ &x2, &y2 }, { &z2 }, { 0, 0 });

	auto timeEnd = std::chrono::system_clock::now();

	auto t1 = std::chrono::duration_cast<std::chrono::microseconds>(timeStart2 - timeStart1).count();
	auto t2 = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart2).count();

	nd4j_printf("Time 1: %lld; Time 2: %lld;\n", t1, t2);
}


TEST_F(PlaygroundTests, test_reduce_mean_1) {
	auto x = NDArrayFactory::create<float>('c', { 512, 768 });
	auto y = NDArrayFactory::create<int>(1);
	auto z = NDArrayFactory::create<float>('c', { 512 });

	x.assign(1.f);

	sd::ops::reduce_mean op;

	auto timeStart = std::chrono::system_clock::now();

	auto status = op.execute({ &x, &y }, { &z });

	auto timeEnd = std::chrono::system_clock::now();
	auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();

	ASSERT_EQ(Status::OK(), status);

	nd4j_printf("Time: %lld us;\n", outerTime);

	z.printLinearBuffer();
}

TEST_F(PlaygroundTests, test_reduce_mean_2) {
	auto x = NDArrayFactory::create<float>('c', { 512, 768 });
	auto y = NDArrayFactory::create<int>(1);
	auto z = NDArrayFactory::create<float>('c', { 512 });

	auto rows = x.sizeAt(0);
	auto cols = x.sizeAt(1);

	auto inBuff = x.bufferAsT<float>();
	auto outBuff = x.bufferAsT<float>();

	auto timeStart = std::chrono::system_clock::now();

	for (int t = 0; t < rows; t++) {
		auto in = inBuff + (t * cols);

		float sum = 0.f;
		for (int e = 0; e < cols; e++) {
			sum = simdOps::Mean<float, float>::update(sum, simdOps::Mean<float, float>::op(in[e], nullptr), nullptr);
		}

		outBuff[t] = sum / cols;
	}

	auto timeEnd = std::chrono::system_clock::now();
	auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
	nd4j_printf("Time: %lld us;\n", outerTime);

}

TEST_F(PlaygroundTests, test_biasAdd_1) {
	auto x = NDArrayFactory::create<float>('c', { 512, 3072 });
	auto y = NDArrayFactory::create<float>('c', { 3072 });

	std::vector<Nd4jLong> values;

	sd::ops::biasadd op;

	for (int e = 0; e < 100; e++) {
		auto timeStart = std::chrono::system_clock::now();

		op.execute({ &x, &y }, { &x });

		auto timeEnd = std::chrono::system_clock::now();
		auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
		values.emplace_back(outerTime);
	}

	std::sort(values.begin(), values.end());

	nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);
}

TEST_F(PlaygroundTests, test_bert_full_1) {
	// this test will run ONLY if this model exists
	if (sd::graph::getFileSize("/home/raver119/Downloads/BertFull/model.fb") < 0)
		return;

	auto graph = GraphExecutioner::importFromFlatBuffers("/home/raver119/Downloads/BertFull/model.fb");

	nd4j_printf("Graph successfully loaded\n", "");

	auto t = NDArrayFactory::fromNpyFile("/home/raver119/Downloads/BertFull/in0_IteratorGetNext.npy");
	auto u = NDArrayFactory::fromNpyFile("/home/raver119/Downloads/BertFull/in1_IteratorGetNext_1.npy");
	auto v = NDArrayFactory::fromNpyFile("/home/raver119/Downloads/BertFull/in2_IteratorGetNext_4.npy");
	auto z = NDArrayFactory::fromNpyFile("/home/raver119/Downloads/BertFull/out_loss-Softmax.npy");

	//graph->printOut();

	graph->tagInplaceNodes();

	graph->getVariableSpace()->putVariable(658, 0, t);
	graph->getVariableSpace()->putVariable(659, 0, u);
	graph->getVariableSpace()->putVariable(660, 0, v);

	/*
		// validating graph now
		auto status = GraphExecutioner::execute(graph);
		ASSERT_EQ(Status::OK(), status);
		ASSERT_TRUE(graph->getVariableSpace()->hasVariable(1620));

		auto array = graph->getVariableSpace()->getVariable(1620)->getNDArray();
		ASSERT_EQ(z, *array);

	*/

	sd::Environment::getInstance()->setProfiling(true);
	auto profile = GraphProfilingHelper::profile(graph, 1);

	profile->printOut();

	sd::Environment::getInstance()->setProfiling(false);
	delete profile;

	/*
		std::vector<Nd4jLong> values;

		for (int e = 0; e < 1; e++) {
			auto timeStart = std::chrono::system_clock::now();

			GraphExecutioner::execute(graph);

			auto timeEnd = std::chrono::system_clock::now();
			auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
			values.emplace_back(outerTime);
		}

		std::sort(values.begin(), values.end());

		nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);
	*/
	delete graph;
}


TEST_F(PlaygroundTests, test_bert_1) {
	// this test will run ONLY if this model exists
	if (sd::graph::getFileSize("/home/raver119/Downloads/Bert_minimal_model/bert_minimal_model.fb") < 0)
		return;

	auto graph = GraphExecutioner::importFromFlatBuffers("/home/raver119/Downloads/Bert_minimal_model/bert_minimal_model.fb");

	auto t = NDArrayFactory::fromNpyFile("/home/raver119/Downloads/Bert_minimal_model/bert_minimal_input_IteratorGetNext.numpy");
	auto u = NDArrayFactory::fromNpyFile("/home/raver119/Downloads/Bert_minimal_model/bert_minimal_input_IteratorGetNext_1.numpy");
	auto v = NDArrayFactory::fromNpyFile("/home/raver119/Downloads/Bert_minimal_model/bert_minimal_input_IteratorGetNext_4.numpy");
	auto z = NDArrayFactory::fromNpyFile("/home/raver119/Downloads/Bert_minimal_model/bert_minimal_model_output.numpy");

	//graph->printOut();

	graph->tagInplaceNodes();

	graph->getVariableSpace()->putVariable(85, 0, t);
	graph->getVariableSpace()->putVariable(86, 0, u);
	graph->getVariableSpace()->putVariable(87, 0, v);

	/*
		// validating graph now
		auto status = GraphExecutioner::execute(graph);
		ASSERT_EQ(Status::OK(), status);
		ASSERT_TRUE(graph->getVariableSpace()->hasVariable(198));

		auto array = graph->getVariableSpace()->getVariable(198)->getNDArray();
		ASSERT_EQ(z, *array);
	*/

	sd::Environment::getInstance()->setProfiling(true);
	auto profile = GraphProfilingHelper::profile(graph, 1);

	profile->printOut();

	sd::Environment::getInstance()->setProfiling(false);
	delete profile;

	/*
	std::vector<Nd4jLong> values;

	for (int e = 0; e < 1; e++) {
		auto timeStart = std::chrono::system_clock::now();

		GraphExecutioner::execute(graph);

		auto timeEnd = std::chrono::system_clock::now();
		auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
	   values.emplace_back(outerTime);
	}

	std::sort(values.begin(), values.end());

	nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);
	*/

	delete graph;
}

TEST_F(PlaygroundTests, test_bert_2) {
	// this test will run ONLY if this model exists
	if (sd::graph::getFileSize("/home/raver119/Downloads/Bert_minimal_model/bert_like_ops.fb") < 0)
		return;

	auto graph = GraphExecutioner::importFromFlatBuffers("/home/raver119/Downloads/Bert_minimal_model/bert_like_ops.fb");

	//graph->printOut();

	graph->tagInplaceNodes();


	/*
		// validating graph now
		auto status = GraphExecutioner::execute(graph);
		ASSERT_EQ(Status::OK(), status);
		ASSERT_TRUE(graph->getVariableSpace()->hasVariable(198));

		auto array = graph->getVariableSpace()->getVariable(198)->getNDArray();
		ASSERT_EQ(z, *array);
	*/

	sd::Environment::getInstance()->setProfiling(true);
	auto profile = GraphProfilingHelper::profile(graph, 1);

	profile->printOut();

	sd::Environment::getInstance()->setProfiling(false);
	delete profile;

	/*
		std::vector<Nd4jLong> values;

		for (int e = 0; e < 1; e++) {
			auto timeStart = std::chrono::system_clock::now();

			GraphExecutioner::execute(graph);

			auto timeEnd = std::chrono::system_clock::now();
			auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
			values.emplace_back(outerTime);
		}

		std::sort(values.begin(), values.end());

		nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);
	*/
	delete graph;
}

/*

TEST_F(PlaygroundTests, test_one_off_ops_1) {
	int pool = 1000;
	std::vector<NDArray*> aX(pool);
	std::vector<NDArray*> aY(pool);
	std::vector<NDArray*> aZ(pool);

	for (int e = 0; e < pool; e++) {
		aX[e] = NDArrayFactory::create_<float>('c', {512, 3072});
		aY[e] = NDArrayFactory::create_<float>('c', {3072});
		aZ[e] = NDArrayFactory::create_<float>('c', {512, 3072});

		aX[e]->assign(119 * (e+1));
		aY[e]->assign(119 * (e+3));
	}

	std::vector<Nd4jLong> values;
	Context ctx(1);

	sd::ops::biasadd op;

	for (int e = 0; e < 1000; e++) {
		auto x = aX[e < pool ? e : e % pool];
		auto y = aY[e < pool ? e : e % pool];
		auto z = aZ[e < pool ? e : e % pool];

		auto timeStart = std::chrono::system_clock::now();

		//op.execute({x, y}, {z});
		sd::ops::helpers::addBias(ctx, *x, *y, *z, false);

		auto timeEnd = std::chrono::system_clock::now();
		auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
		values.emplace_back(outerTime);
	}

	std::sort(values.begin(), values.end());

	nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);

	for (int e = 0; e < pool; e++) {
		delete aX[e];
		delete aY[e];
		delete aZ[e];
	}
}
*/

TEST_F(PlaygroundTests, test_broadcast_1) {
	int pool = 1000;
	std::vector<NDArray*> aX(pool);
	std::vector<NDArray*> aY(pool);
	std::vector<NDArray*> aZ(pool);

	for (int e = 0; e < pool; e++) {
		aX[e] = NDArrayFactory::create_<float>('c', { 512, 3072 });
		aY[e] = NDArrayFactory::create_<float>('c', { 3072 });
		aZ[e] = NDArrayFactory::create_<float>('c', { 512, 3072 });

		aX[e]->assign(119 * (e + 1));
		aY[e]->assign(119 * (e + 3));
	}

	std::vector<Nd4jLong> values;
	Context ctx(1);

	sd::ops::biasadd op;

	for (int e = 0; e < 1000; e++) {
		auto x = aX[e < pool ? e : e % pool];
		auto y = aY[e < pool ? e : e % pool];
		auto z = aZ[e < pool ? e : e % pool];

		auto timeStart = std::chrono::system_clock::now();

		//op.execute({x, y}, {z});
		sd::ops::helpers::addBias(ctx, *x, *y, *z, false);

		auto timeEnd = std::chrono::system_clock::now();
		auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
		values.emplace_back(outerTime);
	}

	std::sort(values.begin(), values.end());

	nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);

	for (int e = 0; e < pool; e++) {
		delete aX[e];
		delete aY[e];
		delete aZ[e];
	}
}


/*
TEST_F(PlaygroundTests, test_broadcast_1) {
	int pool = 500;
	std::vector<NDArray*> aX(pool);
	std::vector<NDArray*> aY(pool);
	std::vector<NDArray*> aZ(pool);

	for (int e = 0; e < pool; e++) {
		aX[e] = NDArrayFactory::create_<float>('c', {512, 3072});
		aY[e] = NDArrayFactory::create_<float>('c', {768});
		aZ[e] = NDArrayFactory::create_<float>('c', {512, 3072});

		aX[e]->assign( (e+1) / 119);
		aY[e]->assign( (e+3) / 119);
	}



	std::vector<Nd4jLong> values;

	for (int e = 0; e < 1000; e++) {
		auto x = aX[e < pool ? e : e % pool];
		auto y = aY[e < pool ? e : e % pool];
		auto z = aZ[e < pool ? e : e % pool];

		auto timeStart = std::chrono::system_clock::now();

		//x->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), *y, *z);
		x->applyTransform(transform::Tanh, *z, nullptr);

		auto timeEnd = std::chrono::system_clock::now();
		auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
		values.emplace_back(outerTime);
	}

	std::sort(values.begin(), values.end());

	nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);

	for (int e = 0; e < pool; e++) {
		delete aX[e];
		delete aY[e];
		delete aZ[e];
	}
}

*/
/*

TEST_F(PlaygroundTests, test_s_0) {
	std::vector<std::vector<Nd4jLong>> shapes = {{32, 224, 224, 3}, {32, 56, 56, 64}, {32, 7, 7, 512}};
	std::vector<int> threads = {1, 2, 4, 8, 16};

	for (auto shape: shapes) {
		for (auto t: threads) {
			sd::Environment::getInstance()->setMaxMasterThreads(t);

			auto x = NDArrayFactory::create<float>('c', shape);
			auto y = NDArrayFactory::create<float>('c', {shape[3]});
			auto z = x.ulike();

			std::vector<Nd4jLong> values;
			Context ctx(1);
			ctx.setInputArray(0, &x);
			ctx.setInputArray(1, &y);
			ctx.setOutputArray(0, &z);

			sd::ops::biasadd op;


			for (int e = 0; e < 10000; e++) {
				auto timeStart = std::chrono::system_clock::now();

				op.execute(&ctx);
				sd::ops::helpers::addBias(ctx, x, y, z, false);

				auto timeEnd = std::chrono::system_clock::now();
				auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
				values.emplace_back(outerTime);
			}

			std::sort(values.begin(), values.end());

			nd4j_printf("Shape: [%lld, %lld, %lld, %lld]; Threads: [%i]; Time: %lld us;\n", shape[0], shape[1], shape[2], shape[3], t, values[values.size() / 2]);
		}
	}
}

TEST_F(PlaygroundTests, test_s_1) {
	std::vector<std::vector<Nd4jLong>> shapes = {{32, 3, 224, 224}, {32, 64, 56, 56}, {32, 512, 7, 7}};
	std::vector<int> threads = {1, 2, 4, 8, 16};

	for (auto shape: shapes) {
		for (auto t: threads) {
			sd::Environment::getInstance()->setMaxMasterThreads(t);

			auto x = NDArrayFactory::create<float>('c', shape);
			auto y = NDArrayFactory::create<float>('c', {shape[1]});
			auto z = x.ulike();

			std::vector<Nd4jLong> values;
			Context ctx(1);
			ctx.setInputArray(0, &x);
			ctx.setInputArray(1, &y);
			ctx.setOutputArray(0, &z);

			sd::ops::biasadd op;


			for (int e = 0; e < 10000; e++) {
				auto timeStart = std::chrono::system_clock::now();

				//op.execute({&x, &y}, {&z}, {true});
				sd::ops::helpers::addBias(ctx, x, y, z, true);

				auto timeEnd = std::chrono::system_clock::now();
				auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
				values.emplace_back(outerTime);
			}

			std::sort(values.begin(), values.end());

			nd4j_printf("Shape: [%lld, %lld, %lld, %lld]; Threads: [%i]; Time: %lld us;\n", shape[0], shape[1], shape[2], shape[3], t, values[values.size() / 2]);
		}
	}
}
*/

/*
TEST_F(PlaygroundTests, test_s_0) {
	auto x = NDArrayFactory::create<float>('c', {32, 112, 112, 16});
	auto y = NDArrayFactory::create<float>('c', {16});
	auto z = x.ulike();

	std::vector<Nd4jLong> values;
	Context ctx(1);
	ctx.setInputArray(0, &x);
	ctx.setInputArray(1, &y);
	ctx.setOutputArray(0, &z);

	sd::ops::biasadd op;


	for (int e = 0; e < 10000; e++) {
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

	sd::ops::concat op;
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

	  sd::ops::ConvolutionUtils::calcOutSizeDeconv3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, 0);

	  printf("!!%i, %i, %i\n", oD,oH,oW);

	  NDArray col('c', {bS, iC, kD, kH, kW, iD, iH, iW}, sd::DataType::DOUBLE);
	  NDArray vol('c', {bS, iC, oD, oH, oW}, sd::DataType::DOUBLE);

	  col = 3.77;
	  vol = -10.33;

	  auto variableSpace = new VariableSpace();
	  auto block = new Context(1, variableSpace, false);  // not-in-place

	  auto timeStart = std::chrono::system_clock::now();
	  sd::ops::ConvolutionUtils::col2vol(*block, col, vol, sD, sH, sW, pD, pH, pW, dD, dH, dW);
	  auto timeEnd = std::chrono::system_clock::now();
	  auto time = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

	  printf("time: %i \n", time);

	  delete block;
	  delete variableSpace;
  }

  TEST_F(PlaygroundTests, my) {

	  int bS=32, iD=32,iH=64,iW=64,  iC=128,  kD=2,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=2,dH=2,dW=2;
	  int       oD,oH,oW;

	  // sd::ops::ConvolutionUtils::calcOutSizeDeconv3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, 0);
	  sd::ops::ConvolutionUtils::calcOutSizeDeconv2D(oH, oW, kH, kW, sH, sW, pH, pW,dH, dW, iH, iW, 0);

	  printf("!!%i, %i, %i\n", oD,oH,oW);

	  // NDArray col('c', {bS, iC, kD, kH, kW, iD, iH, iW}, sd::DataType::DOUBLE);
	  // NDArray vol('c', {bS, iC, oD, oH, oW}, sd::DataType::DOUBLE);
	  NDArray col('c', {bS, iC, kH, kW, iH, iW}, sd::DataType::DOUBLE);
	  NDArray im('c', {bS, iC, oH, oW}, sd::DataType::DOUBLE);

	  col = 3.77;
	  // vol = -10.33;
	  im = -10.33;

	  auto variableSpace = new VariableSpace();
	  auto block = new Context(1, variableSpace, false);  // not-in-place

	  auto timeStart = std::chrono::system_clock::now();
	  // sd::ops::ConvolutionUtils::col2vol(*block, col, vol, sD, sH, sW, pD, pH, pW, dD, dH, dW);
	  sd::ops::helpers::col2im(*col.getContext(), col, im, sH, sW, pH, pW, iH, iW, dH, dW);
	  auto timeEnd = std::chrono::system_clock::now();
	  auto time = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

	  printf("time: %i \n", time);

	  delete block;
	  delete variableSpace;
  }

  TEST_F(PlaygroundTests, my) {

	  int N = 100;
	  int bS=16, iH=128,iW=128,  iC=32,oC=64,  kH=4,kW=4,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
	  int        oH=128,oW=128;

	  int paddingMode = 1;             // 1-SAME, 0-VALID;
	  int dataFormat  = 1;             // 1-NHWC, 0-NCHW

	  // NDArray input('c', {bS, iC, iH, iW}, sd::DataType::FLOAT32);
	  // NDArray output('c', {bS, oC, oH, oW}, sd::DataType::FLOAT32);
	  NDArray input('c', {bS, iH, iW, iC}, sd::DataType::FLOAT32);
	  NDArray output('c', {bS, oH, oW, oC}, sd::DataType::FLOAT32);
	  // NDArray weights('c', {kH, kW, iC, oC}, sd::DataType::FLOAT32);    // permute [kH, kW, iC, oC] -> [oC, iC, kH, kW]
	  NDArray weights('c', {oC, iC, kH, kW}, sd::DataType::FLOAT32);
	  NDArray bias('c', {oC}, sd::DataType::FLOAT32);

	  input = 5.;
	  weights = 3.;
	  bias = 1.;

	  sd::ops::conv2d op;
	  auto err = op.execute({&input, &weights, &bias}, {&output}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});

	  auto timeStart = std::chrono::system_clock::now();
	  for (int i = 0; i < N; ++i)
		  err = op.execute({&input, &weights, &bias}, {&output}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
	  auto timeEnd = std::chrono::system_clock::now();
	  auto time = std::chrono::duration_cast<std::chrono::microseconds> ((timeEnd - timeStart) / N).count();

	  printf("time: %i \n", time);
  }


  */


#endif