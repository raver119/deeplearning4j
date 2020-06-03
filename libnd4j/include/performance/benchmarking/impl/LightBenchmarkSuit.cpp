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
// @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>
#include "performance/benchmarking/LightBenchmarkSuit.h"

#ifdef RELEASE_BUILD
#define WARMUP 5
#define NUM_ITER 100

#else

#define WARMUP 5
#define NUM_ITER 100

#endif

namespace sd {

    template <typename T>
    static std::string transformBenchmark() {
        std::string output;
        output += "transformBenchmark " + DataTypeUtils::asString(DataTypeUtils::fromT<T>());

        BenchmarkHelper helper(WARMUP, NUM_ITER);
        IntPowerParameters length("length", 2, 8, 20, 4);      //2^8, 2^12, 2^16, 2^20 - 4MB
        BoolParameters inplace("inplace");

        ParametersBatch batch({&length, &inplace});

        auto generator = PARAMETRIC_XZ() {
            auto arr = NDArrayFactory::create_<T>('c', {p.getIntParam("length")});
            arr.assign(1.0);
            x.push_back(arr.dup());
            if(p.getIntParam("inplace") == 1){
                z.push_back(arr);
            } else {
                z.push_back(NDArrayFactory::create<T>('c', {p.getIntParam("length")}).dup());
            }
        };

        ScalarBenchmark sbRelu(scalar::Ops::RELU, "RELU");
        sbRelu.setY(NDArrayFactory::create<T>(0.0).dup());

        TransformBenchmark tbSigmoid(transform::StrictOps::Sigmoid, "sigmoid");
        //TransformBenchmark tbSoftmax(transform::StrictOps::SoftMax, "softmax");

        output += helper.runOperationSuit(&sbRelu, generator, batch, "RELU");
        output += helper.runOperationSuit(&tbSigmoid, generator, batch, "Sigmoid");
        //output += helper.runOperationSuit(&tbSigmoid, generator, batch, "Softmax");

        return output;
    }

    template <typename T>
    static std::string scalarBenchmark() {
        std::string output;
        output += "scalarBenchmark " + DataTypeUtils::asString(DataTypeUtils::fromT<T>());

        BenchmarkHelper helper(WARMUP, NUM_ITER);

        IntPowerParameters length("length", 2, 8, 20, 4);      //2^8, 2^12, 2^16, 2^20
        BoolParameters inplace("inplace");

        ParametersBatch batch({&length, &inplace});

        auto generator = PARAMETRIC_XZ() {
            auto arr = NDArrayFactory::create<T>('c', {p.getIntParam("length")});
            arr.assign(1.0);
            x.push_back(arr.dup());
            if(p.getIntParam("inplace") == 1){
                z.push_back(arr);
            } else {
                z.push_back(NDArrayFactory::create_<T>('c', {p.getIntParam("length")}));
            }
        };

        ScalarBenchmark sbAdd(scalar::Ops::Add, "sAdd");
        ScalarBenchmark sbDiv(scalar::Ops::Divide, "sDiv");
        ScalarBenchmark sbPow(scalar::Ops::Pow, "sPow");


        sbAdd.setY(NDArrayFactory::create_<T>(3.14159265359));
        sbDiv.setY(NDArrayFactory::create_<T>(3.14159265359));
        sbPow.setY(NDArrayFactory::create_<T>(3.14159265359));


        output += helper.runOperationSuit(&sbAdd, generator, batch, "Scalar Addition - x.add(3.14159265359)");
        output += helper.runOperationSuit(&sbDiv, generator, batch, "Scalar Division - x.div(3.14159265359)");
        output += helper.runOperationSuit(&sbPow, generator, batch, "Scalar Power - x.pow(3.14159265359)");

        return output;
    }


    template <typename T>
    static std::string pairwiseBenchmark() {
        std::string output;
        output += "pairwiseBenchmark " + DataTypeUtils::asString(DataTypeUtils::fromT<T>());

        BenchmarkHelper helper(WARMUP, NUM_ITER);
        IntPowerParameters length("length", 2, 8, 20, 4);      //2^4 to 2^20 in steps of 4 - 2^4, 2^8, 2^16, 2^20
        BoolParameters inplace("inplace");

        ParametersBatch batch({&length, &inplace});

        auto generator = PARAMETRIC_XYZ() {
            auto arr1 = NDArrayFactory::create<T>('c', {p.getIntParam("length")});
            auto arr2 = NDArrayFactory::create<T>('c', {p.getIntParam("length")});
            x.push_back(arr1.dup());
            y.push_back(arr2.dup());
            if(p.getIntParam("inplace") == 1){
                z.push_back(arr1);
            } else {
                z.push_back(NDArrayFactory::create<T>('c', {p.getIntParam("length")}).dup());
            }
        };

        PairwiseBenchmark pb1(pairwise::Ops::Add, "Add");
        output += helper.runOperationSuit(&pb1, generator, batch, "Pairwise Add");

        PairwiseBenchmark pb2(pairwise::Ops::Divide, "Divide");
        output += helper.runOperationSuit(&pb2, generator, batch, "Pairwise Divide");

        return output;
    }

    static std::string mismatchedOrderAssign() {
        std::string output;
        BenchmarkHelper helper(WARMUP, NUM_ITER);

        IntPowerParameters rows("rows", 2, 8, 20, 4);      //2^8, 2^12, 2^16, 2^20
        BoolParameters cf("cf");

        ParametersBatch batch({&rows, &cf});

        auto generator = PARAMETRIC_XZ() {
            int numElements = 4194304;    //2^24
            int rows = p.getIntParam("rows");
            int cols = numElements / rows;
            bool c = p.getIntParam("cf");

            auto arr = NDArrayFactory::create_<float>(c ? 'c' : 'f', {rows, cols});
            auto arr2 = NDArrayFactory::create_<float>(c ? 'f' : 'c', {rows, cols});
            x.push_back(arr.dup());
            z.push_back(arr2.dup());
        };

        TransformBenchmark tb(transform::AnyOps::Assign, "assign");
        output += helper.runOperationSuit(&tb, generator, batch, "C->F and F->C Assign F32");

        //Also test: NCHW to NHWC and back
        BoolParameters nchw("nchw");
        int mb = 8;
        int hw = 64;
        int c = 3;
        ParametersBatch batch2({&nchw});
        auto generator2 = PARAMETRIC_XZ() {
            bool nchw = p.getIntParam("nchw");

            if(nchw) {
                auto orig = NDArrayFactory::create_<float>('c', {mb, c, hw, hw});
                orig.permutei({0,2,3,1});
                x.push_back(orig.dup());
                z.push_back(NDArrayFactory::create<float>('c', {mb, hw, hw, c}).dup());
            } else {
                auto orig = NDArrayFactory::create_<float>('c', {mb, hw, hw, c});
                orig.permutei({0,3,1,2});
                x.push_back(orig.dup());
                z.push_back(NDArrayFactory::create_<float>('c', {mb, c, hw, hw}).dup());
            }
        };

        TransformBenchmark tb2(transform::AnyOps::Assign, "assign_nchw");
        output += helper.runOperationSuit(&tb2, generator2, batch2, "nchw->nhwc and nhwc->nchw Assign FP32");
        return output;
    }

    template <typename T>
    static std::string gemmBenchmark() {
        std::string output;
        output += "gemm " + DataTypeUtils::asString(DataTypeUtils::fromT<T>());
        BenchmarkHelper helper(WARMUP, NUM_ITER);

        for (int o = 0; o <= 1; o++) {
            char resultOrder = (o == 0 ? 'f' : 'c');
            IntPowerParameters sz("sz", 2, 4, 10, 2);          //2^4=16, ..., 2^10=1024   ->  4 elements

            ParametersBatch b({&sz});

            auto generator = PARAMETRIC_XYZ() {
                auto a = p.getIntParam("sz");
                auto b = p.getIntParam("sz");
                auto c = p.getIntParam("sz");
                std::vector<Nd4jLong> shapeA;
                std::vector<Nd4jLong> shapeB;
                shapeA = {a, b};
                shapeB = {b, c};
                auto A = NDArrayFactory::create<T>('c', shapeA);
                auto B = NDArrayFactory::create<T>('c', shapeB);
                auto C = NDArrayFactory::create<T>(resultOrder, {a, c});

                x.push_back(A.dup());
                y.push_back(B.dup());
                z.push_back(C.dup());
            };

            std::string n;
            n += "Gemm - cOrder=";
            n += resultOrder;

            MatrixBenchmark mb(1.0, 0.0, false, false, n);

            output += helper.runOperationSuit(&mb, generator, b, n.c_str());
        }

        return output;
    }

    template <typename T>
    static std::string reduceFullBenchmark() {
        std::string output;
        output += "reduceFullBenchmark " + DataTypeUtils::asString(DataTypeUtils::fromT<T>());

        BenchmarkHelper helper(WARMUP, NUM_ITER);

        IntPowerParameters length("length", 2, 8, 20, 4);      //2^8, 2^12, 2^16, 2^20

        ParametersBatch batch({&length});

        auto generator = PARAMETRIC_XYZ() {
            auto arr = NDArrayFactory::create<T>('c', {p.getIntParam("length")});

            x.push_back(arr.dup());
            y.push_back(nullptr);
            z.push_back(NDArrayFactory::create<T>(0.0f).dup());
        };

        ReductionBenchmark rbSum(reduce::SameOps::Sum, "sum");
        ReductionBenchmark rbProd(reduce::SameOps::Prod, "prod");
        ReductionBenchmark rbMax(reduce::SameOps::Max, "max");

        output += helper.runOperationSuit(&rbSum, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, "Sum - Full Array Reduction");
        output += helper.runOperationSuit(&rbProd, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, "Product - Full Array Reduction");
        output += helper.runOperationSuit(&rbMax, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, "Maximum - Full Array Reduction");

        //Index reduction
        sd::ops::argmax opArgmax;
        DeclarableBenchmark dbArgmax(opArgmax, "Argmax");
        auto generator3 = PARAMETRIC_D(){
            auto ctx = new Context(1);

            ctx->setInputArray(0, NDArrayFactory::create<T>('c', {p.getIntParam("length")}).dup(), true);
            ctx->setInputArray(1, NDArrayFactory::create<Nd4jLong>((Nd4jLong)0), true);
            ctx->setOutputArray(0, NDArrayFactory::create<Nd4jLong>(0), true);

            return ctx;
        };
        output += helper.runOperationSuit(&dbArgmax, generator3, batch, "Argmax Full Array Reduction");
        return output;
    }

    template <typename T>
    static std::string reduceDimBenchmark(){
        std::string output;
        output += "reduceDimBenchmark " + DataTypeUtils::asString(DataTypeUtils::fromT<T>());

        BenchmarkHelper helper(WARMUP, NUM_ITER);

        int length[] = {1024*1024};
        int pow[] = {10};

        for( int i=0; i<1; i++ ){
            IntPowerParameters rows("rows", 2, 0, pow[i], 2);
            BoolParameters dim("dim");


            ParametersBatch batch({&rows, &dim});

            auto generator = PARAMETRIC_XYZ() {
                int rows = p.getIntParam("rows");
                int cols = length[i] / rows;
                int dim = p.getIntParam("dim");
                auto arr = NDArrayFactory::create<T>('c', {rows, cols});


                x.push_back(arr.dup());
                y.push_back(NDArrayFactory::create<Nd4jLong>(dim).dup());

                NDArray result;
                if(dim == 0){
                    result = NDArrayFactory::create<T>('c', {cols});
                } else {
                    result = NDArrayFactory::create<T>('c', {rows});
                }
                z.push_back(result.dup());
            };

            ReductionBenchmark rbSum(reduce::SameOps::Sum, "sum");
            ReductionBenchmark rbMax(reduce::SameOps::Max, "max");

            std::string s1("Sum Along Dimension - ");
            s1 += std::to_string(length[i]);
            std::string s3("Maximum Along Dimension - ");
            s3 += std::to_string(length[i]);

            output += helper.runOperationSuit(&rbSum, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, s1.c_str());
            output += helper.runOperationSuit(&rbMax, (const std::function<void (Parameters &, ResultSet &, ResultSet &, ResultSet &)>)(generator), batch, s3.c_str());



            auto generator3 = PARAMETRIC_D(){
                auto ctx = new Context(1);
                int rows = p.getIntParam("rows");
                int cols = length[i] / rows;
                int dim = p.getIntParam("dim");
                auto arr = NDArrayFactory::create_<T>('c', {rows, cols});

                auto dimArg = new Nd4jLong[1];
                dimArg[0] = dim;
                ctx->setIArguments(dimArg, 1);
                delete[] dimArg;

                ctx->setInputArray(0, arr, true);

                NDArray result;
                if(dim == 0){
                    result = NDArrayFactory::create<Nd4jLong>('c', {cols});
                } else {
                    result = NDArrayFactory::create<Nd4jLong>('c', {rows});
                }
                ctx->setOutputArray(0, result.dup(), true);
                return ctx;
            };

            std::string s5("Argmax Along Dimension - ");
            s5 += std::to_string(length[i]);

            sd::ops::argmax opArgmax;
            DeclarableBenchmark dbArgmax(opArgmax, "Argmax");
            output += helper.runOperationSuit(&dbArgmax, generator3, batch, s5.c_str());
        }
        return output;
    }

    template <typename T>
    static std::string conv2d(){
        std::string output;
        output += "conv2d " + DataTypeUtils::asString(DataTypeUtils::fromT<T>());
        BenchmarkHelper helper(WARMUP, NUM_ITER);

        //Convolution2D op
        BoolParameters nhwc("nhwc");
        PredefinedParameters k("k", {2, 3});

        ParametersBatch batch({&nhwc, &k});
        sd::ops::conv2d conv2d;
        DeclarableBenchmark benchmark(conv2d, "conv2d");

        int hw = 64;

        auto generator = PARAMETRIC_D() {
            auto ctx = new Context(1);
            int n = p.getIntParam("nhwc");
            int khw = p.getIntParam("k");

            if (n == 0) {
                auto input = NDArrayFactory::create<T>('c', {8, 3, hw, hw});
                auto output = NDArrayFactory::create<T>('c', {8, 3, hw, hw});
                ctx->setInputArray(0, input, true);
                ctx->setOutputArray(0, output, true);
            } else {
                auto input = NDArrayFactory::create<T>('c', {8, hw, hw, 3});
                auto output = NDArrayFactory::create<T>('c', {8, hw, hw, 3});
                ctx->setInputArray(0, input, true);
                ctx->setOutputArray(0, output, true);
            }

            auto b = NDArrayFactory::create<T>('c', {3});
            auto w = NDArrayFactory::create<T>('c', {khw, khw, 3, 3});   // [kH, kW, iC, oC] always

            ctx->setInputArray(1, w, true);
            ctx->setInputArray(2, b, true);

            auto args = new Nd4jLong[10];
            args[0] = args[1] = khw; //Kernel
            args[2] = args[3] = 1;//Stride
            args[4] = args[5] = 0;  //Pad
            args[6] = args[7] = 1;  //Dilation
            args[8] = 1;     //SAME
            args[9] = n;//0-nchw, 1=nhwc
            ctx->setIArguments(args, 10);
            delete[] args;

            return ctx;
        };

        output += helper.runOperationSuit(&benchmark, generator, batch, "Conv2d");
        return output;
    }

    template <typename T>
    static std::string pool2d() {
        std::string output;
        output += "pool2d " + DataTypeUtils::asString(DataTypeUtils::fromT<T>());
        BenchmarkHelper helper(WARMUP, NUM_ITER);

        //Convolution2D op
        BoolParameters nhwc("nhwc");
        PredefinedParameters k("k", {2, 3});

        ParametersBatch batch({&nhwc, &k});

        int c = 3;
        int hw = 64;

        auto generator = PARAMETRIC_D() {
            auto ctx = new Context(1);
            int n = p.getIntParam("nhwc");
            int khw = p.getIntParam("k");

            if (n == 0) {
                auto input = NDArrayFactory::create<T>('c', {8, c, hw, hw});
                auto output = NDArrayFactory::create<T>('c', {8, c, hw, hw});
                ctx->setInputArray(0, input.dup(), true);
                ctx->setOutputArray(0, output.dup(), true);
            } else {
                auto input = NDArrayFactory::create<T>('c', {8, hw, hw, c});
                auto output = NDArrayFactory::create<T>('c', {8, hw, hw, c});
                ctx->setInputArray(0, input.dup(), true);
                ctx->setOutputArray(0, output.dup(), true);
            }

            std::vector<Nd4jLong> args(11);// = new Nd4jLong[11];
            args[0] = args[1] = khw; //Kernel
            args[2] = args[3] = 1;//Stride
            args[4] = args[5] = 0;  //Pad
            args[6] = args[7] = 1;  //Dilation
            args[8] = 1;     //SAME
            args[9] = 0;     //Divisor mode - 0 = exclude padding in divisor
            args[10] = n;//0-nchw, 1=nhwc
            ctx->setIArguments(args.data(), 11);

            return ctx;
        };

        sd::ops::avgpool2d avgpool2d;
        DeclarableBenchmark benchmark1(avgpool2d, "avgpool");
        output += helper.runOperationSuit(&benchmark1, generator, batch, "Average Pool 2d");

        sd::ops::maxpool2d maxpool2d;
        DeclarableBenchmark benchmark2(maxpool2d, "maxpool");
        output += helper.runOperationSuit(&benchmark2, generator, batch, "Max Pool 2d");
        return output;
    }

    template <typename T>
    static std::string lstmBenchmark() {
        std::string output;
        output += "lstm " + DataTypeUtils::asString(DataTypeUtils::fromT<T>());
        BenchmarkHelper helper(WARMUP, NUM_ITER);

        BoolParameters format("format");    //0=TNS=[seqLen,mb,size]; 1=NST=[mb,size,seqLen]
        PredefinedParameters mb("mb", {1, 8});
        int n = 128;

        ParametersBatch batch({&format, &mb});
        sd::ops::lstmBlock lstmBlock;
        DeclarableBenchmark benchmark(lstmBlock, "lstm");

        int seqLength = 8;

        auto generator = PARAMETRIC_D() {
            auto ctx = new Context(1);
            int f = p.getIntParam("format");
            int m = p.getIntParam("mb");

            Nd4jLong l = 0;
            ctx->setInputArray(0, NDArrayFactory::create<Nd4jLong>(l).dup(), true);  //Max TS length (unused)


            if (f == 0) {
                //TNS format
                ctx->setInputArray(1, NDArrayFactory::create<T>('c', {seqLength, m, n}).dup(), true);     //x
                ctx->setOutputArray(0, NDArrayFactory::create<T>('c', {seqLength, m, n}).dup(), true);    //i
                ctx->setOutputArray(1, NDArrayFactory::create<T>('c', {seqLength, m, n}).dup(), true);    //c
                ctx->setOutputArray(2, NDArrayFactory::create<T>('c', {seqLength, m, n}).dup(), true);    //f
                ctx->setOutputArray(3, NDArrayFactory::create<T>('c', {seqLength, m, n}).dup(), true);    //o
                ctx->setOutputArray(4, NDArrayFactory::create<T>('c', {seqLength, m, n}).dup(), true);    //z
                ctx->setOutputArray(5, NDArrayFactory::create<T>('c', {seqLength, m, n}).dup(), true);    //h
                ctx->setOutputArray(6, NDArrayFactory::create<T>('c', {seqLength, m, n}).dup(), true);    //y
            } else {
                //NST format
                ctx->setInputArray(1, NDArrayFactory::create<T>('f', {m, n, seqLength}).dup(), true);     //x
                ctx->setOutputArray(0, NDArrayFactory::create<T>('f', {m, n, seqLength}).dup(), true);    //i
                ctx->setOutputArray(1, NDArrayFactory::create<T>('f', {m, n, seqLength}).dup(), true);    //c
                ctx->setOutputArray(2, NDArrayFactory::create<T>('f', {m, n, seqLength}).dup(), true);    //f
                ctx->setOutputArray(3, NDArrayFactory::create<T>('f', {m, n, seqLength}).dup(), true);    //o
                ctx->setOutputArray(4, NDArrayFactory::create<T>('f', {m, n, seqLength}).dup(), true);    //z
                ctx->setOutputArray(5, NDArrayFactory::create<T>('f', {m, n, seqLength}).dup(), true);    //h
                ctx->setOutputArray(6, NDArrayFactory::create<T>('f', {m, n, seqLength}).dup(), true);    //y
            }

            auto cLast = NDArrayFactory::create<T>('c', {m, n});
            auto yLast = NDArrayFactory::create<T>('c', {m, n});
            auto W = NDArrayFactory::create<T>('c', {2 * n, 4 * n});
            auto Wci = NDArrayFactory::create<T>('c', {n});
            auto Wcf = NDArrayFactory::create<T>('c', {n});
            auto Wco = NDArrayFactory::create<T>('c', {n});
            auto b = NDArrayFactory::create<T>('c', {4 * n});

            ctx->setInputArray(2, cLast.dup(), true);
            ctx->setInputArray(3, yLast.dup(), true);
            ctx->setInputArray(4, W.dup(), true);
            ctx->setInputArray(5, Wci.dup(), true);
            ctx->setInputArray(6, Wcf.dup(), true);
            ctx->setInputArray(7, Wco.dup(), true);
            ctx->setInputArray(8, b.dup(), true);

            std::vector<Nd4jLong> iargs(2);
            iargs[0] = 0;   //No peephole
            iargs[1] = f;
            ctx->setIArguments(iargs.data(), 2);

            std::vector<double> targs(2); 
            targs[0] = 1.0; //forget bias
            targs[1] = 0.0; //cell clipping value
            ctx->setTArguments(targs.data(), 2);
            return ctx;
        };

        output += helper.runOperationSuit(&benchmark, generator, batch, "LSTMBlock");
        return output;
    }

    static std::string broadcast2d() {
        std::string output;
        BenchmarkHelper helper(WARMUP, NUM_ITER);

        int rows = 65536;
        IntPowerParameters cols("cols", 2, 2, 12, 4);      //2^2 to 2^12 in steps of 2 - 2^1=2, ..., 2^10=1024
        BoolParameters axis("axis");
        BoolParameters inplace("inplace");

        ParametersBatch batch({&cols, &axis, &inplace});

        auto generator = PARAMETRIC_D() {
            auto a = p.getIntParam("axis");
            auto arr = NDArrayFactory::create<float>('c', {rows, p.getIntParam("cols")});

            auto ctx = new Context(1);
            ctx->setInputArray(0, arr, true);
            if(a == 0){
                ctx->setInputArray(1, NDArrayFactory::create<float>('c', {rows, 1}).dup(), true);
            } else {
                ctx->setInputArray(1, NDArrayFactory::create<float>('c', {1, p.getIntParam("cols")}).dup(), true);
            }
            if (p.getIntParam("inplace") == 1) {
                ctx->setOutputArray(0, arr);
                ctx->markInplace(true);
            } else {
                ctx->setOutputArray(0, NDArrayFactory::create<float>('c', {rows, p.getIntParam("cols")}).dup(), true);
            }
            return ctx;
        };

        std::string s("add");
        sd::ops::add op;
        DeclarableBenchmark benchmark(op, "add");
        output += helper.runOperationSuit(&benchmark, generator, batch, "Broadcast (Custom) Add - 2d");
        return output;
    }

    std::string LightBenchmarkSuit::runSuit() {
#ifdef RELEASE_BUILD
        std::vector<sd::DataType> dtypes({sd::DataType::FLOAT32, sd::DataType::HALF});
#else
        std::vector<sd::DataType> dtypes({sd::DataType::FLOAT32});
#endif

        std::string result;

        for (auto t:dtypes) {
            nd4j_printf("Running LightBenchmarkSuite.transformBenchmark [%s]\n", DataTypeUtils::asString(t).c_str());
            BUILD_SINGLE_SELECTOR(t, result += transformBenchmark, (), LIBND4J_TYPES);

            nd4j_printf("Running LightBenchmarkSuite.scalarBenchmark [%s]\n", DataTypeUtils::asString(t).c_str());
            BUILD_SINGLE_SELECTOR(t, result += scalarBenchmark, (), LIBND4J_TYPES);

            nd4j_printf("Running LightBenchmarkSuite.pairwiseBenchmark [%s]\n", DataTypeUtils::asString(t).c_str());
            BUILD_SINGLE_SELECTOR(t, result += pairwiseBenchmark, (), LIBND4J_TYPES);

            nd4j_printf("Running LightBenchmarkSuite.reduceFullBenchmark [%s]\n", DataTypeUtils::asString(t).c_str());
            BUILD_SINGLE_SELECTOR(t, result += reduceFullBenchmark, (), LIBND4J_TYPES);

            nd4j_printf("Running LightBenchmarkSuite.reduceDimBenchmark [%s]\n", DataTypeUtils::asString(t).c_str());
            BUILD_SINGLE_SELECTOR(t, result += reduceDimBenchmark, (), LIBND4J_TYPES);

            nd4j_printf("Running LightBenchmarkSuite.gemmBenchmark [%s]\n", DataTypeUtils::asString(t).c_str());
            BUILD_SINGLE_SELECTOR(t, result += gemmBenchmark, (), LIBND4J_TYPES);

            nd4j_printf("Running LightBenchmarkSuite.conv2d [%s]\n", DataTypeUtils::asString(t).c_str());
            BUILD_SINGLE_SELECTOR(t, result += conv2d, (), LIBND4J_TYPES);

            nd4j_printf("Running LightBenchmarkSuite.pool2d [%s]\n", DataTypeUtils::asString(t).c_str());
            BUILD_SINGLE_SELECTOR(t, result += pool2d, (), LIBND4J_TYPES);

            nd4j_printf("Running LightBenchmarkSuite.lstmBenchmark [%s]\n", DataTypeUtils::asString(t).c_str());
            BUILD_SINGLE_SELECTOR(t, result += lstmBenchmark, (), LIBND4J_TYPES);

        }

        nd4j_printf("Running LightBenchmarkSuite.broadcast2d\n", "");
        result += broadcast2d();
        nd4j_printf("Running LightBenchmarkSuite.mismatchedOrderAssign\n", "");
        result += mismatchedOrderAssign();

        return result;
    }
}