/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
 // #include <array/NDArrayList.h>
#include <memory>
using namespace nd4j;


class MyTests : public testing::Test {
public:

    MyTests() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(MyTests, Test_Create_Empty_1) {
    auto empty = NDArrayFactory::empty_<float>();
    ASSERT_TRUE(empty->isEmpty());

    ASSERT_EQ(0, empty->lengthOf());
    ASSERT_TRUE(empty->buffer() == nullptr);

    ASSERT_TRUE(shape::isEmpty(empty->shapeInfo()));

    delete empty;
}

TEST_F(MyTests, Test_Create_Empty_2) {
    auto empty = NDArrayFactory::empty<float>();
    ASSERT_TRUE(empty.isEmpty());

    ASSERT_EQ(0, empty.lengthOf());
    ASSERT_TRUE(empty.buffer() == nullptr);

    ASSERT_TRUE(shape::isEmpty(empty.shapeInfo()));
    ASSERT_TRUE(empty.isEmpty());
}

TEST_F(MyTests, Test_Concat_1) {
    //    auto empty = NDArrayFactory::empty_<float>();
    auto empty = new NDArray('c', { 0 }, nd4j::DataType::FLOAT32);//NDArrayFactory::create_<float>('c', {(Nd4jLong)0}};
    auto vector = NDArrayFactory::create_<float>('c', { 1 }, { 1.0f });

    ASSERT_TRUE(empty->isEmpty());

    nd4j::ops::concat op;
    auto result = op.execute({ empty, vector }, {}, { 0 });
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    //    z->printShapeInfo("z shape");
    //    z->printIndexedBuffer("z buffr");

    ASSERT_EQ(*vector, *z);

    delete empty;
    delete vector;
    delete result;
}


TEST_F(MyTests, Test_Concat_2) {
    auto empty = new NDArray('c', { 0 }, nd4j::DataType::FLOAT32); //NDArrayFactory::empty_<float>();
    auto scalar1 = NDArrayFactory::create_<float>('c', { 1 }, { 1.0f });
    auto scalar2 = NDArrayFactory::create_<float>('c', { 1 }, { 2.0f });
    auto exp = NDArrayFactory::create<float>('c', { 2 }, { 1.f, 2.f });

    ASSERT_TRUE(empty->isEmpty());

    nd4j::ops::concat op;
    auto result = op.execute({ empty, scalar1, scalar2 }, {}, { 0 });
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    //    z->printShapeInfo("z shape");
    //    z->printIndexedBuffer("z buffr");

    ASSERT_EQ(exp, *z);

    delete empty;
    delete scalar1;
    delete scalar2;
    delete result;
}

TEST_F(MyTests, Test_Concat_3) {
    auto empty = NDArrayFactory::empty<float>(); //NDArrayFactory::empty_<float>();
    auto scalar1 = NDArrayFactory::create<float>(1.0f);
    auto scalar2 = NDArrayFactory::create<float>(2.0f);
    auto exp = NDArrayFactory::create<float>('c', { 2 }, { 1.f, 2.f });

    ASSERT_TRUE(empty.isEmpty());

    nd4j::ops::concat op;
    auto result = op.execute({ &empty, &scalar1, &scalar2 }, {}, { 0 });
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(exp, *z);

    delete result;
}

TEST_F(MyTests, Test_Concat_4) {
    auto empty = NDArrayFactory::empty<float>(); //NDArrayFactory::empty_<float>();
    auto scalar1 = NDArrayFactory::create<float>(1.0f);
    auto scalar2 = NDArrayFactory::create<float>(2.0f);
    auto exp = NDArrayFactory::create<float>('c', { 2 }, { 1.f, 2.f });

    ASSERT_TRUE(empty.isEmpty());

    nd4j::ops::concat op;
    auto result = op.execute({ &scalar1, &empty, &scalar2 }, {}, { 0 });
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(exp, *z);

    delete result;
}

TEST_F(MyTests, Test_Reshape_1) {
    auto vector = NDArrayFactory::create<float>('c', { 1 }, { 119.0f });
    auto exp = NDArrayFactory::create<float>(119.f);
    auto empty = NDArrayFactory::empty_<int>();

    nd4j::ops::reshape op;
    auto result = op.execute({ &vector, empty }, {}, {});

    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(exp, *result->at(0));

    delete empty;
    delete result;
}

TEST_F(MyTests, Test_Reshape_2) {
    auto vector = NDArrayFactory::create<float>('c', { 1 }, { 119.0f });
    auto exp = NDArrayFactory::create<float>(119.0f);
    auto empty = NDArrayFactory::empty_<Nd4jLong>();

    nd4j::ops::reshape op;
    auto result = op.execute({ &vector, empty }, {}, {}, {}, true);

    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(exp, *result->at(0));
    ASSERT_EQ(exp, vector);

    delete empty;
    delete result;
}

TEST_F(MyTests, Test_Reshape_3) {
    auto x = NDArrayFactory::create<float>('c', { 1, 0, 0, 2 });
    auto y = NDArrayFactory::create<int>('c', { 2 }, { 10, 0 });
    auto e = NDArrayFactory::create<float>('c', { 10, 0 });

    nd4j::ops::reshape op;
    auto result = op.execute({ &x, &y }, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(MyTests, Test_dup_1) {
    auto empty = NDArrayFactory::empty<int>();
    auto dup = empty.dup();

    ASSERT_TRUE(dup->isEmpty());
    ASSERT_EQ(empty, *dup);

    delete dup;
}

TEST_F(MyTests, test_empty_scatter_1) {
    auto x = NDArrayFactory::create<float>('c', { 5 });
    auto indices = NDArrayFactory::create<int>('c', { 0 });
    auto updates = NDArrayFactory::create<float>('c', { 0 });

    x.linspace(1.0f);

    nd4j::ops::scatter_upd op;
    auto result = op.execute({ &x, &indices, &updates }, {}, {}, { true });
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(x, *z);

    delete result;
}

TEST_F(MyTests, test_empty_scatter_2) {
    auto x = NDArrayFactory::create<float>('c', { 5 });
    auto z = NDArrayFactory::create<float>('c', { 5 });
    auto indices = NDArrayFactory::create<int>('c', { 0 });
    auto updates = NDArrayFactory::create<float>('c', { 0 });

    x.linspace(1.0f);

    Context ctx(1);
    ctx.setInputArray(0, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo());
    ctx.setInputArray(1, indices.buffer(), indices.shapeInfo(), indices.specialBuffer(), indices.specialShapeInfo());
    ctx.setInputArray(2, updates.buffer(), updates.shapeInfo(), updates.specialBuffer(), updates.specialShapeInfo());
    ctx.setOutputArray(0, z.buffer(), z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo());
    bool args[] = { true };
    ctx.setBArguments(args, 1);

    nd4j::ops::scatter_upd op;
    auto result = op.execute(&ctx);
    ASSERT_EQ(Status::OK(), result);

    ASSERT_EQ(x, z);
}

TEST_F(MyTests, test_shaped_empty_1) {
    auto empty = NDArrayFactory::create<float>('c', { 2, 0, 3 });
    std::vector<Nd4jLong> shape = { 2, 0, 3 };

    ASSERT_EQ(nd4j::DataType::FLOAT32, empty.dataType());
    ASSERT_EQ(0, empty.lengthOf());
    ASSERT_TRUE(empty.isEmpty());
    ASSERT_EQ(shape, empty.getShapeAsVector());
    ASSERT_EQ(3, empty.rankOf());
}

TEST_F(MyTests, test_shaped_empty_2) {
    auto empty = NDArrayFactory::create<float>('c', { 0, 3 });
    std::vector<Nd4jLong> shape = { 0, 3 };

    ASSERT_EQ(nd4j::DataType::FLOAT32, empty.dataType());
    ASSERT_EQ(0, empty.lengthOf());
    ASSERT_TRUE(empty.isEmpty());
    ASSERT_EQ(shape, empty.getShapeAsVector());
    ASSERT_EQ(2, empty.rankOf());
}

TEST_F(MyTests, test_shaped_empty_3) {
    auto empty = NDArrayFactory::create<float>('c', { 0 });
    std::vector<Nd4jLong> shape = { 0 };

    ASSERT_EQ(nd4j::DataType::FLOAT32, empty.dataType());
    ASSERT_EQ(0, empty.lengthOf());
    ASSERT_TRUE(empty.isEmpty());
    ASSERT_EQ(shape, empty.getShapeAsVector());
    ASSERT_EQ(1, empty.rankOf());
}

TEST_F(MyTests, test_shaped_empty_4) {
    auto shape = ConstantShapeHelper::getInstance()->vectorShapeInfo(0, nd4j::DataType::FLOAT32);
    NDArray array(shape, true, nd4j::LaunchContext::defaultContext());
    std::vector<Nd4jLong> shapeOf({ 0 });

    ASSERT_TRUE(array.isEmpty());
    ASSERT_EQ(1, array.rankOf());
    ASSERT_EQ(shapeOf, array.getShapeAsVector());
}





TEST_F(MyTests, test_rgb_to_hsv_1) {
    /*
     test case generated by python colorsys and scaled to suit our needs
     from colorsys import *
     from random import *
     import numpy as np 
     rgbs = np.array([randint(0,255) for x in range(0,3*4*5)]).reshape([5,4,3])
     hsvs=np.apply_along_axis(lambda x: np.array(rgb_to_hsv(x[0]/255,x[1]/255,x[2]/255))*np.array([360,1,1]),2,rgbs)
     rgbs.ravel()
     hsvs.ravel() 
    */ 
    auto rgbs = NDArrayFactory::create<float>('c', { 5, 4, 3 }, 
        { 
           213.f, 220.f, 164.f, 121.f, 180.f, 180.f,  18.f, 245.f,  75.f, 235.f,  76.f,  74.f, 168.f,
            50.f, 233.f, 191.f, 132.f, 100.f, 207.f,  37.f, 245.f,  77.f, 250.f, 182.f, 111.f,  52.f,
            59.f, 193.f, 147.f, 137.f, 168.f, 103.f, 121.f,  48.f, 191.f, 187.f,  53.f,  82.f, 239.f,
           156.f,  37.f, 118.f, 244.f,  90.f,   7.f, 221.f,  98.f, 243.f,  12.f, 209.f, 192.f,   2.f,
           115.f, 205.f,  79.f, 247.f,  32.f,  70.f, 152.f, 180.f 
        });
    auto expected = NDArrayFactory::create<float>('c', { 5, 4, 3 },
        {
           6.75000000e+01, 2.54545455e-01, 8.62745098e-01, 1.80000000e+02,
           3.27777778e-01, 7.05882353e-01, 1.35066079e+02, 9.26530612e-01,
           9.60784314e-01, 7.45341615e-01, 6.85106383e-01, 9.21568627e-01,
           2.78688525e+02, 7.85407725e-01, 9.13725490e-01, 2.10989011e+01,
           4.76439791e-01, 7.49019608e-01, 2.89038462e+02, 8.48979592e-01,
           9.60784314e-01, 1.56416185e+02, 6.92000000e-01, 9.80392157e-01,
           3.52881356e+02, 5.31531532e-01, 4.35294118e-01, 1.07142857e+01,
           2.90155440e-01, 7.56862745e-01, 3.43384615e+02, 3.86904762e-01,
           6.58823529e-01, 1.78321678e+02, 7.48691099e-01, 7.49019608e-01,
           2.30645161e+02, 7.78242678e-01, 9.37254902e-01, 3.19159664e+02,
           7.62820513e-01, 6.11764706e-01, 2.10126582e+01, 9.71311475e-01,
           9.56862745e-01, 2.90896552e+02, 5.96707819e-01, 9.52941176e-01,
           1.74822335e+02, 9.42583732e-01, 8.19607843e-01, 2.06600985e+02,
           9.90243902e-01, 8.03921569e-01, 1.06883721e+02, 8.70445344e-01,
           9.68627451e-01, 1.95272727e+02, 6.11111111e-01, 7.05882353e-01
        });


    auto actual = NDArrayFactory::create<float>('c', { 5,4,3 });

    Context ctx(1);
    ctx.setInputArray(0, &rgbs);
    ctx.setOutputArray(0, &actual);

    nd4j::ops::rgb_to_hsv op;
    auto status = op.execute(&ctx);
#if 0
    //visual check
    rgbs.printBuffer("rgbs ");
    actual.printBuffer("HSV ");
    expected.printBuffer("exp");
#endif
    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual)); 

}

TEST_F(MyTests, test_rgb_to_hsv_2) {
    /*
      swapped_rgbs=rgbs.swapaxes(1,2).ravel()
      swapped_hsvs=hsvs.swapaxes(1,2).ravel()
    */
    auto rgbs = NDArrayFactory::create<float>('c', { 5,3,4},
        {  
            213.f, 121.f,  18.f, 235.f, 220.f, 180.f, 245.f,  76.f, 164.f, 180.f,  75.f,  74.f, 168.f,
            191.f, 207.f,  77.f,  50.f, 132.f,  37.f, 250.f, 233.f, 100.f, 245.f, 182.f, 111.f, 193.f,
            168.f,  48.f,  52.f, 147.f, 103.f, 191.f,  59.f, 137.f, 121.f, 187.f,  53.f, 156.f, 244.f,
            221.f,  82.f,  37.f,  90.f,  98.f, 239.f, 118.f,   7.f, 243.f,  12.f,   2.f,  79.f,  70.f,
            209.f, 115.f, 247.f, 152.f, 192.f, 205.f,  32.f, 180.f
        });
    auto expected = NDArrayFactory::create<float>('c', { 5,3,4 },
        {
           6.75000000e+01, 1.80000000e+02, 1.35066079e+02, 7.45341615e-01,
           2.54545455e-01, 3.27777778e-01, 9.26530612e-01, 6.85106383e-01,
           8.62745098e-01, 7.05882353e-01, 9.60784314e-01, 9.21568627e-01,
           2.78688525e+02, 2.10989011e+01, 2.89038462e+02, 1.56416185e+02,
           7.85407725e-01, 4.76439791e-01, 8.48979592e-01, 6.92000000e-01,
           9.13725490e-01, 7.49019608e-01, 9.60784314e-01, 9.80392157e-01,
           3.52881356e+02, 1.07142857e+01, 3.43384615e+02, 1.78321678e+02,
           5.31531532e-01, 2.90155440e-01, 3.86904762e-01, 7.48691099e-01,
           4.35294118e-01, 7.56862745e-01, 6.58823529e-01, 7.49019608e-01,
           2.30645161e+02, 3.19159664e+02, 2.10126582e+01, 2.90896552e+02,
           7.78242678e-01, 7.62820513e-01, 9.71311475e-01, 5.96707819e-01,
           9.37254902e-01, 6.11764706e-01, 9.56862745e-01, 9.52941176e-01,
           1.74822335e+02, 2.06600985e+02, 1.06883721e+02, 1.95272727e+02,
           9.42583732e-01, 9.90243902e-01, 8.70445344e-01, 6.11111111e-01,
           8.19607843e-01, 8.03921569e-01, 9.68627451e-01, 7.05882353e-01
        });


    auto actual = NDArrayFactory::create<float>('c', { 5,3,4 });

    Context ctx(1);
    ctx.setInputArray(0, &rgbs);
    ctx.setOutputArray(0, &actual);
    ctx.setIArguments({ 1 });
    nd4j::ops::rgb_to_hsv op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}

TEST_F(MyTests, test_rgb_to_hsv_3) {
    /*
      2D
    */
    auto rgbs = NDArrayFactory::create<float>('c', { 8,3 },
        { 130.f,  61.f, 239.f, 117.f,  16.f, 168.f, 181.f, 223.f,   0.f,  49.f, 195.f, 195.f, 131.f,
       153.f,  78.f,  86.f,  21.f, 209.f, 101.f,  14.f, 107.f, 191.f,  98.f, 210.f });
    auto expected = NDArrayFactory::create<float>('c', { 8,3 },
        { 263.25842697f,   0.74476987f,   0.9372549f, 279.86842105f,
         0.9047619f,   0.65882353f,  71.30044843f,   1.f,
         0.8745098f, 180.f,   0.74871795f,   0.76470588f,
        77.6f,   0.49019608f,   0.6f, 260.74468085f,
         0.89952153f,   0.81960784f, 296.12903226f,   0.86915888f,
         0.41960784f, 289.82142857f,   0.53333333f,   0.82352941f });


    auto actual = NDArrayFactory::create<float>('c', { 8,3 });

    Context ctx(1);
    ctx.setInputArray(0, &rgbs);
    ctx.setOutputArray(0, &actual);

    nd4j::ops::rgb_to_hsv op;
    auto status = op.execute(&ctx);
#if 0
    //visual check
    rgbs.printBuffer("rgbs ");
    actual.printBuffer("HSV ");
    expected.printBuffer("exp");
#endif
    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}


TEST_F(MyTests, test_rgb_to_hsv_4) {
    /*
      2D
    */
    auto rgbs = NDArrayFactory::create<float>('c', { 3,8 },
        { 130.f, 117.f, 181.f,  49.f, 131.f,  86.f, 101.f, 191.f,  61.f,  16.f, 223.f, 195.f, 153.f,
        21.f,  14.f,  98.f, 239.f, 168.f,   0.f, 195.f,  78.f, 209.f, 107.f, 210.f });
    auto expected = NDArrayFactory::create<float>('c', {3,8 },
        { 263.25842697f, 279.86842105f,  71.30044843f, 180.f,
        77.6f, 260.74468085f, 296.12903226f, 289.82142857f,
         0.74476987f,   0.9047619f,   1.f,   0.74871795f,
         0.49019608f,   0.89952153f,   0.86915888f,   0.53333333f,
         0.9372549f,   0.65882353f,   0.8745098f,   0.76470588f,
         0.6f,   0.81960784f,   0.41960784f,   0.82352941f });


    auto actual = NDArrayFactory::create<float>('c', { 3,8 });

    Context ctx(1);
    ctx.setInputArray(0, &rgbs);
    ctx.setOutputArray(0, &actual);
    ctx.setIArguments({ 0 });
    nd4j::ops::rgb_to_hsv op;
    auto status = op.execute(&ctx);
#if 0
    //visual check
    rgbs.printBuffer("rgbs ");
    actual.printBuffer("HSV ");
    expected.printBuffer("exp");
#endif
    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}

TEST_F(MyTests, test_rgb_to_hsv_5) {
    /* 

    */
    auto rgbs = NDArrayFactory::create<float>('c', {  3 },
        { 213.f, 220.f, 164.f }  );
    auto expected = NDArrayFactory::create<float>('c', { 3 },
        {   6.75000000e+01, 2.54545455e-01, 8.62745098e-01  });


    auto actual = NDArrayFactory::create<float>('c', { 3 });

    Context ctx(1);
    ctx.setInputArray(0, &rgbs);
    ctx.setOutputArray(0, &actual);

    nd4j::ops::rgb_to_hsv op;
    auto status = op.execute(&ctx);
#if 0
    //visual check
    rgbs.printBuffer("rgbs ");
    actual.printBuffer("HSV ");
    expected.printBuffer("exp");
#endif
    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}


TEST_F(MyTests, test_rgb_to_hsv_6) {
    /*

    */
    auto rgbs = NDArrayFactory::create<float>('c', { 3,8 },
        { 130.f, 117.f, 181.f,  49.f, 131.f,  86.f, 101.f, 191.f,  61.f,  16.f, 223.f, 195.f, 153.f,
        21.f,  14.f,  98.f, 239.f, 168.f,   0.f, 195.f,  78.f, 209.f, 107.f, 210.f });

    auto expected = NDArrayFactory::create<float>('c', { 3  },
        { 263.25842697f, 0.74476987f,  0.9372549f });

    //get subarray 
    std::unique_ptr<NDArray> subArrRgbs(rgbs.subarray({ NDIndex::all(), NDIndex::point(0) }));
    subArrRgbs->reshapei({ 3 });
#if 0
    //[RANK][SHAPE][STRIDES][OPTIONS][EWS][ORDER]
    subArrRgbs->printShapeInfo("subArrRgbs");
#endif
    auto actual = NDArrayFactory::create<float>('c', { 3  });

    Context ctx(1);
    ctx.setInputArray(0, subArrRgbs.get());
    ctx.setOutputArray(0, &actual);
    ctx.setIArguments({ 0 });
    nd4j::ops::rgb_to_hsv op;
    auto status = op.execute(&ctx);
#if 0
    //visual check
    subArrRgbs->printBuffer("subArrRgbs ");
    actual.printBuffer("HSV ");
    expected.printBuffer("exp");
#endif
    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}

/*
TEST_F(MyTests, test_rgb_to_hsv_4) {
   
    auto rgbs = NDArrayFactory::create<float>('c', { 3 },
        { 213.f,0,0, 220.f,0,0, 164.f,0,0 });
    auto expected = NDArrayFactory::create<float>('c', { 3 },
        { 6.75000000e+01,0,0, 2.54545455e-01,0,0, 8.62745098e-01,0,0 });


    auto actual = NDArrayFactory::create<float>('c', { 3 });

    Context ctx(1);
    ctx.setInputArray(0, &rgbs);
    ctx.setOutputArray(0, &actual);

    nd4j::ops::rgb_to_hsv op;
    auto status = op.execute(&ctx);
#if 0
    //visual check
    rgbs.printBuffer("rgbs ");
    actual.printBuffer("HSV ");
    expected.printBuffer("exp");
#endif
    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}
*/
TEST_F(MyTests, test_hsv_to_rgb_1) {
    /*
     using the same numbers of rgb_to_hsv_1 test
    */
    auto expected = NDArrayFactory::create<float>('c', { 5,4,3 },
        { 213.f, 220.f, 164.f, 121.f, 180.f, 180.f,  18.f, 245.f,  75.f, 235.f,  76.f,  74.f, 168.f,
        50.f, 233.f, 191.f, 132.f, 100.f, 207.f,  37.f, 245.f,  77.f, 250.f, 182.f, 111.f,  52.f,
        59.f, 193.f, 147.f, 137.f, 168.f, 103.f, 121.f,  48.f, 191.f, 187.f,  53.f,  82.f, 239.f,
       156.f,  37.f, 118.f, 244.f,  90.f,   7.f, 221.f,  98.f, 243.f,  12.f, 209.f, 192.f,   2.f,
       115.f, 205.f,  79.f, 247.f,  32.f,  70.f, 152.f, 180.f }
    );
    auto hsvs = NDArrayFactory::create<float>('c', { 5,4,3 },
        {
      6.75000000e+01, 2.54545455e-01, 8.62745098e-01, 1.80000000e+02,
       3.27777778e-01, 7.05882353e-01, 1.35066079e+02, 9.26530612e-01,
       9.60784314e-01, 7.45341615e-01, 6.85106383e-01, 9.21568627e-01,
       2.78688525e+02, 7.85407725e-01, 9.13725490e-01, 2.10989011e+01,
       4.76439791e-01, 7.49019608e-01, 2.89038462e+02, 8.48979592e-01,
       9.60784314e-01, 1.56416185e+02, 6.92000000e-01, 9.80392157e-01,
       3.52881356e+02, 5.31531532e-01, 4.35294118e-01, 1.07142857e+01,
       2.90155440e-01, 7.56862745e-01, 3.43384615e+02, 3.86904762e-01,
       6.58823529e-01, 1.78321678e+02, 7.48691099e-01, 7.49019608e-01,
       2.30645161e+02, 7.78242678e-01, 9.37254902e-01, 3.19159664e+02,
       7.62820513e-01, 6.11764706e-01, 2.10126582e+01, 9.71311475e-01,
       9.56862745e-01, 2.90896552e+02, 5.96707819e-01, 9.52941176e-01,
       1.74822335e+02, 9.42583732e-01, 8.19607843e-01, 2.06600985e+02,
       9.90243902e-01, 8.03921569e-01, 1.06883721e+02, 8.70445344e-01,
       9.68627451e-01, 1.95272727e+02, 6.11111111e-01, 7.05882353e-01
        });


    auto actual = NDArrayFactory::create<float>('c', { 5,4,3 });

    Context ctx(1);
    ctx.setInputArray(0, &hsvs);
    ctx.setOutputArray(0, &actual);

    nd4j::ops::hsv_to_rgb op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}

TEST_F(MyTests, test_hsv_to_rgb_2) {
    /*
      using the same numbers of hsv_to_rgb_2
    */
    auto expected = NDArrayFactory::create<float>('c', { 5,3,4 },
        { 213.f, 121.f,  18.f, 235.f, 220.f, 180.f, 245.f,  76.f, 164.f, 180.f,  75.f,  74.f, 168.f,
       191.f, 207.f,  77.f,  50.f, 132.f,  37.f, 250.f, 233.f, 100.f, 245.f, 182.f, 111.f, 193.f,
       168.f,  48.f,  52.f, 147.f, 103.f, 191.f,  59.f, 137.f, 121.f, 187.f,  53.f, 156.f, 244.f,
       221.f,  82.f,  37.f,  90.f,  98.f, 239.f, 118.f,   7.f, 243.f,  12.f,   2.f,  79.f,  70.f,
       209.f, 115.f, 247.f, 152.f, 192.f, 205.f,  32.f, 180.f }
    );
    auto hsvs = NDArrayFactory::create<float>('c', { 5,3,4 },
        {
      6.75000000e+01, 1.80000000e+02, 1.35066079e+02, 7.45341615e-01,
       2.54545455e-01, 3.27777778e-01, 9.26530612e-01, 6.85106383e-01,
       8.62745098e-01, 7.05882353e-01, 9.60784314e-01, 9.21568627e-01,
       2.78688525e+02, 2.10989011e+01, 2.89038462e+02, 1.56416185e+02,
       7.85407725e-01, 4.76439791e-01, 8.48979592e-01, 6.92000000e-01,
       9.13725490e-01, 7.49019608e-01, 9.60784314e-01, 9.80392157e-01,
       3.52881356e+02, 1.07142857e+01, 3.43384615e+02, 1.78321678e+02,
       5.31531532e-01, 2.90155440e-01, 3.86904762e-01, 7.48691099e-01,
       4.35294118e-01, 7.56862745e-01, 6.58823529e-01, 7.49019608e-01,
       2.30645161e+02, 3.19159664e+02, 2.10126582e+01, 2.90896552e+02,
       7.78242678e-01, 7.62820513e-01, 9.71311475e-01, 5.96707819e-01,
       9.37254902e-01, 6.11764706e-01, 9.56862745e-01, 9.52941176e-01,
       1.74822335e+02, 2.06600985e+02, 1.06883721e+02, 1.95272727e+02,
       9.42583732e-01, 9.90243902e-01, 8.70445344e-01, 6.11111111e-01,
       8.19607843e-01, 8.03921569e-01, 9.68627451e-01, 7.05882353e-01
        });


    auto actual = NDArrayFactory::create<float>('c', { 5,3,4 });

    Context ctx(1);
    ctx.setInputArray(0, &hsvs);
    ctx.setOutputArray(0, &actual);
    ctx.setIArguments({ 1 });
    nd4j::ops::hsv_to_rgb op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}

TEST_F(MyTests, test_hsv_to_rgb_3) {
    /*
      2D
    */
    auto expected = NDArrayFactory::create<float>('c', { 8,3 },
        { 130.f,  61.f, 239.f, 117.f,  16.f, 168.f, 181.f, 223.f,   0.f,  49.f, 195.f, 195.f, 131.f,
       153.f,  78.f,  86.f,  21.f, 209.f, 101.f,  14.f, 107.f, 191.f,  98.f, 210.f });
    auto hsvs = NDArrayFactory::create<float>('c', { 8,3 },
        { 263.25842697f,   0.74476987f,   0.9372549f, 279.86842105f,
         0.9047619f,   0.65882353f,  71.30044843f,   1.f,
         0.8745098f, 180.f,   0.74871795f,   0.76470588f,
        77.6f,   0.49019608f,   0.6f, 260.74468085f,
         0.89952153f,   0.81960784f, 296.12903226f,   0.86915888f,
         0.41960784f, 289.82142857f,   0.53333333f,   0.82352941f });


    auto actual = NDArrayFactory::create<float>('c', { 8,3 });

    Context ctx(1);
    ctx.setInputArray(0, &hsvs);
    ctx.setOutputArray(0, &actual);

    nd4j::ops::hsv_to_rgb op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}


TEST_F(MyTests, test_hsv_to_rgb_4) {
    /*
      2D
    */
    auto expected = NDArrayFactory::create<float>('c', { 3,8 },
        { 130.f, 117.f, 181.f,  49.f, 131.f,  86.f, 101.f, 191.f,  61.f,  16.f, 223.f, 195.f, 153.f,
        21.f,  14.f,  98.f, 239.f, 168.f,   0.f, 195.f,  78.f, 209.f, 107.f, 210.f });
    auto hsvs = NDArrayFactory::create<float>('c', { 3,8 },
        { 263.25842697f, 279.86842105f,  71.30044843f, 180.f,
        77.6f, 260.74468085f, 296.12903226f, 289.82142857f,
         0.74476987f,   0.9047619f,   1.f,   0.74871795f,
         0.49019608f,   0.89952153f,   0.86915888f,   0.53333333f,
         0.9372549f,   0.65882353f,   0.8745098f,   0.76470588f,
         0.6f,   0.81960784f,   0.41960784f,   0.82352941f });


    auto actual = NDArrayFactory::create<float>('c', { 3,8 });

    Context ctx(1);
    ctx.setInputArray(0, &hsvs);
    ctx.setOutputArray(0, &actual);
    ctx.setIArguments({ 0 });
    nd4j::ops::hsv_to_rgb op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}

TEST_F(MyTests, test_hsv_to_rgb_5) {
    /*

    */
    auto expected = NDArrayFactory::create<float>('c', { 3 },
        { 213.f, 220.f, 164.f });
    auto hsvs = NDArrayFactory::create<float>('c', { 3 },
        { 6.75000000e+01, 2.54545455e-01, 8.62745098e-01 });


    auto actual = NDArrayFactory::create<float>('c', { 3 });

    Context ctx(1);
    ctx.setInputArray(0, &hsvs);
    ctx.setOutputArray(0, &actual);

    nd4j::ops::hsv_to_rgb op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}


TEST_F(MyTests, test_hsv_to_rgb_6) {
   
    auto expected = NDArrayFactory::create<double>('c', { 3 },
        { 130.0, 61.0, 239.0 });
    auto hsvs = NDArrayFactory::create<double>('c', { 3,8 },
        { 263.25842697, 279.86842105,  71.30044843, 180,
        77.6, 260.74468085, 296.12903226, 289.82142857,
         0.74476987,   0.9047619,   1.,   0.74871795,
         0.49019608,   0.89952153,   0.86915888,   0.53333333,
         0.9372549,   0.65882353,   0.8745098,   0.76470588,
         0.6,   0.81960784,   0.41960784,   0.82352941
});

    //get subarray 
    std::unique_ptr<NDArray> subArrHsvs(hsvs.subarray({ NDIndex::all(), NDIndex::point(0) }));
    subArrHsvs->reshapei({ 3 });
#if 0
    //[RANK][SHAPE][STRIDES][OPTIONS][EWS][ORDER]
    subArrHsvs->printShapeInfo("subArrHsvs");
#endif
    auto actual = NDArrayFactory::create<double>('c', { 3 });

    Context ctx(1);
    ctx.setInputArray(0, subArrHsvs.get());
    ctx.setOutputArray(0, &actual); 
    nd4j::ops::hsv_to_rgb op;
    auto status = op.execute(&ctx);
#if 0
    //visual check
    subArrHsvs->printBuffer("subArrHsvs ");
    actual.printBuffer("rgb ");
    expected.printBuffer("exp");
#endif
    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.equalsTo(actual));

}