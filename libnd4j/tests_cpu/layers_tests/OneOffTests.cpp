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
// Created by raver119 on 11.10.2017.
//

#include <helpers/MmulHelper.h>
#include <memory/MemoryReport.h>
#include <memory/MemoryUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/OpTuple.h>

#include <vector>

#include "testlayers.h"

using namespace sd;
using namespace sd::ops;

class OneOffTests : public testing::Test {
 public:
};

TEST_F(OneOffTests, test_avg_pool_3d_1) {
  auto graph = Graph::fromFlatBuffers("./resources/avg_pooling3d.fb");

  graph.printOut();
  graph.execute();
}

TEST_F(OneOffTests, test_avg_pool_3d_2) {
  auto graph = Graph::fromFlatBuffers("./resources/avg_pooling3d.fb");

  graph.execute();
}

TEST_F(OneOffTests, test_non2d_0A_1) {
  auto graph = Graph::fromFlatBuffers("./resources/non2d_0A.fb");

  graph.execute();
}

TEST_F(OneOffTests, test_assert_scalar_float32_2) {
  sd::ops::Assert op;
  sd::ops::identity op1;
  sd::ops::noop op2;
  auto graph = Graph::fromFlatBuffers("./resources/assertsomething.fb");

  graph.printOut();

  // graph.execute();
}

TEST_F(OneOffTests, test_pad_1D_1) {
  auto e = NDArrayFactory::create<float>(
      'c', {7},
      {10.f, 0.778786f, 0.801198f, 0.724375f, 0.230894f, 0.727141f, 10.f});
  auto graph = Graph::fromFlatBuffers("./resources/pad_1D.fb");

  graph.execute();

  ASSERT_TRUE(graph.variableSpace().hasVariable(4));

  auto z = graph.variableSpace().getVariable(4)->getNDArray();
  ASSERT_TRUE(z != nullptr);

  ASSERT_EQ(e, *z);
}

TEST_F(OneOffTests, test_tensor_array_1) {
  auto e =
      NDArrayFactory::create<float>('c', {2, 3},
                                    {0.77878559f, 0.80119777f, 0.72437465f,
                                     0.23089433f, 0.72714126f, 0.18039072f});

  auto graph = Graph::fromFlatBuffers(
      "./resources/tensor_array_close_sz1_float32_nodynamic_noname_noshape.fb");

  graph.execute();

  ASSERT_TRUE(graph.variableSpace().hasVariable(5));

  auto z = graph.variableSpace().getVariable(5)->getNDArray();
  ASSERT_TRUE(z != nullptr);

  ASSERT_EQ(e, *z);
}

TEST_F(OneOffTests, test_tensor_array_2) {
  auto e =
      NDArrayFactory::create<float>('c', {2, 3},
                                    {0.77878559f, 0.80119777f, 0.72437465f,
                                     0.23089433f, 0.72714126f, 0.18039072f});

  auto graph = Graph::fromFlatBuffers(
      "./resources/tensor_array_split_sz1_float32_nodynamic_noname_noshape.fb");

  graph.execute();

  ASSERT_TRUE(graph.variableSpace().hasVariable(6));

  auto z = graph.variableSpace().getVariable(6)->getNDArray();
  ASSERT_TRUE(z != nullptr);

  ASSERT_EQ(e, *z);
}

TEST_F(OneOffTests, test_tensor_array_3) {
  auto e = NDArrayFactory::create<int>(
      'c', {3, 2, 3}, {7, 2, 9, 4, 3, 3, 8, 7, 0, 0, 6, 8, 7, 9, 0, 1, 1, 4});

  auto graph = Graph::fromFlatBuffers(
      "./resources/tensor_array_stack_sz3-1_int32_dynamic_name_shape.fb");

  graph.execute();

  ASSERT_TRUE(graph.variableSpace().hasVariable(15));

  auto z = graph.variableSpace().getVariable(15)->getNDArray();
  ASSERT_TRUE(z != nullptr);

  ASSERT_EQ(e, *z);
}

TEST_F(OneOffTests, test_tensor_array_4) {
  auto e = NDArrayFactory::create<Nd4jLong>('c', {2, 3}, {4, 3, 1, 1, 1, 0});

  auto graph = Graph::fromFlatBuffers(
      "./resources/"
      "tensor_array_unstack_sz1_int64_nodynamic_noname_shape2-3.fb");

  graph.execute();

  ASSERT_TRUE(graph.variableSpace().hasVariable(11));

  auto z = graph.variableSpace().getVariable(11)->getNDArray();
  ASSERT_TRUE(z != nullptr);

  ASSERT_EQ(e, *z);
}

TEST_F(OneOffTests, test_assert_4) {
  auto e = NDArrayFactory::create<Nd4jLong>('c', {2, 2}, {1, 1, 1, 1});

  auto graph = Graph::fromFlatBuffers("./resources/assert_type_rank2_int64.fb");

  graph.execute();

  ASSERT_TRUE(graph.variableSpace().hasVariable(1));

  auto z = graph.variableSpace().getVariable(1)->getNDArray();
  ASSERT_TRUE(z != nullptr);

  ASSERT_EQ(e, *z);
}

TEST_F(OneOffTests, test_identity_n_2) {
  auto e =
      NDArrayFactory::create<float>('c', {2, 3},
                                    {0.77878559f, 0.80119777f, 0.72437465f,
                                     0.23089433f, 0.72714126f, 0.18039072f});

  sd::ops::identity_n op;

  auto graph = Graph::fromFlatBuffers("./resources/identity_n_2.fb");

  graph.execute();

  ASSERT_TRUE(graph.variableSpace().hasVariable(1));
  ASSERT_TRUE(graph.variableSpace().hasVariable(1, 1));

  auto z = graph.variableSpace().getVariable(1)->getNDArray();
  ASSERT_TRUE(z != nullptr);

  ASSERT_EQ(e, *z);
}

TEST_F(OneOffTests, test_non2d_1) {
  auto e = NDArrayFactory::create<float>('c', {1, 2}, {2.07706356f, 2.66380072f});

  auto graph = Graph::fromFlatBuffers("./resources/non2d_1.fb");

  graph.execute();

  ASSERT_TRUE(graph.variableSpace().hasVariable(6));

  auto z = graph.variableSpace().getVariable(6)->getNDArray();
  ASSERT_TRUE(z != nullptr);

  ASSERT_EQ(e, *z);
}

TEST_F(OneOffTests, test_reduce_all_1) {
  auto e = NDArrayFactory::create<bool>('c', {1, 4}, {true, false, false, false});

  auto graph = Graph::fromFlatBuffers("./resources/reduce_all_rank2_d0_keep.fb");

  graph.execute();

  ASSERT_TRUE(graph.variableSpace().hasVariable(1));

  ASSERT_TRUE(graph.variableSpace().hasVariable(2));
  auto in = graph.variableSpace().getVariable(2)->getNDArray();

  auto z = graph.variableSpace().getVariable(1)->getNDArray();
  ASSERT_TRUE(z != nullptr);

  ASSERT_EQ(e, *z);
}