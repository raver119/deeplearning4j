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
// Created by raver119 on 21.02.18.
//

#include <array/NDArray.h>
#include <flatbuffers/flatbuffers.h>
#include <graph/Variable.h>
#include <ops/declarable/headers/broadcastable.h>

#include "testlayers.h"

using namespace sd;
using namespace sd::graph;

class NodeTests : public testing::Test {
 public:
};

TEST_F(NodeTests, test_copy_1) {
  Node a(sd::ops::add(), "add");

  Node b(sd::ops::divide(), "div");

  ASSERT_NE(a.name(), b.name());
  ASSERT_NE(a.customOp()->getOpName(), b.customOp()->getOpName());
  ASSERT_NE(a.contextPrototype().name(), b.contextPrototype().name());

  b = a;

  ASSERT_EQ(a.name(), b.name());
  ASSERT_EQ(a.customOp()->getOpName(), b.customOp()->getOpName());
  ASSERT_EQ(a.contextPrototype().name(), b.contextPrototype().name());

  ASSERT_NE(&a.contextPrototype(), &b.contextPrototype());
}

static FORCEINLINE Node copy(const Node& node) {
  return Node(node);
}

TEST_F(NodeTests, test_copy_2) {
  Node a(sd::ops::add(), "add");

  Node b(sd::ops::divide(), "div");

  ASSERT_NE(a.name(), b.name());
  ASSERT_NE(a.customOp()->getOpName(), b.customOp()->getOpName());
  ASSERT_NE(a.contextPrototype().name(), b.contextPrototype().name());

  b = copy(a);

  ASSERT_EQ(a.name(), b.name());
  ASSERT_EQ(a.customOp()->getOpName(), b.customOp()->getOpName());
  ASSERT_EQ(a.contextPrototype().name(), b.contextPrototype().name());

  ASSERT_NE(&a.contextPrototype(), &b.contextPrototype());
}

TEST_F(NodeTests, test_copy_3) {
  Node a(sd::ops::add(), "add");

  Node b = copy(a);

  ASSERT_EQ(a.name(), b.name());
  ASSERT_EQ(a.customOp()->getOpName(), b.customOp()->getOpName());
  ASSERT_EQ(a.contextPrototype().name(), b.contextPrototype().name());

  ASSERT_NE(&a.contextPrototype(), &b.contextPrototype());
}

static MAP_IMPL<int, Node> modifier(MAP_IMPL<int, Node> map) {
  if (map.size() < 5) {
    map[3] = Node(sd::ops::multiply(), "mul");
    return map;
  } else {
    map.erase(1);
    return map;
  }
}

TEST_F(NodeTests, test_copy_4) {
  MAP_IMPL<int, Node> map;
  map[1] = Node(sd::ops::add(), "add");
  map[2] = Node(sd::ops::divide(), "div");

  auto other = modifier(map);

  ASSERT_EQ(3, other.size());

  ASSERT_EQ(map[1].name(), other[1].name());
  ASSERT_EQ(map[1].contextPrototype().name(), other[1].contextPrototype().name());

  ASSERT_NE(&map[1].contextPrototype(), &other[1].contextPrototype());
}