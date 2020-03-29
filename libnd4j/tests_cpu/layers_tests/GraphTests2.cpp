/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

#include "testlayers.h"
#include <flatbuffers/flatbuffers.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/Node.h>
#include <graph/Graph.h>
#include <graph/GraphUtils.h>
#include <array/NDArray.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/generic/parity_ops.cpp>

using namespace sd;
using namespace sd::graph;

class GraphTests2 : public testing::Test {
public:

    GraphTests2() {
        //
    }
};

TEST_F(GraphTests2, test_placeholder_1) {
    Graph graph;

    graph.addPlaceholder("input", 0, DataType::BFLOAT16, {4, 12, 48});

    ASSERT_TRUE(graph.getVariableSpace()->hasVariable("input"));

    auto variable = graph.getVariableSpace()->getVariable("input");

    ASSERT_NE(nullptr, variable);
    ASSERT_TRUE(variable->isPlaceholder());
    ASSERT_EQ(DataType::BFLOAT16, variable->dataType());
    ASSERT_EQ(std::vector<Nd4jLong>({4, 12, 48}), variable->shape());

    auto placeholders = graph.getPlaceholders();
    ASSERT_EQ(1, placeholders.size());
    ASSERT_EQ(placeholders[0], variable);
}