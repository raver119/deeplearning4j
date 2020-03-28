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
#include <graph/Graph.h>
#include <chrono>
#include <graph/Node.h>
#include <helpers/OpTracker.h>
#include <ops/declarable/CustomOperations.h>
#include <graph/execution/OpSequence.h>
#include <graph/OptimizedGraph.h>

using namespace sd;
using namespace sd::ops;
using namespace sd::graph;

class OpSequenceTests : public testing::Test {
public:

    OpSequenceTests() {
    }
};

TEST_F(OpSequenceTests, test_iterator_1) {
    OpSequence sequence;

    ASSERT_EQ(0, sequence.length());

    ops::add op1;
    ops::multiply op2;

    Context ctx1(1);
    Context ctx2(2);

    sequence.append(&op1, &ctx1);
    sequence.append(&op2, &ctx2);

    ASSERT_EQ(2, sequence.length());

    int cnt = 1;
    for (const auto &v:sequence) {
        ASSERT_EQ(cnt++, v.second->nodeId());
    }

    ASSERT_EQ(3, cnt);

    OptimizedGraph optimizedGraph;
    ASSERT_EQ(0, optimizedGraph.layers());

    optimizedGraph.append(sequence);
    ASSERT_EQ(1, optimizedGraph.layers());

    auto layer = optimizedGraph.layer(0);

    // we expect exactly 1 sequence in this layer
    ASSERT_EQ(1, layer.width());

    auto seq = layer[0];

    ASSERT_EQ(2, seq.length());
}
