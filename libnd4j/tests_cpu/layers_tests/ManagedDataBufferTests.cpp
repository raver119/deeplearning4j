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
#include <ops/declarable/CustomOperations.h>
#include <array/NDArray.h>
#include <legacy/NativeOps.h>
#include <fstream>
#include <array/ManagedDataBuffer.h>

using namespace sd;
using namespace sd::graph;

class ManagedDataBufferTests : public testing::Test {
public:
    ManagedDataBufferTests() {
        ///
    }
};

TEST_F(ManagedDataBufferTests, basic_constructor_test_1) {
    GraphMemoryManager mgr;
    auto mdb = std::make_shared<ManagedDataBuffer>(mgr, 0, DataType::FLOAT32, memory::MemoryZone::HOT);

    NDArray array(mdb, 'c', {0});
}

TEST_F(ManagedDataBufferTests, basic_constructor_test_2) {
    auto exp = NDArrayFactory::create<float>('c', {5}, {1.f, 1.f, 1.f, 1.f, 1.f});

    GraphMemoryManager mgr;
    auto mdb = std::make_shared<ManagedDataBuffer>(mgr, 20, DataType::FLOAT32, memory::MemoryZone::HOT);

    ASSERT_NE(nullptr, mdb->platform());

    NDArray array(mdb, 'c', {5});
    array.assign(1.0f);

    ASSERT_EQ(exp, array);
}