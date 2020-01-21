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

#include "testlayers.h"
#include <initializer_list>
#include <NDArrayFactory.h>
#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/CustomOperations.h>
#include <execution/Engine.h>

#ifdef HAVE_CNML

#include <ops/declarable/platform/cnml/cnmlUtils.h>

#endif

using namespace nd4j;

class CnmlTests : public testing::Test {
public:

};

static void printer(std::initializer_list<nd4j::ops::platforms::PlatformHelper*> helpers) {

    for (auto v:helpers) {
        nd4j_printf("Initialized [%s]\n", v->name().c_str());
    }
}


TEST_F(CnmlTests, helpers_includer) {
    // we need this block, to make sure all helpers are still available within binary, and not optimized out by linker
#ifdef HAVE_CNML
    nd4j::ops::platforms::PLATFORM_add_ENGINE_MLU add;
    printer({&add});
#endif
}

TEST_F(CnmlTests, basic_add_test_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 2, 2, 2}, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    auto y = NDArrayFactory::create<float>('c', {2, 2, 2, 2}, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    auto e = NDArrayFactory::create<float>('c', {2, 2, 2, 2}, {2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f});
    auto z = x.ulike();

    nd4j::ops::add op;
    auto status = op.execute({&x, &y}, {&z}, {}, {}, {});
    ASSERT_EQ(Status::OK(), status);

    ASSERT_EQ(z, e);
}
