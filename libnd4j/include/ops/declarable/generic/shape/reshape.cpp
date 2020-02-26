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
// Created by raver119 on 29/10/17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_reshape)

#include <ops/declarable/CustomOperations.h>
#include <performance/benchmarking/global_timers.h>

namespace nd4j {
namespace ops  {

//////////////////////////////////////////////////////////////////////////
// here iArgs is a vector with (optional) negative of order as first element:
// ({-order, dim1, dim2, dim3, ...})
CUSTOM_OP_IMPL(reshape, 1, 1, false, 0, -2) {
    GlobalTimers* timers = GlobalTimers::getInstance();
    timers->stopWatch(__LINE__, 10);
    auto x = INPUT_VARIABLE(0);
    timers->stopWatch(__LINE__, 10);
    auto z = OUTPUT_VARIABLE(0);
    timers->stopWatch(__LINE__, 10);

    //Special case: empty.reshape(<other empty shape>) -> return empty
    timers->stopWatch(__LINE__, 10);
        if (x->isEmpty()) {
            timers->stopWatch(__LINE__, 10);
            REQUIRE_TRUE(z->isEmpty(), 0, "Reshape: when input is empty, output must also be empty");
            timers->stopWatch(__LINE__, 10);
            return Status::OK();    //No op
        }
    timers->stopWatch(__LINE__, 10);
    if (block.width() == 1) {
timers->stopWatch(__LINE__, 10);
        auto arguments = block.getIArguments();
        timers->stopWatch(__LINE__, 10);
        int argsSize = arguments->size();
timers->stopWatch(__LINE__, 10);


        int e = 1;
        char order = (char) -(*arguments)[0];
        timers->stopWatch(__LINE__, 10);
        if (order != 'c' && order != 'f') {
            order = 'c'; //x->ordering();
            e = 0;
        }
timers->stopWatch(__LINE__, 10);
        REQUIRE_TRUE(argsSize - e >= 1, 0, "Reshape arguments should have at least 1 dimension");
timers->stopWatch(__LINE__, 10);
        std::vector<Nd4jLong> shapeNew;
        timers->stopWatch(__LINE__, 10);
        int e2 = e;
        for (; e < (int) arguments->size(); e++) {
            timers->stopWatch(__LINE__, 10);
            if (arguments->at(e) == -1){
                Nd4jLong shapeLength = 1;
                for(; e2 < e; e2++){
                    shapeLength *= arguments->at(e2);
                }
                timers->stopWatch(__LINE__, 10);
                for(e2 = e + 1; e2 < arguments->size(); e2++){
                    shapeLength *= arguments->at(e2);
                }
                timers->stopWatch(__LINE__, 10);
                Nd4jLong realShape = x->lengthOf() / shapeLength;
                shapeNew.push_back(realShape);
                timers->stopWatch(__LINE__, 10);
            }
            else{
                timers->stopWatch(__LINE__, 10);
                shapeNew.push_back(arguments->at(e));
                timers->stopWatch(__LINE__, 10);
            }

        }
timers->stopWatch(__LINE__, 10);
        auto len = shape::prodLong(shapeNew.data(), shapeNew.size());
        timers->stopWatch(__LINE__, 10);
        REQUIRE_TRUE(len == x->lengthOf(), 0, "Reshape: lengths before and after reshape should match, but got %i vs %i", x->lengthOf(), len);
timers->stopWatch(__LINE__, 10);
        if (Environment::getInstance()->isDebugAndVerbose()) {
            nd4j_printv("Reshape: new shape", shapeNew);
        }
timers->stopWatch(__LINE__, 10);
        auto xr = x->reshape(order, shapeNew);
        timers->stopWatch(__LINE__, 10);
        z->assign(xr);
timers->stopWatch(__LINE__, 10);
        STORE_RESULT(*z);
timers->stopWatch(__LINE__, 10);
        return Status::OK();

    } else if (block.width() == 2) {
timers->stopWatch(__LINE__, 10);
        auto s = INPUT_VARIABLE(1);
timers->stopWatch(__LINE__, 10);
        char order = 'c';
        timers->stopWatch(__LINE__, 10);
        if (block.numI() > 0)
            order = (char) -INT_ARG(0);
timers->stopWatch(__LINE__, 10);
        std::vector<Nd4jLong> shapeNew(s->lengthOf());
timers->stopWatch(__LINE__, 10);
        for (int e = 0; e < (int) s->lengthOf(); e++) {
            timers->stopWatch(__LINE__, 10);
            auto dim = s->e<Nd4jLong >(e);
            timers->stopWatch(__LINE__, 10);
            if (dim == -1){
                Nd4jLong shapeLength = 1;
                timers->stopWatch(__LINE__, 10);
                for(int e2 = 0; e2 < e; e2++){
                    shapeLength *= s->e<Nd4jLong>(e2);
                }
                timers->stopWatch(__LINE__, 10);
                for(int e2 = e + 1; e2 < (int) s->lengthOf(); e2++){
                    REQUIRE_TRUE(s->e<Nd4jLong>(e2) != -1, 0, "Reshape : Only one unknown dimension (-1) is allowed.");
                    shapeLength *= s->e<Nd4jLong>(e2);
                }
                timers->stopWatch(__LINE__, 10);
                Nd4jLong realShape = x->lengthOf() / shapeLength;
                timers->stopWatch(__LINE__, 10);
                shapeNew[e] = realShape;
                timers->stopWatch(__LINE__, 10);
            }
            else{
                shapeNew[e] = dim;
            }
        }
        timers->stopWatch(__LINE__, 10);
        if (Environment::getInstance()->isDebugAndVerbose()) {
            nd4j_printv("Reshape: new shape", shapeNew);
        }
timers->stopWatch(__LINE__, 10);
        if (s->isEmpty()) {
            timers->stopWatch(__LINE__, 10);
            // just a scalar
            z->assign(x);
            timers->stopWatch(__LINE__, 10);
        } else {
            timers->stopWatch(__LINE__, 10);
            auto xr = x->reshape(order, shapeNew);
            timers->stopWatch(__LINE__, 10);
            z->assign(xr);
            timers->stopWatch(__LINE__, 10);
        }
timers->stopWatch(__LINE__, 10);
        return Status::OK();

    }
timers->stopWatch(__LINE__, 10);
    return ND4J_STATUS_BAD_INPUT;
}


DECLARE_TYPES(reshape) {
    getOpDescriptor()
            ->setAllowedInputTypes(0, nd4j::DataType::ANY)
            ->setAllowedInputTypes(1, {ALL_INTS})
            ->setSameMode(true);
}

DECLARE_SHAPE_FN(reshape) {
    auto inp = inputShape->at(0);

    // we can launch op using Int arguments
    if (inputShape->size() == 1) {
        REQUIRE_TRUE(block.numI() > 0, 0, "Reshape: new shape should be provided as NDArray or int arguments, but nothing was defined");
        std::vector<int> *arguments = block.getIArguments();

        int e = 1;
        char order = (char) -(*arguments)[0];
        if (order != 'c' && order != 'f') {
            order = shape::order(inp);
            e = 0;
        }

        std::vector<Nd4jLong> shapeNew;

        int e2 = e;
        for (; e < (int) arguments->size(); e++) {
            if ((int) arguments->at(e) == -1){

                Nd4jLong shapeLength = 1;
                for(; e2 < e; e2 ++){
                    shapeLength *= arguments->at(e2);
                }
                for(e2 = e + 1; e2 < arguments->size(); e2++){
                    REQUIRE_TRUE(arguments->at(e2) != -1, 0, "Reshape : Only one unknown dimension (-1) is allowed.");
                    shapeLength *= arguments->at(e2);
                }

                if(shapeLength == 0){
                    //Edge case for empty:
                    shapeNew.push_back(0);
                } else {
                    //Standard case
                    Nd4jLong realShape = shape::length(inp) / shapeLength;
                    shapeNew.push_back(realShape);
                }
            }
            else{
                shapeNew.push_back(arguments->at(e));
            }
        }

        return SHAPELIST(ConstantShapeHelper::getInstance()->createShapeInfo(ShapeDescriptor(ArrayOptions::dataType(inp), order, shapeNew)));
    } else {
        // or, with second input "as shape"
        auto x = INPUT_VARIABLE(0);
        auto y = INPUT_VARIABLE(1);

        // special case here
        if (y->isEmpty()) {
            REQUIRE_TRUE(x->lengthOf() == 1, 0, "Reshape: new length doesn't match existing array");
            return SHAPELIST(ConstantShapeHelper::getInstance()->scalarShapeInfo(ArrayOptions::dataType(inp)));
        }
        //Special case: empty.reshape(-1) -> return empty
        if (x->isEmpty()) {
            //REQUIRE_TRUE(y->lengthOf() == 1 && y->e<Nd4jLong>(0) == -1, 0, "Reshape: when input is empty, shape must be [-1]");
            auto shapeOf = y->getBufferAsVector<Nd4jLong>();
            Nd4jLong prod = 1;
            bool hasNegs = false;
            for (auto v:shapeOf) {
                if (v < 0) {
                    hasNegs = true;
                    v = 0;
                }

                prod *= v;
            }

            REQUIRE_TRUE(prod == 0, 0, "Reshape: in case of empty arrays reshape must return empty array as well");

            // if there are -1s - we turn them into zeros
            if (hasNegs) {
                for (int e = 0; e < shapeOf.size(); e++)
                    if (shapeOf[e] < 0)
                        shapeOf[e] = 0;
            }

            auto newShape = ShapeBuilders::createShapeInfo(ArrayOptions::dataType(inp), shape::order(inp), y->lengthOf(), shapeOf.data());
            return SHAPELIST(CONSTANT(newShape));
        }

        std::vector<Nd4jLong> shapeNew(y->lengthOf());

        for (int e = 0; e < (int) y->lengthOf(); e++) {
            auto dim = y->e<Nd4jLong>(e);
            if (dim == -1){
                Nd4jLong shapeLength = 1;
                for(int e2 = 0; e2 < e; e2++){
                    shapeLength *= y->e<Nd4jLong>(e2);
                }
                for(int e2 = e + 1; e2 < (int)y->lengthOf(); e2++){
                    REQUIRE_TRUE(y->e<Nd4jLong>(e2) != -1, 0, "Reshape : Only one unknown dimension (-1) is allowed.");
                    shapeLength *= y->e<Nd4jLong>(e2);
                }

                if(shapeLength == 0){
                    //Edge case for empty:
                    shapeNew[e] = 0;
                } else {
                    Nd4jLong realShape = shape::length(inp) / shapeLength;
                    shapeNew[e] = realShape;
                }
            }else {
                shapeNew[e] = dim;
            }
        }

        return SHAPELIST(ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inp), 'c', shapeNew));
    }
}
}
}

#endif
