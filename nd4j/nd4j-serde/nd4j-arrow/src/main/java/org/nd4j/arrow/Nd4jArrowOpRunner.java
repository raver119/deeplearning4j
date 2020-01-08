/*
 *  Copyright (c) 2019 Konduit KK
 *
 *   This program and the accompanying materials are made available under the
 *   terms of the Apache License, Version 2.0 which is available at
 *   https://www.apache.org/licenses/LICENSE-2.0.
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *   WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *   License for the specific language governing permissions and limitations
 *   under the License.
 *
 *   SPDX-License-Identifier: Apache-2.0
 *
 */

package org.nd4j.arrow;

import org.bytedeco.arrow.FlatArray;
import org.bytedeco.arrow.PrimitiveArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp.DynamicCustomOpsBuilder;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.arrow.ByteDecoArrowSerde.ndarrayFromArrowArray;

/**
 * Runs {@link DynamicCustomOp}
 * on arrow based data types.
 *
 * @author Adam Gibson
 */
public class Nd4jArrowOpRunner {

    /**
     * Runs operations
     * @param array the input {@link PrimitiveArray}
     * @param opName the op name to run
     * @param args the args (booleans, integers,..)
     * @return the {@link PrimitiveArray} equivalents
     * from the outputs from the execution of {@link DynamicCustomOp}
     * derived from the input names.
     */
    public static FlatArray[] runOpOn(FlatArray[] array, String opName, Object...args) {
        DynamicCustomOpsBuilder opBuilder =  DynamicCustomOp.builder(opName);
        for(Object arg : args) {
            if(arg instanceof Integer || arg instanceof Long) {
                Number integer = (Number) arg;
                opBuilder.addIntegerArguments(integer.longValue());
            }
            else if(arg instanceof Float || arg instanceof Double) {
                Number floatArg = (Number) arg;
                opBuilder.addFloatingPointArguments(floatArg.doubleValue());
            }
            else if(arg instanceof Boolean) {
                Boolean boolArg = (Boolean) arg;
                opBuilder.addBooleanArguments(boolArg);
            }
        }

        INDArray[] inputs = new INDArray[array.length];
        for(int i = 0; i < inputs.length; i++) {
            inputs[i] = ndarrayFromArrowArray(array[i]);
        }

        opBuilder.addInputs(inputs);

        DynamicCustomOp build = opBuilder.build();
        Nd4j.getExecutioner().exec(build);
        INDArray[] ret = build.outputArguments().toArray(new INDArray[0]);
        FlatArray[] outputArrays = new FlatArray[ret.length];
        for(int i = 0; i < ret.length; i++) {
            outputArrays[i] = ByteDecoArrowSerde.arrayFromExistingINDArray(ret[i]);
        }

        return outputArrays;
    }



}
