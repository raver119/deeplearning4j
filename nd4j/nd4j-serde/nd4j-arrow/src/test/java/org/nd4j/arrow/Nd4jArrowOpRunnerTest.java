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

import org.bytedeco.arrow.PrimitiveArray;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class Nd4jArrowOpRunnerTest {

    @Test
    public void testOpExec() {
        INDArray arr = Nd4j.scalar(1.0);
        INDArray arr2 = Nd4j.scalar(2.0);
        PrimitiveArray conversionOne = ByteDecoArrowSerde.arrayFromExistingINDArray(arr);
        PrimitiveArray conversionTwo = ByteDecoArrowSerde.arrayFromExistingINDArray(arr2);
        INDArray verifyFirst = ByteDecoArrowSerde.ndarrayFromArrowArray(conversionOne).reshape(new long[0]);
        INDArray verifySecond = ByteDecoArrowSerde.ndarrayFromArrowArray(conversionTwo).reshape(new long[0]);
        assertEquals(arr,verifyFirst);
        assertEquals(arr2,verifySecond);
        Nd4jArrowOpRunner.runOpOn(new PrimitiveArray[]{conversionOne,conversionTwo},"add");


    }


}
