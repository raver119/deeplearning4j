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

package org.datavec.arrow.table;

import org.bytedeco.arrow.FlatArray;
import org.bytedeco.arrow.PrimitiveArray;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;


import static org.junit.Assert.assertArrayEquals;

public class DataVecArrowUtilsTest {

    @Test
    public void testToArrayDataConversion() {
        for(DataType dataType : DataType.values()) {
            switch(dataType) {
                case UINT32:
                    break;
                case UBYTE:
                    break;
                case BOOL:
                    boolean[] inputBoolean = {true};
                    FlatArray primitiveArrayBoolean = DataVecArrowUtils.convertBooleanArray(inputBoolean);
                    boolean[] booleans = DataVecArrowUtils.convertArrayToBoolean(primitiveArrayBoolean);
                    assertArrayEquals(inputBoolean,booleans);
                    break;
                case LONG:
                    long[] input = {1};
                    FlatArray primitiveArrayLong = DataVecArrowUtils.convertLongArray(input);
                    long[] longs = DataVecArrowUtils.convertArrayToLong(primitiveArrayLong);
                    assertArrayEquals(input,longs);
                    break;
                case UNKNOWN:
                    break;
                case SHORT:
                    break;
                case DOUBLE:
                    double[] inputDouble = {1.0};
                    FlatArray primitiveArrayDouble = DataVecArrowUtils.convertDoubleArray(inputDouble);
                    double[] doubles = DataVecArrowUtils.convertArrayToDouble(primitiveArrayDouble);
                    assertArrayEquals(inputDouble,doubles,1e-3);
                    break;
                case UTF8:
                    String[] inputString = {"input","input2"};
                    FlatArray primitiveArray = DataVecArrowUtils.convertStringArray(inputString);
                    String[] strings = DataVecArrowUtils.convertArrayToString(primitiveArray);
                    assertArrayEquals(inputString,strings);
                    break;
                case BFLOAT16:
                    break;
                case UINT16:
                    break;
                case INT:
                    int[] ret = {1};
                    FlatArray primitiveArray1 = DataVecArrowUtils.convertIntArray(ret);
                    int[] ints = DataVecArrowUtils.convertArrayToInt(primitiveArray1);
                    assertArrayEquals(ret,ints);
                    break;
                case BYTE:
                    break;
                case UINT64:
                    break;
                case HALF:
                    break;
                case FLOAT:
                    float[] retFloat = {1.0f};
                    FlatArray primitiveArrayFloat = DataVecArrowUtils.convertFloatArray(retFloat);
                    float[] floats = DataVecArrowUtils.convertArrayToFloat(primitiveArrayFloat);
                    assertArrayEquals(retFloat,floats,1e-3f);
                    break;
                case COMPRESSED:
                    break;
            }
        }
    }
}
