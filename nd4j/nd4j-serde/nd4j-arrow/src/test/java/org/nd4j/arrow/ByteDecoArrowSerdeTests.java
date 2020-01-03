/*******************************************************************************
 * Copyright (c) 2019 Konduit KK
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

package org.nd4j.arrow;

import lombok.val;
import org.bytedeco.arrow.ArrayData;
import org.bytedeco.arrow.ArrowBuffer;
import org.bytedeco.arrow.FlatArray;
import org.bytedeco.arrow.Tensor;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.Utf8Buffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.util.Arrays;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;

public class ByteDecoArrowSerdeTests {


    @Test
    public void testBufferConversion() {
        for(DataType value : DataType.values()) {
            assertBufferCreation(Nd4j.createBuffer(new int[]{1,1},value,0));
        }

    }

    @Test
    public void testStringOffsetsGeneration() {
        DataBuffer dataBuffer = Nd4j.createBufferOfType(DataType.UTF8,new String[]{"hello1","hello2"});
        DataBuffer offsets = dataBuffer.binaryOffsets();
        assertEquals(dataBuffer.length(),offsets.length());
    }

    @Test
    public void testToTensor() {
        for(DataType value : DataType.values()) {
            if(value == DataType.UTF8)
                continue;

            INDArray arr = Nd4j.create(Nd4j.createBuffer(new int[]{1,1},value,0));
            Tensor convert = ByteDecoArrowSerde.toTensor(arr);
            INDArray convertedBack = ByteDecoArrowSerde.fromTensor(convert);
            assertEquals(arr,convertedBack);
        }
    }

    @Test
    public void testToFromTensorDataTypes() {
        for(DataType dataType : DataType.values()) {
            if(dataType == DataType.COMPRESSED || dataType == DataType.BFLOAT16 || dataType == DataType.UNKNOWN)
                continue;

            org.bytedeco.arrow.DataType dataType1 = ByteDecoArrowSerde.arrowDataTypeForNd4j(dataType);
            DataType dataType2 = ByteDecoArrowSerde.dataBufferTypeTypeForArrow(dataType1);

            assertEquals(dataType,dataType2);
        }
    }

    private void assertBufferCreation(DataBuffer buffer) {
        Pair<ArrowBuffer, org.bytedeco.arrow.DataType> arrowBuffer = ByteDecoArrowSerde.fromNd4jBuffer(buffer);
        assertEquals(buffer.dataType(),ByteDecoArrowSerde.dataBufferTypeTypeForArrow(arrowBuffer.getRight()));
        DataBuffer buffer1 = ByteDecoArrowSerde.fromArrowBuffer(arrowBuffer.getFirst(), arrowBuffer.getRight());
        assertEquals(buffer1,buffer1);
    }

    @Test
    public void testArrayDataFromArrowBuffer() {
        // Setup
        for(DataType dataType : DataType.values()) {
            if(dataType == DataType.COMPRESSED || dataType == DataType.UNKNOWN || dataType == DataType.BFLOAT16)
                continue;

            DataBuffer dataBuffer = null;
            if(dataType != DataType.UTF8) {
                dataBuffer = Nd4j.createBuffer(new int[]{1,2},dataType,0);
            }
            else {
                dataBuffer = Nd4j.createBuffer(new int[]{1,"hello world".length() * 2},dataType,0);
                assertEquals(1,dataBuffer.length());
                assertTrue(dataBuffer instanceof Utf8Buffer);
            }
            switch(dataType) {
                case BOOL:
                    dataBuffer.put(0,true);
                    break;
                case INT:
                    dataBuffer.put(0,(int) 1);
                    break;
                case LONG:
                    dataBuffer.put(0,1L);
                    break;
                case FLOAT:
                    dataBuffer.put(0,1.0f);
                    break;
                case DOUBLE:
                    dataBuffer.put(0,1.0d);
                    break;
                case UTF8:
                    dataBuffer.put(0,"hello world");
                    break;
            }

            val pair = ByteDecoArrowSerde.makeArrayData(dataBuffer);
            assertEquals(dataType,ByteDecoArrowSerde.dataBufferTypeTypeForArrow(pair.type()));
            switch(dataType) {
                case BOOL:
                    assertEquals(true,pair.GetValuesBoolean(0).get());
                    break;
                case INT:
                case LONG:
                    assertEquals(1,pair.GetValuesInt(0).get());
                    break;
                case FLOAT:
                    assertEquals(1.0f, pair.GetValuesFloat(0).get(),1e-1f);
                    break;
                case DOUBLE:
                    assertEquals(1.0,pair.GetValuesDouble(0).get(),1e-2);
                    break;
                case UTF8:
                    /**
                     * Note that the header needs to be somehow acknowledged
                     * in the pointer from array data.
                     * If we load from array data for utf-8
                     * we need to make sure we can load strings properly..
                     */
                    BytePointer bytePointer = pair.GetValuesByte(0);
                    bytePointer.position(9);
                    bytePointer.capacity(27);
                    String assertionString = "hello world";
                    String testString = bytePointer.getString().trim();
                    assertEquals(assertionString,testString);
                    break;

            }

        }

    }

    @Test
    public void testConvertToNdArray() {
        INDArray arr = Nd4j.scalar(1.0).reshape(1,1);
        FlatArray array1 = ByteDecoArrowSerde.arrayFromExistingINDArray(arr);
        INDArray convertBack = ByteDecoArrowSerde.ndarrayFromArrowArray(array1).reshape(1,1);
        assertEquals(arr,convertBack);
    }
}
