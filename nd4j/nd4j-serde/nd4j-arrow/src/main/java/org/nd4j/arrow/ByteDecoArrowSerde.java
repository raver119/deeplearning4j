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
import org.bytedeco.arrow.global.arrow;
import org.bytedeco.arrow.*;
import org.bytedeco.javacpp.BytePointer;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.Utf8Buffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;


/**
 * Arrow serialization utilities
 * using the javacpp arrow bindings.
 *
 * @author Adam Gibson
 */
public class ByteDecoArrowSerde {

    /**
     * Convert a {@link Tensor}
     * to an {@link INDArray}
     * @param tensor the input tensor
     * @return the equivalent {@link INDArray}
     */
    public static INDArray fromTensor(Tensor tensor) {
        long[] shape = new long[tensor.ndim()];
        long[] stride = new long[tensor.ndim()];

        long bufferCapacity = 1;
        for(int i = 0; i < tensor.ndim(); i++) {
            shape[i] = tensor.shape().get(i);
            stride[i] = tensor.strides().get(i);
            bufferCapacity *= shape[i];
        }


        org.nd4j.linalg.api.buffer.DataType dtype = dataBufferTypeTypeForArrow(tensor.type());
        //buffer capacity needs to be initialized properly, otherwise defaults to zero
        ArrowBuffer arrowBuffer = tensor.data().capacity(bufferCapacity);
        DataBuffer buffer = fromArrowBuffer(arrowBuffer,arrowDataTypeForNd4j(dtype));
        Preconditions.checkState(buffer.length() == ArrayUtil.prod(shape),"Data buffer creation from arrow failed. Data buffer is empty and not the same length as the shape.");
        INDArray arr = Nd4j.create(buffer,shape,stride,0);
        return arr;
    }

    /**
     *
     * @param input
     * @return
     */
    public static Tensor toTensor(INDArray input) {
        if(input.dataType() == org.nd4j.linalg.api.buffer.DataType.BOOL)
            throw new IllegalArgumentException("Arrow does not currently support converting boolean arrays to tensors.");
        ArrowBuffer arrowBuffer = fromNd4jBuffer(input.data()).getFirst();
        long[] shape = input.shape();
        long[] stride = input.stride();
        if(shape.length == 0) {
            shape = new long[] {1};
            stride = new long[] {1};
        }

        Tensor ret = new Tensor(arrowDataTypeForNd4j(input.dataType()),arrowBuffer,shape,stride);
        ret.data().capacity(arrowBuffer.capacity());
        ret.data().limit(arrowBuffer.limit());
        return ret;
    }



    /**
     * Convert a {@link org.nd4j.linalg.api.buffer.DataType}
     *  to an arrow {@link DataType}
     * @param dataType the input data type
     * @return the equivalent arrow data type
     */
    public static DataType arrowDataTypeForNd4j(org.nd4j.linalg.api.buffer.DataType dataType) {
        switch(dataType) {
            case UINT64:
                return arrow.uint64();
            case COMPRESSED:
                throw new IllegalArgumentException("Unable to convert data type " + dataType.name());
            case UINT16:
                return arrow.uint16();
            case UBYTE:
                return arrow.uint8();
            case SHORT:
                return arrow.int16();
            case BYTE:
                return arrow.int8();
            case FLOAT:
                return arrow.float32();
            case LONG:
                return arrow.int64();
            case BOOL:
                return arrow._boolean();
            case UTF8:
                return arrow.utf8();
            case INT:
                return arrow.int32();
            case HALF:
                return arrow.float16();
            case DOUBLE:
                return arrow.float64();
            case UNKNOWN:
                throw new IllegalArgumentException("Unable to convert data type " + dataType.name());
            case BFLOAT16:
                return arrow.float16();
            case UINT32:
                return arrow.uint32();
            default:
                throw new IllegalArgumentException("Unable to convert data type " + dataType.name());
        }

    }

    /**
     * Convert the input {@link DataType}
     * to the nd4j equivalent of {@link org.nd4j.linalg.api.buffer.DataType}
     * @param dataType the input data type
     * @return the equivalent nd4j data type
     */
    public static org.nd4j.linalg.api.buffer.DataType dataBufferTypeTypeForArrow(DataType dataType) {
        if(dataType.equals(arrow._boolean())) {
            return org.nd4j.linalg.api.buffer.DataType.BOOL;
        }
        else if(dataType.equals(arrow.uint8())) {
            return org.nd4j.linalg.api.buffer.DataType.UBYTE;
        }
        else if(dataType.equals(arrow.uint16())) {
            return org.nd4j.linalg.api.buffer.DataType.UINT16;
        }
        else if(dataType.equals(arrow.uint32())) {
            return org.nd4j.linalg.api.buffer.DataType.UINT32;
        }
        else if(dataType.equals(arrow.uint64())) {
            return org.nd4j.linalg.api.buffer.DataType.UINT64;

        }
        else if(dataType.equals(arrow.int8())) {
            return org.nd4j.linalg.api.buffer.DataType.BYTE;
        }
        else if(dataType.equals(arrow.int16())) {
            return org.nd4j.linalg.api.buffer.DataType.SHORT;
        }
        else if(dataType.equals(arrow.int32())) {
            return org.nd4j.linalg.api.buffer.DataType.INT;
        }
        else if(dataType.equals(arrow.int64())) {
            return org.nd4j.linalg.api.buffer.DataType.LONG;
        }
        else if(dataType.equals(arrow.float16())) {
            return org.nd4j.linalg.api.buffer.DataType.HALF;
        }
        else if(dataType.equals(arrow.float32())) {
            return org.nd4j.linalg.api.buffer.DataType.FLOAT;
        }
        else if(dataType.equals(arrow.float64())) {
            return org.nd4j.linalg.api.buffer.DataType.DOUBLE;
        }
        else if(dataType.equals(arrow.date32())) {
            throw new IllegalArgumentException("Unable to convert type " + dataType.name());
        }
        else if(dataType.equals(arrow.date64())) {
            throw new IllegalArgumentException("Unable to convert type " + dataType.name());
        }
        else if(dataType.equals(arrow.day_time_interval())) {
            throw new IllegalArgumentException("Unable to convert type " + dataType.name());

        }
        else if(dataType.equals(arrow.large_utf8())) {
            return org.nd4j.linalg.api.buffer.DataType.UTF8;
        }
        else if(dataType.equals(arrow.utf8())) {
            return org.nd4j.linalg.api.buffer.DataType.UTF8;
        }
        else if(dataType.equals(arrow.binary())) {
            return org.nd4j.linalg.api.buffer.DataType.BYTE;
        }
        else {
            throw new IllegalArgumentException("Unable to convert type " + dataType.name());
        }
    }

    /**
     *
     * @param arrowBuffer
     * @param dataType
     * @return
     */
    public static DataBuffer fromArrowBuffer(ArrowBuffer arrowBuffer,DataType dataType) {
        org.nd4j.linalg.api.buffer.DataType dataType1 = dataBufferTypeTypeForArrow(dataType);
        if(dataType1 != org.nd4j.linalg.api.buffer.DataType.UTF8) {
            BytePointer bytePointer = arrowBuffer.data().capacity(arrowBuffer.size() * dataType1.width());
            return Nd4j.createBuffer(bytePointer,arrowBuffer.size(),dataBufferTypeTypeForArrow(dataType));

        }
        else {
            BytePointer bytePointer = arrowBuffer.data();
            return Nd4j.createBuffer(bytePointer,arrowBuffer.size(),dataBufferTypeTypeForArrow(dataType));

        }
    }

    /**
     * Create a {@link Pair}
     * of {@link ArrowBuffer} and {@link org.nd4j.linalg.api.buffer.DataType}
     * based on the input {@link DataBuffer}
     * @param dataBuffer the input data buffer
     * @return the pair
     */
    public static Pair<ArrowBuffer,DataType> fromNd4jBuffer(DataBuffer dataBuffer) {
        BytePointer bytePointer = new BytePointer(dataBuffer.pointer());
        ArrowBuffer arrowBuffer = new ArrowBuffer(bytePointer,dataBuffer.length());
        return Pair.of(arrowBuffer,arrowDataTypeForNd4j(dataBuffer.dataType()));
    }


    /**
     * Creates an {@link INDArray} from an arrow {@link Array}
     * @param array the input {@link Array}
     * @return the equivalent {@link INDArray} zero copied
     */
    public static INDArray ndarrayFromArrowArray(FlatArray array) {
        if(array instanceof PrimitiveArray) {
            PrimitiveArray primitiveArray = (PrimitiveArray) array;
            ArrowBuffer arrowBuffer = primitiveArray.values();
            DataBuffer nd4jBuffer = fromArrowBuffer(arrowBuffer,array.data().type());
            return Nd4j.create(nd4jBuffer,1,nd4jBuffer.length());
        }
        else {
            StringArray stringArray = (StringArray) array;
            ArrowBuffer arrowBuffer = stringArray.value_data().capacity(array.capacity()).limit(array.limit());
            DataBuffer nd4jBuffer = fromArrowBuffer(arrowBuffer,array.data().type());
            return Nd4j.create(nd4jBuffer,1,nd4jBuffer.length());
        }

    }

    /**
     * Create an {@link Array}
     * from the given {@link INDArray}
     * with zero copy
     * @param input the input {@link INDArray}
     * @return the equivalent wrapped {@link Array}
     * for the given input {@link INDArray}
     */
    public static FlatArray arrayFromExistingINDArray(INDArray input) {
        Pair<ArrowBuffer, DataType> fromNd4jBuffer = fromNd4jBuffer(input.data());
        ArrowBuffer arrowBuffer = fromNd4jBuffer.getFirst();
        return createArrayFromArrayData(arrowBuffer,input.dataType());
    }


    /**
     * Create an {@link ArrowBuffer} and {@link org.nd4j.linalg.api.buffer.DataType}
     * pair for the offsets of a string/utf8 buffer. The databuffer
     * will come from {@link Utf8Buffer#binaryOffsets()}
     * @param stringBuffer the input data buffer where offsets are. The
     *                     input data buffer must be a Utf8 buffer
     * @return the arrow buffer for the offsets accompanied by the data type
     */
    public static Pair<ArrowBuffer, DataType> arrowBufferForStringOffsets(DataBuffer stringBuffer) {
        Preconditions.checkState(stringBuffer.dataType() == org.nd4j.linalg.api.buffer.DataType.UTF8,"Passed in data buffer has to be a utf8 buffer.");
        DataBuffer offsets = stringBuffer.binaryOffsets();
        return fromNd4jBuffer(offsets);
    }

    /**
     * Create an {@link Array}
     * with the passed in {@link ArrayData}
     *
     * @param numElements the number of elements in the array
     * @param arrowBuffer the array data to create the {@link Array} from
     * @param offsets  the offsets for each string
     * @param dataType the {@link DataType} for the array
     * @return the created {@link Array}
     */
    public static FlatArray createArrayFromArrayData(long numElements, ArrowBuffer arrowBuffer, ArrowBuffer offsets, org.nd4j.linalg.api.buffer.DataType dataType) {
        switch (dataType) {
            case UTF8:
                //note the size - 1 here is due to appending the final boundary
                ArrowBuffer nullVectorBitMap = new ArrowBuffer(new byte[(int) numElements],numElements);
                nullVectorBitMap.fill(1);
                return new StringArray(numElements,offsets,arrowBuffer,nullVectorBitMap,0,0);
            default:
                throw new IllegalArgumentException("Illegal type for array creation. For other data types, please avoid specifying offsets." + dataType);

        }
    }

    /**
     * Create an {@link Array}
     * with the passed in {@link ArrayData}
     * @param arrowBuffer the array data to create the {@link Array} from
     * @param dataType the {@link DataType} for the array
     * @return the created {@link Array}
     */
    public static FlatArray createArrayFromArrayData(ArrowBuffer arrowBuffer, org.nd4j.linalg.api.buffer.DataType dataType) {
        ArrayData arrayData = arrayDataFromArrowBuffer(arrowBuffer,arrowDataTypeForNd4j(dataType), true);
        FlatArray flatArray = null;
        switch (dataType) {
            case DOUBLE:
                flatArray = new DoubleArray(arrayData);
                break;
            case BOOL:
                flatArray = new BooleanArray(arrayData);
                break;
            case FLOAT:
                flatArray = new FloatArray(arrayData);
                break;
            case INT:
                flatArray = new Int32Array(arrayData);
                break;
            case UTF8:
                throw new UnsupportedOperationException("Please use createArrayFromArrayData that forces specifications of offsets.");
            case LONG:
                flatArray = new Int64Array(arrayData);
                break;
            case UINT32:
                flatArray = new UInt32Array(arrayData);
                break;
            case HALF:
                flatArray = new HalfFloatArray(arrayData);
                break;
            case UINT64:
                flatArray = new UInt64Array(arrayData);
                break;
            case BYTE:
                flatArray = new BinaryArray(arrayData);
                break;
            case UINT16:
                flatArray = new UInt16Array(arrayData);
                break;


        }

        return flatArray;
    }



    /**
     * Create array data for a given arrow buffer and data type
     * @param arrowBuffer
     * @param dataType
     * @param nullBitMaskIncluded
     * @return
     */
    public static ArrayData arrayDataFromArrowBuffer(ArrowBuffer arrowBuffer, DataType dataType, boolean nullBitMaskIncluded) {
       if(nullBitMaskIncluded) {
           ArrowBuffer nullVectorBitMap = new ArrowBuffer(new byte[(int) arrowBuffer.size()],arrowBuffer.size());
           //all items are present
           nullVectorBitMap.fill(1);
           ArrowBufferVector arrowBufferVector = new ArrowBufferVector(nullVectorBitMap,arrowBuffer);
           return ArrayData.Make(dataType,arrowBufferVector.size(),arrowBufferVector,0,0);
       }
       else {
           ArrowBufferVector arrowBufferVector = new ArrowBufferVector(arrowBuffer);
           return ArrayData.Make(dataType,arrowBufferVector.size(),arrowBufferVector,0,0);
       }

    }


}
