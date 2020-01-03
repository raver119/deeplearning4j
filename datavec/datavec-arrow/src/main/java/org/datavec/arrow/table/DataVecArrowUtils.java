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

import org.apache.commons.lang3.ArrayUtils;
import org.bytedeco.arrow.*;
import org.bytedeco.arrow.global.arrow;
import org.bytedeco.javacpp.BytePointer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.Schema.Builder;
import org.datavec.arrow.table.column.DataVecColumn;
import org.datavec.arrow.table.column.impl.*;
import org.nd4j.arrow.ByteDecoArrowSerde;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

import java.util.TimeZone;

import static org.bytedeco.arrow.global.arrow.*;
import static org.nd4j.arrow.ByteDecoArrowSerde.fromArrowBuffer;

/**
 * Utilities for interop between data vec types
 * and arrow types.
 *
 * @author Adam Gibson
 */
public class DataVecArrowUtils {


    /**
     *
     * @param schema
     * @param data
     * @return
     */
    public static Table tableFromSchema(Schema schema,ChunkedArrayVector data) {
        return tableFromSchema(schema,data,data.size());
    }

    /**
     *
     * @param schema
     * @param data
     * @param numRows
     * @return
     */
    public static Table tableFromSchema(Schema schema,ChunkedArrayVector data,long numRows) {
        return Table.Make(toArrowSchema(schema),data,numRows);
    }

    /**
     *
     * @param schema
     * @param arrayVector
     * @param numRows
     * @return
     */
    public static Table tableFromSchema(Schema schema, ArrayVector arrayVector,long numRows) {
        return Table.Make(toArrowSchema(schema),arrayVector,numRows);
    }

    /**
     *
     * @param schema
     * @param arrayVector
     * @return
     */
    public static Table tableFromSchema(Schema schema, ArrayVector arrayVector) {
        return tableFromSchema(schema,arrayVector,arrayVector.size());
    }


    /**
     * Convert an existing data vec {@link Schema}
     * to an {@link  org.bytedeco.arrow.Schema }
     * @param schema the input schema
     * @return the arrow schema
     */
    public static org.bytedeco.arrow.Schema toArrowSchema(Schema schema) {
        Field[] fields = new Field[schema.numColumns()];
        FieldVector schemaVector = new FieldVector(fields);
        for(int i = 0; i < schema.numColumns(); i++) {
            switch(schema.getType(i)) {
                case Double:
                    fields[i] = new Field(schema.getName(i),float64());
                    break;
                case NDArray:
                    fields[i] = new Field(schema.getName(i),binary());
                    break;
                case Bytes:
                    fields[i] = new Field(schema.getName(i),binary());
                    break;
                case String:
                    fields[i] = new Field(schema.getName(i),utf8());
                    break;
                case Integer:
                    fields[i] = new Field(schema.getName(i),int32());
                    break;
                case Time:
                    //note datavec times are stored as longs
                    fields[i] = new Field(schema.getName(i),int64());
                    break;
                case Categorical:
                    fields[i] = new Field(schema.getName(i),utf8());
                    break;
                case Float:
                    fields[i] = new Field(schema.getName(i),float32());
                    break;
                case Long:
                    fields[i] = new Field(schema.getName(i),int64());
                    break;
                case Boolean:
                    fields[i] = new Field(schema.getName(i),_boolean());
                    break;
            }
        }

        return new org.bytedeco.arrow.Schema(schemaVector);
    }


    /**
     * Convert the given input
     * to a boolean array
     * @param array the input
     * @return the equivalent boolean data
     */
    public static boolean[] convertArrayToBoolean(FlatArray array) {
        PrimitiveArray primitiveArray = (PrimitiveArray) array;
        ArrowBuffer arrowBuffer = primitiveArray.values().capacity(array.capacity()).limit(array.limit());
        DataBuffer nd4jBuffer = fromArrowBuffer(arrowBuffer,array.data().type());
        return nd4jBuffer.asBoolean();
    }

    /**
     * Convert the given input
     * to a float array
     * @param array the input
     * @return the equivalent float data
     */
    public static float[] convertArrayToFloat(FlatArray array) {
        PrimitiveArray primitiveArray = (PrimitiveArray) array;
        ArrowBuffer arrowBuffer = primitiveArray.values().capacity(array.capacity()).limit(array.limit());
        DataBuffer nd4jBuffer = fromArrowBuffer(arrowBuffer,array.data().type());
        return nd4jBuffer.asFloat();
    }

    /**
     * Convert the given input
     * to a double array
     * @param array the input
     * @return the equivalent double data
     */
    public static double[] convertArrayToDouble(FlatArray array) {
        PrimitiveArray primitiveArray = (PrimitiveArray) array;
        ArrowBuffer arrowBuffer = primitiveArray.values().capacity(array.capacity()).limit(array.limit());
        DataBuffer nd4jBuffer = fromArrowBuffer(arrowBuffer,array.data().type());
        return nd4jBuffer.asDouble();
    }


    public static String elementAt(StringArray stringArray,long i) {
        long valLength = stringArray.value_length(i);
        long offset = stringArray.value_offset(i);
        ArrowBuffer currData = stringArray.value_data();
        long masksAndOffsets = stringArray.value_offsets().size() * 2;
        return currData.data().position(offset + masksAndOffsets)
                .capacity(valLength)
                .limit(offset + masksAndOffsets + valLength)
                .getString();
    }

    /**
     * Convert the given input
     * to a string array
     * @param array the input
     * @return the equivalent string data
     */
    public static String[] convertArrayToString(FlatArray array) {
        StringArray primitiveArray = (StringArray) array;
        String[] ret = new String[(int) primitiveArray.length()];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = elementAt(primitiveArray,i);
        }

        return ret;
    }

    /**
     * Convert the given input
     * to a long array
     * @param array the input
     * @return the equivalent long data
     */
    public static long[] convertArrayToLong(FlatArray array) {
        PrimitiveArray primitiveArray = (PrimitiveArray) array;
        ArrowBuffer arrowBuffer = primitiveArray.values().capacity(array.capacity()).limit(array.limit());
        DataBuffer nd4jBuffer = fromArrowBuffer(arrowBuffer,array.data().type());
        return nd4jBuffer.asLong();
    }

    /**
     * Convert the given input
     * to a int array
     * @param array the input
     * @return the equivalent int data
     */
    public static int[] convertArrayToInt(FlatArray array) {
        PrimitiveArray primitiveArray = (PrimitiveArray) array;
        ArrowBuffer arrowBuffer = primitiveArray.values().capacity(array.capacity()).limit(array.limit());
        DataBuffer nd4jBuffer = fromArrowBuffer(arrowBuffer,array.data().type());
        return nd4jBuffer.asInt();
    }

    /**
     * Convert a boolean array to a {@link BooleanArray}
     * @param input the input
     * @return the converted array
     */
    public static FlatArray convertBooleanArray(boolean[] input) {
        DataBuffer dataBuffer = Nd4j.createBufferOfType(org.nd4j.linalg.api.buffer.DataType.BOOL,input);
        ArrowBuffer arrowBuffer = new ArrowBuffer(new BytePointer(dataBuffer.pointer()),input.length);
        return ByteDecoArrowSerde.createArrayFromArrayData(arrowBuffer,dataBuffer.dataType());
    }

    /**
     * Convert a boolean array to a {@link BooleanArray}
     * @param input the input
     * @return the converted array
     */
    public static FlatArray convertBooleanArray(Boolean[] input) {
        return convertBooleanArray(ArrayUtils.toPrimitive(input));
    }

    /**
     * Convert a long array to a {@link Int64Array}
     * @param input the input
     * @return the converted array
     */
    public static FlatArray convertLongArray(Long[] input) {
        return convertLongArray(ArrayUtils.toPrimitive(input));
    }

    /**
     * Convert a long array to a {@link Int64Array}
     * @param input the input
     * @return the converted array
     */
    public static FlatArray convertLongArray(long[] input) {
        DataBuffer dataBuffer = Nd4j.createBuffer(input);
        ArrowBuffer arrowBuffer = new ArrowBuffer(new BytePointer(dataBuffer.pointer()),input.length);
        return ByteDecoArrowSerde.createArrayFromArrayData(arrowBuffer,dataBuffer.dataType());
    }

    /**
     * Convert a double array to a {@link DoubleArray}
     * @param input the input
     * @return the converted array
     */
    public static FlatArray convertDoubleArray(double[] input) {
        DataBuffer dataBuffer = Nd4j.createBuffer(input);
        ArrowBuffer arrowBuffer = new ArrowBuffer(new BytePointer(dataBuffer.pointer()),dataBuffer.byteLength());
        return ByteDecoArrowSerde.createArrayFromArrayData(arrowBuffer,dataBuffer.dataType());
    }


    /**
     * Convert a double array to a {@link DoubleArray}
     * @param input the input
     * @return the converted array
     */
    public static FlatArray convertDoubleArray(Double[] input) {
        return convertDoubleArray(ArrayUtils.toPrimitive(input));
    }
    /**
     * Convert a float array to a {@link FloatArray}
     * @param input the input
     * @return the converted array
     */
    public static FlatArray convertFloatArray(float[] input) {
        DataBuffer dataBuffer = Nd4j.createBuffer(input);
        ArrowBuffer arrowBuffer = new ArrowBuffer(new BytePointer(dataBuffer.pointer()),input.length);
        return ByteDecoArrowSerde.createArrayFromArrayData(arrowBuffer,dataBuffer.dataType());
    }

    /**
     * Convert a float array to a {@link FloatArray}
     * @param input the input
     * @return the converted array
     */
    public static FlatArray convertFloatArray(Float[] input) {
        return convertFloatArray(ArrayUtils.toPrimitive(input));
    }

    /**
     * Convert an int array to a {@link Int32Array}
     * @param input the input
     * @return the converted array
     */
    public static FlatArray convertIntArray(int[] input) {
        DataBuffer dataBuffer = Nd4j.createBuffer(input);
        ArrowBuffer arrowBuffer = new ArrowBuffer(new BytePointer(dataBuffer.pointer()),input.length);
        return ByteDecoArrowSerde.createArrayFromArrayData(arrowBuffer,dataBuffer.dataType());
    }



    /**
     * Convert an int array to a {@link Int32Array}
     * @param input the input
     * @return the converted array
     */
    public static FlatArray convertIntArray(Integer[] input) {
        return convertIntArray(ArrayUtils.toPrimitive(input));
    }

    /**
     * Convert a string array to a {@link PrimitiveArray}
     * @param input the input data
     * @return the converted array
     */
    public static FlatArray convertStringArray(String[] input) {
        DataBuffer dataBuffer = Nd4j.createBufferOfType(org.nd4j.linalg.api.buffer.DataType.UTF8,input);
        BytePointer bytePointer = new BytePointer(dataBuffer.pointer());
        ArrowBuffer offsets = ByteDecoArrowSerde.arrowBufferForStringOffsets(dataBuffer).getFirst();

        /**
         *  public StringArray(@Cast("int64_t") long length, @Const @SharedPtr @ByRef ArrowBuffer value_offsets,
         *                 @Const @SharedPtr @ByRef ArrowBuffer data,
         *                 @Const @SharedPtr @ByRef(nullValue = "std::shared_ptr<arrow::Buffer>(nullptr)") ArrowBuffer null_bitmap,
         *                 @Cast("int64_t") long null_count/*=arrow::kUnknownNullCount
         @Cast("int64_t") long offset=0)
         */
        ArrowBuffer arrowBuffer = new ArrowBuffer(bytePointer,dataBuffer.byteLength());
        return ByteDecoArrowSerde.createArrayFromArrayData(input.length, arrowBuffer, offsets, dataBuffer.dataType());
    }



    /**
     * Convert a {@link org.bytedeco.arrow.Schema }
     * to a datavec {@link Schema}
     * @param schema the input schema
     * @return the {@link Schema}
     */
    public static Schema toDataVecSchema(org.bytedeco.arrow.Schema schema) {
        Schema.Builder schemaBuilder = new Builder();
        for(int i = 0; i < schema.num_fields(); i++) {
            Field field = schema.field(i);
            DataType dataType = field.type();
            if(dataType.equals(arrow._boolean())) {
                schemaBuilder.addColumnBoolean(field.name());
            }
            else if(dataType.equals(arrow.uint8())) {
                schemaBuilder.addColumnInteger(field.name());
            }
            else if(dataType.equals(arrow.uint16())) {
                schemaBuilder.addColumnInteger(field.name());
            }
            else if(dataType.equals(arrow.uint32())) {
                schemaBuilder.addColumnLong(field.name());
            }
            else if(dataType.equals(arrow.uint64())) {
                schemaBuilder.addColumnLong(field.name());
            }
            else if(dataType.equals(arrow.int8())) {
                schemaBuilder.addColumnInteger(field.name());
            }
            else if(dataType.equals(arrow.int16())) {
                schemaBuilder.addColumnInteger(field.name());
            }
            else if(dataType.equals(arrow.int32())) {
                schemaBuilder.addColumnInteger(field.name());
            }
            else if(dataType.equals(int64())) {
                schemaBuilder.addColumnLong(field.name());
            }
            else if(dataType.equals(arrow.float16())) {
                schemaBuilder.addColumnFloat(field.name());
            }
            else if(dataType.equals(arrow.float32())) {
                schemaBuilder.addColumnFloat(field.name());
            }
            else if(dataType.equals(float64())) {
                schemaBuilder.addColumnDouble(field.name());
            }
            else if(dataType.equals(arrow.date32()) || dataType.equals(arrow.date64())) {
                schemaBuilder.addColumnTime(field.name(), TimeZone.getTimeZone("UTC"));
            }
            else if(dataType.equals(arrow.day_time_interval())) {
                throw new IllegalArgumentException("Unable to convert type " + dataType.name());

            }
            else if(dataType.equals(arrow.large_utf8())) {
                schemaBuilder.addColumnString(field.name());
            }
            else if(dataType.equals(arrow.utf8())) {
                schemaBuilder.addColumnString(field.name());
            }
            else if(dataType.equals(arrow.binary())) {
                schemaBuilder.addColumnBytes(field.name());
            }
            else {
                throw new IllegalArgumentException("Unable to convert type " + dataType.name());
            }
        }

        return schemaBuilder.build();
    }


    /**
     * Convert a set of {@link PrimitiveArray}
     * to {@link DataVecColumn}
     * @param primitiveArrays the primitive arrays
     * @param names the names of the columns
     * @return the equivalent {@link DataVecColumn}s
     * given the types of {@link PrimitiveArray}
     */
    public static DataVecColumn[] convertPrimitiveArraysToColumns(FlatArray[] primitiveArrays, String[] names) {
        Preconditions.checkState(primitiveArrays != null && names != null && primitiveArrays.length == names.length,
                "Arrays and names must not be null and must be same length arrays");
        DataVecColumn[] ret = new DataVecColumn[primitiveArrays.length];
        for(int i = 0; i < ret.length; i++) {
            switch(ByteDecoArrowSerde.dataBufferTypeTypeForArrow(primitiveArrays[i].data().type())) {
                case UTF8:
                    StringColumn stringColumn = new StringColumn(names[i],primitiveArrays[i]);
                    ret[i] = stringColumn;
                    break;
                case INT:
                    IntColumn intColumn = new IntColumn(names[i],primitiveArrays[i]);
                    ret[i] = intColumn;
                    break;
                case DOUBLE:
                    DoubleColumn doubleColumn = new DoubleColumn(names[i],primitiveArrays[i]);
                    ret[i] = doubleColumn;
                    break;
                case LONG:
                    LongColumn longColumn = new LongColumn(names[i],primitiveArrays[i]);
                    ret[i] = longColumn;
                    break;
                case BOOL:
                    BooleanColumn booleanColumn = new BooleanColumn(names[i],primitiveArrays[i]);
                    ret[i] = booleanColumn;
                    break;
                case FLOAT:
                    FloatColumn floatColumn = new FloatColumn(names[i],primitiveArrays[i]);
                    ret[i] = floatColumn;
                    break;

            }
        }

        return ret;
    }

}
