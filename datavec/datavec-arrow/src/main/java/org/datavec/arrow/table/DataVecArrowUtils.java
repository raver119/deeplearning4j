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
import org.bytedeco.javacpp.LongPointer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.Schema.Builder;
import org.datavec.arrow.table.column.DataVecColumn;
import org.datavec.arrow.table.column.impl.*;
import org.nd4j.arrow.ByteDecoArrowSerde;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.primitives.*;

import java.util.List;
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
     * Returns the number of elements in the given
     * {@link FlatArray}.
     * This accesses buffer[buffer.length - 1] and returns its size
     * @param flatArray the flat array to return the number
     *                  of elements for
     * @return
     */
    public static long numberOfElementsInBuffer(FlatArray flatArray) {
        long indexOfBuffer = flatArray.data().buffers().size() - 1;
        Preconditions.checkState(flatArray.data().length() > 1,"Flat array size must be at least size 2.");
        return flatArray.data().buffers().get(indexOfBuffer).size();
    }


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
        FieldVector schemaVector = null;
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

        schemaVector = new FieldVector(fields);
        return new org.bytedeco.arrow.Schema(schemaVector);
    }


    /**
     * Convert the given input
     * to a boolean array
     * @param array the input
     * @return the equivalent boolean data
     */
    public static boolean[] convertArrayToBoolean(FlatArray array) {
        BooleanArray primitiveArray = (BooleanArray) array;
        long length = numberOfElementsInBuffer(array);
        boolean[] ret = new boolean[(int)  length];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = primitiveArray.Value(i);
        }

        return ret;
    }

    /**
     * Convert the given input
     * to a float array
     * @param array the input
     * @return the equivalent float data
     */
    public static float[] convertArrayToFloat(FlatArray array) {
        FloatArray primitiveArray = (FloatArray) array;
        long length = numberOfElementsInBuffer(array);
        float[] ret = new float[(int) length];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = primitiveArray.Value(i);
        }

        return ret;
    }

    /**
     * Convert the given input
     * to a double array
     * @param array the input
     * @return the equivalent double data
     */
    public static double[] convertArrayToDouble(FlatArray array) {
        DoubleArray primitiveArray = (DoubleArray) array;
        long length = numberOfElementsInBuffer(array);
        double[] ret = new double[(int) length];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = primitiveArray.Value(i);
        }
        return ret;
    }


    /**
     * Find the element at a particular ror
     * in the {@link StringArray}
     * @param stringArray the string array to get the item from
     * @param i the index
     * @return the string at the specified index
     */
    public static String elementAt(StringArray stringArray,long i) {
        long numElements = numberOfElementsInBuffer(stringArray);
        return elementAt(stringArray,i,numElements);
    }

    /**
     * Find the element at a particular ror
     * in the {@link StringArray}
     * @param stringArray the string array to get the item from
     * @param i the index
     * @param length  the number of elements
     * @return the string at the specified index
     */
    public static String elementAt(StringArray stringArray,long i,long length) {
        long valLength = stringArray.value_length(i);
        long offset = stringArray.value_offset(i);
        ArrowBuffer currData = stringArray.value_data();
        //offsets: each begin/end for each element that isn't the last
        long offsetSize = (length + 1) * 8;
        return currData.data().position(offset + offsetSize)
                .capacity(valLength)
                .limit(offset + offsetSize + valLength)
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
        long length = numberOfElementsInBuffer(array);
        String[] ret = new String[(int) length];
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
        Int64Array primitiveArray = (Int64Array) array;
        long length = numberOfElementsInBuffer(array);
        long[] ret = new long[(int) length];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = primitiveArray.Value(i);
        }

        return ret;
    }

    /**
     * Convert the given input
     * to a int array
     * @param array the input
     * @return the equivalent int data
     */
    public static int[] convertArrayToInt(FlatArray array) {
        Int32Array primitiveArray = (Int32Array) array;
        long length = numberOfElementsInBuffer(array);
        int[] ret = new int[(int) length];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = primitiveArray.Value(i);
        }
        return ret;
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
     * Convert a boolean array to a {@link BooleanArray}
     * @param input the input
     * @return the converted array
     */
    public static FlatArray convertBooleanArray(List<Boolean> input) {
        return convertBooleanArray(Booleans.toArray(input));
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
    public static FlatArray convertLongArray(List<Long> input) {
        return convertLongArray(Longs.toArray(input));
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
        ArrowBuffer arrowBuffer = new ArrowBuffer(new BytePointer(dataBuffer.pointer()),dataBuffer.length());
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
     * Convert a double array to a {@link DoubleArray}
     * @param input the input
     * @return the converted array
     */
    public static FlatArray convertDoubleArray(List<Double> input) {
        return convertDoubleArray(Doubles.toArray(input));
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
     * Convert a float array to a {@link FloatArray}
     * @param input the input
     * @return the converted array
     */
    public static FlatArray convertFloatArray(List<Float> input) {
        return convertFloatArray(Floats.toArray(input));
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
     * Convert an int array to a {@link Int32Array}
     * @param input the input
     * @return the converted array
     */
    public static FlatArray convertIntArray(List<Integer> input) {
        return convertIntArray(Ints.toArray(input));
    }



    /**
     * Convert a string array to a {@link PrimitiveArray}
     * @param input the input data
     * @return the converted array
     */
    public static FlatArray convertStringArray(List<String> input) {
        return convertStringArray(input.toArray(new String[input.size()]));
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
        ArrowBuffer arrowBuffer = new ArrowBuffer(bytePointer,dataBuffer.length());
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
