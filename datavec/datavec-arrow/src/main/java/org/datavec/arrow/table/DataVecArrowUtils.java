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

import org.apache.arrow.vector.VarBinaryVector;
import org.bytedeco.arrow.*;
import org.bytedeco.arrow.global.arrow;
import org.bytedeco.javacpp.BytePointer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.Schema.Builder;
import org.nd4j.arrow.ByteDecoArrowSerde;
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
    public static boolean[] convertArrayToBoolean(PrimitiveArray array) {
        ArrowBuffer arrowBuffer = array.values().capacity(array.capacity()).limit(array.limit());
        DataBuffer nd4jBuffer = fromArrowBuffer(arrowBuffer,array.data().type());
        return nd4jBuffer.asBoolean();
    }

    /**
     * Convert the given input
     * to a float array
     * @param array the input
     * @return the equivalent float data
     */
    public static float[] convertArrayToFloat(PrimitiveArray array) {
        ArrowBuffer arrowBuffer = array.values().capacity(array.capacity()).limit(array.limit());
        DataBuffer nd4jBuffer = fromArrowBuffer(arrowBuffer,array.data().type());
        return nd4jBuffer.asFloat();
    }

    /**
     * Convert the given input
     * to a double array
     * @param array the input
     * @return the equivalent double data
     */
    public static double[] convertArrayToDouble(PrimitiveArray array) {
        ArrowBuffer arrowBuffer = array.values().capacity(array.capacity()).limit(array.limit());
        DataBuffer nd4jBuffer = fromArrowBuffer(arrowBuffer,array.data().type());
        return nd4jBuffer.asDouble();
    }

    /**
     * Convert the given input
     * to a string array
     * @param array the input
     * @return the equivalent string data
     */
    public static String[] convertArrayToString(PrimitiveArray array) {
        ArrowBuffer arrowBuffer = array.values();
        DataBuffer nd4jBuffer = fromArrowBuffer(arrowBuffer,array.data().type());
        return nd4jBuffer.asUtf8();
    }

    /**
     * Convert the given input
     * to a long array
     * @param array the input
     * @return the equivalent long data
     */
    public static long[] convertArrayToLong(PrimitiveArray array) {
        ArrowBuffer arrowBuffer = array.values().capacity(array.capacity()).limit(array.limit());
        DataBuffer nd4jBuffer = fromArrowBuffer(arrowBuffer,array.data().type());
        return nd4jBuffer.asLong();
    }

    /**
     * Convert the given input
     * to a int array
     * @param array the input
     * @return the equivalent int data
     */
    public static int[] convertArrayToInt(PrimitiveArray array) {
        ArrowBuffer arrowBuffer = array.values().capacity(array.capacity()).limit(array.limit());
        DataBuffer nd4jBuffer = fromArrowBuffer(arrowBuffer,array.data().type());
        return nd4jBuffer.asInt();
    }

    /**
     * Convert a boolean array to a {@link BooleanArray}
     * @param input the input
     * @return the converted array
     */
    public static PrimitiveArray convertBooleanArray(boolean[] input) {
        DataBuffer dataBuffer = Nd4j.createBufferOfType(org.nd4j.linalg.api.buffer.DataType.BOOL,input);
        ArrowBuffer arrowBuffer = new ArrowBuffer(new BytePointer(dataBuffer.pointer()),input.length);
        return ByteDecoArrowSerde.createArrayFromArrayData(arrowBuffer,dataBuffer.dataType());
    }


    /**
     * Convert a long array to a {@link Int64Array}
     * @param input the input
     * @return the converted array
     */
    public static PrimitiveArray convertLongArray(long[] input) {
        DataBuffer dataBuffer = Nd4j.createBuffer(input);
        ArrowBuffer arrowBuffer = new ArrowBuffer(new BytePointer(dataBuffer.pointer()),input.length);
        return ByteDecoArrowSerde.createArrayFromArrayData(arrowBuffer,dataBuffer.dataType());
    }

    /**
     * Convert a double array to a {@link DoubleArray}
     * @param input the input
     * @return the converted array
     */
    public static PrimitiveArray convertDoubleArray(double[] input) {
        DataBuffer dataBuffer = Nd4j.createBuffer(input);
        ArrowBuffer arrowBuffer = new ArrowBuffer(new BytePointer(dataBuffer.pointer()),dataBuffer.byteLength());
        return ByteDecoArrowSerde.createArrayFromArrayData(arrowBuffer,dataBuffer.dataType());
    }

    /**
     * Convert a float array to a {@link FloatArray}
     * @param input the input
     * @return the converted array
     */
    public static PrimitiveArray convertFloatArray(float[] input) {
        DataBuffer dataBuffer = Nd4j.createBuffer(input);
        ArrowBuffer arrowBuffer = new ArrowBuffer(new BytePointer(dataBuffer.pointer()),input.length);
        return ByteDecoArrowSerde.createArrayFromArrayData(arrowBuffer,dataBuffer.dataType());
    }


    /**
     * Convert an int array to a {@link Int32Array}
     * @param input the input
     * @return the converted array
     */
    public static PrimitiveArray convertIntArray(int[] input) {
        DataBuffer dataBuffer = Nd4j.createBuffer(input);
        ArrowBuffer arrowBuffer = new ArrowBuffer(new BytePointer(dataBuffer.pointer()),input.length);
        return ByteDecoArrowSerde.createArrayFromArrayData(arrowBuffer,dataBuffer.dataType());
    }


    /**
     * Convert a string array to a {@link PrimitiveArray}
     * @param input the input data
     * @return the converted array
     */
    public static PrimitiveArray convertStringArray(String[] input) {
        DataBuffer dataBuffer = Nd4j.createBufferOfType(org.nd4j.linalg.api.buffer.DataType.UTF8,input);
        BytePointer bytePointer = new BytePointer(dataBuffer.pointer());
        ArrowBuffer arrowBuffer = new ArrowBuffer(bytePointer,dataBuffer.byteLength());
        return ByteDecoArrowSerde.createArrayFromArrayData(arrowBuffer,dataBuffer.dataType());
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

}
