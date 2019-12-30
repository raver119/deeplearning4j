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

import org.bytedeco.arrow.*;
import org.bytedeco.arrow.global.arrow;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.Schema.Builder;

import java.util.TimeZone;

import static org.bytedeco.arrow.global.arrow.*;

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
