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
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.Schema.Builder;
import org.datavec.arrow.table.column.DataVecColumn;
import org.datavec.arrow.table.column.impl.*;
import org.datavec.arrow.table.row.Row;
import org.datavec.arrow.table.row.RowImpl;
import org.nd4j.base.Preconditions;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TimeZone;

/**
 * A table representing a data frame like datastructure
 * for accessing columnar data
 *
 * @author Adam Gibson
 */
public class DataVecTable {

    private Table table;
    private Schema schema;
    private Map<String,DataVecColumn> columns;

    private DataVecTable(Table table) {
        this.table = table;
        this.schema = DataVecArrowUtils.toDataVecSchema(table.schema());
        this.columns = new LinkedHashMap<>();
        for(int i = 0; i < schema.numColumns(); i++) {
            switch(schema.getType(i)) {
                case String:
                    columns.put(schema.getName(i),new StringColumn(schema.getName(i),table.column(i)));
                    break;
                case Boolean:
                    columns.put(schema.getName(i),new BooleanColumn(schema.getName(i),table.column(i)));
                    break;
                case Long:
                    columns.put(schema.getName(i),new LongColumn(schema.getName(i),table.column(i)));
                    break;
                case Float:
                    columns.put(schema.getName(i),new FloatColumn(schema.getName(i),table.column(i)));
                    break;
                case Double:
                    columns.put(schema.getName(i),new DoubleColumn(schema.getName(i),table.column(i)));
                    break;
                case Categorical:
                    columns.put(schema.getName(i),new StringColumn(schema.getName(i),table.column(i)));
                    break;
                case Integer:
                    columns.put(schema.getName(i),new IntColumn(schema.getName(i),table.column(i)));
                    break;
                case Bytes:
                    columns.put(schema.getName(i),new StringColumn(schema.getName(i),table.column(i)));
                    break;
                case Time:
                    columns.put(schema.getName(i),new StringColumn(schema.getName(i),table.column(i)));
                    break;
                case NDArray:
                    columns.put(schema.getName(i),new StringColumn(schema.getName(i),table.column(i)));
                    break;
            }
        }

    }



    public DataVecTable addRow(Row row) {
        Preconditions.checkState(schema.getColumnNames().equals(row.columnNames()));
        Array[] inputData = new Array[schema.numColumns()];
        for(int i = 0; i < schema.numColumns(); i++) {

        }
        //ArrayVector arrayVector = new ArrayVector(inputData);
        throw new UnsupportedOperationException();
    }

    /**
     * Returns the arrow schema {@link org.bytedeco.arrow.Schema}
     * @return
     */
    public org.bytedeco.arrow.Schema arrowSchema() {
        return DataVecArrowUtils.toArrowSchema(schema);
    }

    /**
     * Returns the {@link Schema}
     * for this tbale
     * @return
     */
    public Schema schema() {
        return schema;
    }

    /**
     * Get the name of the column
     * at the specified index
     * @param index the indes to get the column name at
     * @return the name of the column at the specified index
     */
    public String columnNameAt(int index) {
        return schema.getName(index);
    }

    /**
     * Returns the column of the table
     * at the given index
     * @param columnIndex the index of the column
     *                    to get
     * @return the column at the specified index
     */
    public DataVecColumn column(int columnIndex) {
        return column(schema.getName(columnIndex));
    }


    /**
     * Returns the column in the table with
     * the given name
     * @param name the name of the column
     * @return the column with the given name
     */
    public <T> DataVecColumn<T> column(String name) {
        Preconditions.checkState(columns.containsKey(name),"No column named " + name + " present in table!");
        return columns.get(name);
    }



    /**
     * Create a {@link Row}
     * using this table given the row number
     * @param rowNum the row number
     * @return
     */
    public Row row(int rowNum) {
        Row row = new RowImpl(this,rowNum);
        return row;
    }

    /**
     * Create a {@link DataVecTable}
     * using the given {@link Table}
     * @param table the table to use
     * @return the created table
     */
    public static DataVecTable create(Table table) {
        return new DataVecTable(table);
    }


    /**
     * Create a {@link DataVecTable}
     * based on the columns
     * @param columns the input columns
     * @return the created table
     */
    public static DataVecTable create(DataVecColumn...columns) {
        Preconditions.checkNotNull(columns,"Passed in column array was null!");
        Schema.Builder schemaBuilder = new Builder();
        ArrayVector arrayVector = null;
        Array[] arrays = new Array[columns.length];
        for(int i = 0; i < columns.length; i++) {
            Preconditions.checkNotNull(columns[i],"Column " + i + " was null!");
            switch(columns[i].type()) {
                case Boolean:
                    schemaBuilder.addColumnBoolean(columns[i].name());
                    break;
                case Float:
                    schemaBuilder.addColumnFloat(columns[i].name());
                    break;
                case Double:
                    schemaBuilder.addColumnDouble(columns[i].name());
                    break;
                case Integer:
                    schemaBuilder.addColumnInteger(columns[i].name());
                    break;
                case String:
                    schemaBuilder.addColumnString(columns[i].name());
                    break;
                case Long:
                    schemaBuilder.addColumnLong(columns[i].name());
                    break;
                case Time:
                    schemaBuilder.addColumnTime(columns[i].name(), TimeZone.getDefault());
                    break;
            }

            FlatArray flatArray = columns[i].values();
            arrays[i] = flatArray;
        }

        arrayVector = new ArrayVector(arrays);
        Schema dataVecSchema = schemaBuilder.build();
        org.bytedeco.arrow.Schema arrowSchema = DataVecArrowUtils.toArrowSchema(dataVecSchema);
        Table table = Table.Make(arrowSchema,arrayVector);
        return new DataVecTable(table);
    }


    /**
     * Returns the number of rows in the table
     * @return
     */
    public long numRows() {
        return columns.get(schema.getName(0)).rows();
    }

    /**
     * Returns the number of columns in the table
     * @return
     */
    public long numColumns() {
        return table.num_columns();
    }


    /**
     * Create a {@link DataVecColumn} of the specified type
     * @param name the name of the column
     * @param dataWith the data to create teh column with
     * @param <T> the type
     * @return
     */
    public static <T> DataVecColumn<T> createColumnOfType(String name,T[] dataWith) {
        Class<T> clazz = (Class<T>) dataWith[0].getClass();
        DataVecColumn<T> ret = null;
        if(clazz.equals(Boolean.class)) {
            Boolean[] casted = (Boolean[]) dataWith;
            ret = (DataVecColumn<T>) new BooleanColumn(name,casted);
        }
        else if(clazz.equals(Double.class)) {
            Double[] casted = (Double[]) dataWith;
            ret = (DataVecColumn<T>) new DoubleColumn(name,casted);

        }
        else if(clazz.equals(Float.class)) {
            Float[] casted = (Float[]) dataWith;
            ret = (DataVecColumn<T>) new FloatColumn(name,casted);
        }
        else if(clazz.equals(String.class)) {
            String[] casted = (String[]) dataWith;
            ret = (DataVecColumn<T>) new StringColumn(name,casted);
        }
        else if(clazz.equals(Long.class)) {
            Long[] casted = (Long[]) dataWith;
            ret = (DataVecColumn<T>) new LongColumn(name,casted);
        }
        else if(clazz.equals(Integer.class)) {
            Integer[] casted = (Integer[]) dataWith;
            ret = (DataVecColumn<T>) new IntColumn(name,casted);

        }
        else {
            throw new IllegalArgumentException("Illegal type " + clazz.getName());
        }

        return ret;

    }

}
