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

import org.datavec.api.transform.ColumnType;
import org.datavec.arrow.table.column.DataVecColumn;
import org.datavec.arrow.table.column.impl.*;
import org.datavec.arrow.table.row.Row;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TableTests {

    @Test
    public void testTable() {
        int count = 0;
        ColumnType[] columnTypes = new ColumnType[] {
                ColumnType.Integer,
                ColumnType.Double,
                ColumnType.Float,
                ColumnType.Boolean,
                ColumnType.String
        };

        DataVecColumn[] dataVecColumns = new DataVecColumn[columnTypes.length];
        DataVecColumn[] dataVecColumnsList = new DataVecColumn[columnTypes.length];

        for(ColumnType columnType : columnTypes) {
            switch(columnType) {
                case Double:
                    dataVecColumns[count] = new DoubleColumn(columnType.name().toLowerCase(),new Double[]{1.0});
                    dataVecColumnsList[count] = new DoubleColumn(columnType.name().toLowerCase(), Arrays.asList(1.0));
                    break;
                case Float:
                    dataVecColumns[count] = new FloatColumn(columnType.name().toLowerCase(),new Float[]{1.0f});
                    dataVecColumnsList[count] = new FloatColumn(columnType.name().toLowerCase(),Arrays.asList(1.0f));
                    break;
                case Boolean:
                    dataVecColumns[count] = new BooleanColumn(columnType.name().toLowerCase(),new Boolean[]{true});
                    dataVecColumnsList[count] = new BooleanColumn(columnType.name().toLowerCase(),Arrays.asList(true));
                    break;
                case String:
                    dataVecColumns[count] = new StringColumn(columnType.name().toLowerCase(),new String[]{"1.0"});
                    dataVecColumnsList[count] = new StringColumn(columnType.name().toLowerCase(),Arrays.asList("1.0"));
                    break;
                case Long:
                    dataVecColumns[count] = new LongColumn(columnType.name().toLowerCase(),new Long[]{1L});
                    dataVecColumnsList[count] = new LongColumn(columnType.name().toLowerCase(),Arrays.asList(1L));
                    break;
                case Integer:
                    dataVecColumns[count] = new IntColumn(columnType.name().toUpperCase(),new Integer[]{1});
                    dataVecColumnsList[count] = new IntColumn(columnType.name().toUpperCase(),Arrays.asList(1));
                    break;

            }

            assertEquals(1,dataVecColumns[count].rows());
            assertEquals("Column type of " + columnType + " has wrong number of rows",1,dataVecColumns[count].rows());
            count++;
        }

        DataVecTable dataVecTable1 = DataVecTable.create(dataVecColumns);
        assertEquals(columnTypes.length,dataVecTable1.numColumns());
        DataVecTable dataVecTableList = DataVecTable.create(dataVecColumnsList);

        Row row = dataVecTable1.row(0);
        Row row2 = dataVecTableList.row(0);
        assertEquals(1.0d,row.elementAtColumn("double"),1e-3);
        assertEquals(1.0f,row.elementAtColumn("float"),1e-3f);
        assertEquals("1.0",row.elementAtColumn("string"));
        assertEquals(true, row.elementAtColumn("boolean"));

        assertEquals(1.0d,row2.elementAtColumn("double"),1e-3);
        assertEquals(1.0f,row2.elementAtColumn("float"),1e-3f);
        assertEquals("1.0",row2.elementAtColumn("string"));
        assertEquals(true, row2.elementAtColumn("boolean"));



        for(int i = 0; i < row.columnNames().size(); i++) {
            assertTrue(row.elementAtColumn(i).equals(row.elementAtColumn(row.columnNames().get(i))));
            assertTrue(dataVecTable1.column(i).contains(row.elementAtColumn(i)));
            INDArray arr2 = dataVecTable1.column(i).toNdArray();
            assertEquals(dataVecTable1.column(1).rows(),arr2.length());
            List list = dataVecTable1.column(i).toList();
            assertEquals(1,list.size());


            assertTrue(row2.elementAtColumn(i).equals(row2.elementAtColumn(row2.columnNames().get(i))));
            assertTrue(dataVecTableList.column(i).contains(row2.elementAtColumn(i)));
            arr2 = dataVecTableList.column(i).toNdArray();
            assertEquals(dataVecTableList.column(1).rows(),arr2.length());
            list = dataVecTableList.column(i).toList();
            assertEquals(1,list.size());

        }

        assertEquals(1,dataVecTable1.numRows());

    }
}
