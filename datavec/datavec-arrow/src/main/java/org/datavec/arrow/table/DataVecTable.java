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

import org.bytedeco.arrow.Table;
import org.datavec.api.transform.schema.Schema;
import org.datavec.arrow.table.column.DataVecColumn;
import org.datavec.arrow.table.column.impl.*;

import java.util.LinkedHashMap;
import java.util.Map;

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



    public org.bytedeco.arrow.Schema arrowSchema() {
        return DataVecArrowUtils.toArrowSchema(schema);
    }

    public Schema schema() {
        return schema;
    }

    public DataVecColumn column(String name) {
        return columns.get(name);
    }

    public static DataVecTable create(Table table) {
        return new DataVecTable(table);
    }
}
