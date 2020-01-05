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

package org.datavec.arrow.table.row;

import org.datavec.arrow.table.DataVecTable;
import org.datavec.arrow.table.column.DataVecColumn;

import java.util.List;

/**
 * Row implementation.
 * Represents multiple  {@link org.datavec.arrow.table.column.DataVecColumn}
 * that have all
 * of the same row index.
 *
 * @author Adam Gibson
 */
public class RowImpl implements Row {

    private DataVecTable table;
    private int rowNum;

    /**
     * An implementation of a row.
     * @param table the table to provide the view for
     * @param rowNum the row number representative for the
     *               view
     */
    public RowImpl(DataVecTable table, int rowNum) {
        this.table = table;
        this.rowNum = rowNum;
    }

    @Override
    public DataVecTable table() {
        return table;
    }

    @Override
    public int rowNumber() {
        return rowNum;
    }

    @Override
    public <T> T elementAtColumn(int column) {
        return elementAtColumn(table.columnNameAt(column));
    }

    @Override
    public <T> T elementAtColumn(String columnName) {
        DataVecColumn<T> column = table.column(columnName);
        return column.elementAtRow(rowNumber());
    }

    @Override
    public List<String> columnNames() {
        return table.schema().getColumnNames();
    }
}
