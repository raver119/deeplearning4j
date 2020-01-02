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

import java.util.List;

public class RowImpl implements Row {

    private DataVecTable table;
    private int rowNum;

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
        return (T) table.column(column).elementAtRow(rowNumber());
    }

    @Override
    public <T> T elementAtColumn(String columnName) {
        return null;
    }

    @Override
    public List<String> columnNames() {
        return table.schema().getColumnNames();
    }
}
