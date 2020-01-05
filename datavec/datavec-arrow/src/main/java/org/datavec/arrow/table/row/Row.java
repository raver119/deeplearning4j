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

/**
 * Represents a row in a {@link DataVecTable}
 * This row is just a view of the underlying data.
 *
 * @author Adam Gibson
 */
public interface Row {

    /**
     * The underlying {@link DataVecTable}
     * of the row
     * @return
     */
    DataVecTable table();

    /**
     * The row number of the row
     * @return
     */
    int rowNumber();

    /**
     * Get the element at a particular column
     * @param column the index of the column to get the element at
     * @param <T> the type of return
     * @return
     */
    <T> T elementAtColumn(int column);

    /**
     *
     * @param columnName
     * @param <T>
     * @return
     */
    <T> T elementAtColumn(String columnName);

    /**
     * The column names of the row
     * @return
     */
    List<String> columnNames();

}
