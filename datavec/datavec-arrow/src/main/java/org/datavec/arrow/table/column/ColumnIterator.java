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

package org.datavec.arrow.table.column;

import org.datavec.arrow.table.column.DataVecColumn;

import java.util.Iterator;

/**
 * Simple iterator over a column.
 * @param <T>
 */
public class ColumnIterator<T> implements Iterator<T> {

    private DataVecColumn<T> dataVecColumn;
    private int currRow;

    public ColumnIterator(DataVecColumn<T> dataVecColumn) {
        this.dataVecColumn = dataVecColumn;
    }

    @Override
    public boolean hasNext() {
        return currRow < dataVecColumn.rows();
    }

    @Override
    public T next() {
        T ret = dataVecColumn.elementAtRow(currRow);
        currRow++;
        return ret;
    }
}
