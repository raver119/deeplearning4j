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

package org.datavec.arrow.table.column.impl;

import org.bytedeco.arrow.ChunkedArray;
import org.bytedeco.arrow.DataType;
import org.bytedeco.arrow.FlatArray;
import org.bytedeco.arrow.Int64Array;
import org.datavec.api.transform.ColumnType;
import org.datavec.arrow.table.DataVecArrowUtils;
import org.datavec.arrow.table.column.BaseDataVecColumn;

import java.util.Iterator;

import static org.bytedeco.arrow.global.arrow.int64;

/**
 * Long type column
 *
 * @author Adam Gibson
 */
public class LongColumn extends BaseDataVecColumn<Long> {

    private Int64Array int64Array;

    public LongColumn(String name, ChunkedArray chunkedArray) {
        super(name, chunkedArray);
        this.int64Array = new Int64Array(chunkedArray.chunk(0));
        this.length = int64Array.data().buffers().get()[1].size();
    }

    public LongColumn(String name, FlatArray values) {
        super(name, values);
        this.int64Array = (Int64Array) values;
        this.length = int64Array.data().buffers().get()[1].size();
    }

    public LongColumn(String name, Long[] input) {
        super(name, input);
    }

    @Override
    public void setValues(Long[] values) {
        this.values = DataVecArrowUtils.convertLongArray(values);
        this.chunkedArray = new ChunkedArray(this.values);
        this.int64Array = (Int64Array) this.values;
        this.length = int64Array.data().buffers().get()[1].size();
    }

    @Override
    public Long elementAtRow(int rowNumber) {
        return int64Array.Value(rowNumber);
    }

    @Override
    public ColumnType type() {
        return ColumnType.Long;
    }

    @Override
    public DataType arrowDataType() {
        return int64();
    }

    @Override
    public boolean contains(Long input) {
        return false;
    }

    @Override
    public Iterator<Long> iterator() {
        return null;
    }

    @Override
    public int compare(Long o1, Long o2) {
        return Long.compare(o1,o2);
    }
}
