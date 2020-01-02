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
import org.bytedeco.arrow.Int32Array;
import org.datavec.api.transform.ColumnType;
import org.datavec.arrow.table.DataVecArrowUtils;
import org.datavec.arrow.table.column.BaseDataVecColumn;
import org.nd4j.linalg.collection.IntArrayKeyMap.IntArray;

import java.util.Iterator;

import static org.bytedeco.arrow.global.arrow.int32;

public class IntColumn extends BaseDataVecColumn<Integer> {

    private Int32Array intArray;

    public IntColumn(String name, ChunkedArray chunkedArray) {
        super(name, chunkedArray);
        this.intArray = (Int32Array) chunkedArray.chunk(0);
    }

    public IntColumn(String name, FlatArray values) {
        super(name, values);
        this.intArray = (Int32Array) values;
    }

    public IntColumn(String name, Integer[] input) {
        super(name, input);
        setValues(input);
    }

    @Override
    public void setValues(Integer[] values) {
        this.values = DataVecArrowUtils.convertIntArray(values);
        this.chunkedArray = new ChunkedArray(this.values);
        this.intArray = (Int32Array) this.values;
    }

    @Override
    public Integer elementAtRow(int rowNumber) {
        return intArray.Value(rowNumber);
    }

    @Override
    public ColumnType type() {
        return ColumnType.Integer;
    }

    @Override
    public DataType arrowDataType() {
        return int32();
    }

    @Override
    public boolean contains(Integer input) {
        return false;
    }

    @Override
    public Iterator<Integer> iterator() {
        return null;
    }

    @Override
    public int compare(Integer o1, Integer o2) {
        return Integer.compare(o1,o2);
    }
}
