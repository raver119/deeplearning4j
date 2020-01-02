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

import org.bytedeco.arrow.BooleanArray;
import org.bytedeco.arrow.ChunkedArray;
import org.bytedeco.arrow.DataType;
import org.bytedeco.arrow.FlatArray;
import org.bytedeco.arrow.global.arrow;
import org.datavec.api.transform.ColumnType;
import org.datavec.arrow.table.DataVecArrowUtils;
import org.datavec.arrow.table.column.BaseDataVecColumn;

import java.util.Iterator;

public class BooleanColumn extends BaseDataVecColumn<Boolean> {

    private BooleanArray booleanArray;

    public BooleanColumn(String name, ChunkedArray chunkedArray) {
        super(name, chunkedArray);
        this.booleanArray = (BooleanArray) chunkedArray.chunk(0);
    }

    public BooleanColumn(String name, FlatArray values) {
        super(name, values);
        this.booleanArray = (BooleanArray) values;
    }

    public BooleanColumn(String name, Boolean[] input) {
        super(name, input);
        setValues(input);
    }

    @Override
    public void setValues(Boolean[] values) {
        this.values = DataVecArrowUtils.convertBooleanArray(values);
        this.chunkedArray = new ChunkedArray(this.values);
        this.booleanArray = (BooleanArray) this.values;
    }


    @Override
    public Boolean elementAtRow(int rowNumber) {
        return booleanArray.Value(rowNumber);
    }

    @Override
    public ColumnType type() {
        return ColumnType.Boolean;
    }

    @Override
    public DataType arrowDataType() {
        return arrow._boolean();
    }

    @Override
    public boolean contains(Boolean input) {
        return false;
    }

    @Override
    public Iterator<Boolean> iterator() {
        return null;
    }

    @Override
    public int compare(Boolean o1, Boolean o2) {
        return Boolean.compare(o1,o2);
    }
}