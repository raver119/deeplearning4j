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
import org.bytedeco.arrow.PrimitiveArray;
import org.datavec.api.transform.ColumnType;
import org.datavec.arrow.table.DataVecArrowUtils;
import org.datavec.arrow.table.column.BaseDataVecColumn;
import org.datavec.arrow.table.column.DataVecColumn;

import java.util.Iterator;

import static org.bytedeco.arrow.global.arrow.float32;

public class FloatColumn extends BaseDataVecColumn<Float> {

    public FloatColumn(String name, ChunkedArray chunkedArray) {
        super(name, chunkedArray);
    }

    public FloatColumn(String name, PrimitiveArray values) {
        super(name, values);
    }

    public FloatColumn(String name, Float[] input) {
        super(name, input);
    }

    @Override
    public void setValues(Float[] values) {
        this.values = DataVecArrowUtils.convertFloatArray(values);
        this.chunkedArray = new ChunkedArray(this.values);
    }

    @Override
    public ColumnType type() {
        return ColumnType.Float;
    }

    @Override
    public DataType arrowDataType() {
        return float32();
    }

    @Override
    public boolean contains(Float input) {
        return false;
    }


    @Override
    public Iterator<Float> iterator() {
        return null;
    }

    @Override
    public int compare(Float o1, Float o2) {
        return Float.compare(o1,o2);
    }
}
