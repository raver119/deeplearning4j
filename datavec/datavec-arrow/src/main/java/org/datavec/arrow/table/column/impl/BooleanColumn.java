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
import org.nd4j.arrow.ByteDecoArrowSerde;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Iterator;
import java.util.List;

/**
 * Boolean type column
 *
 * @author Adam Gibson
 */
public class BooleanColumn extends BaseDataVecColumn<Boolean> {

    private BooleanArray booleanArray;

    public BooleanColumn(String name, ChunkedArray chunkedArray) {
        super(name, chunkedArray);
        this.booleanArray = new BooleanArray(chunkedArray.chunk(0));
        this.length = booleanArray.data().buffers().get()[1].size();

    }

    public BooleanColumn(String name, FlatArray values) {
        super(name, values);
        this.booleanArray = (BooleanArray) values;
        this.length = booleanArray.data().buffers().get()[1].size();
    }

    public BooleanColumn(String name, Boolean[] input) {
        super(name, input);
    }

    public BooleanColumn(String name, List<Boolean> input) {
        super(name, input);
    }

    @Override
    public void setValues(Boolean[] values) {
        this.values = DataVecArrowUtils.convertBooleanArray(values);
        this.chunkedArray = new ChunkedArray(this.values);
        this.booleanArray = (BooleanArray) this.values;
        this.length = booleanArray.data().buffers().get()[1].size();

    }

    @Override
    public void setValues(List<Boolean> values) {
        this.values = DataVecArrowUtils.convertBooleanArray(values);
        this.chunkedArray = new ChunkedArray(this.values);
        this.booleanArray = (BooleanArray) this.values;
        this.length = booleanArray.data().buffers().get()[1].size();

    }

    @Override
    public INDArray toNdArray() {
        DataBuffer dataBuffer = ByteDecoArrowSerde.fromArrowBuffer(booleanArray.values(),arrowDataType());
        INDArray ret =  Nd4j.create(dataBuffer);
        return ret;
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
    public int compare(Boolean o1, Boolean o2) {
        return Boolean.compare(o1,o2);
    }
}
