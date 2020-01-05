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
import org.bytedeco.arrow.DoubleArray;
import org.bytedeco.arrow.FlatArray;
import org.datavec.api.transform.ColumnType;
import org.datavec.arrow.table.DataVecArrowUtils;
import org.datavec.arrow.table.column.BaseDataVecColumn;
import org.nd4j.arrow.ByteDecoArrowSerde;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Iterator;
import java.util.List;

import static org.bytedeco.arrow.global.arrow.float64;

/**
 * Double type column
 *
 * @author Adam Gibson
 */
public class DoubleColumn extends BaseDataVecColumn<Double> {

    private DoubleArray doubleArray;

    public DoubleColumn(String name, ChunkedArray chunkedArray) {
        super(name, chunkedArray);
        this.doubleArray = new DoubleArray(chunkedArray.chunk(0));
        this.length = doubleArray.data().buffers().get()[1].size();
    }

    public DoubleColumn(String name, FlatArray values) {
        super(name, values);
        this.doubleArray = (DoubleArray) values;
        this.length = doubleArray.data().buffers().get()[1].size();
    }

    public DoubleColumn(String name, Double[] input) {
        super(name, input);
    }

    public DoubleColumn(String name, List<Double> input) {
        super(name, input);
    }

    @Override
    public void setValues(Double[] values) {
        this.values = DataVecArrowUtils.convertDoubleArray(values);
        this.chunkedArray = new ChunkedArray(this.values);
        this.doubleArray = (DoubleArray) this.values;
        this.length = doubleArray.data().buffers().get()[1].size();
    }

    @Override
    public void setValues(List<Double> values) {
        this.values = DataVecArrowUtils.convertDoubleArray(values);
        this.chunkedArray = new ChunkedArray(this.values);
        this.doubleArray = (DoubleArray) this.values;
        this.length = doubleArray.data().buffers().get()[1].size();
    }

    @Override
    public INDArray toNdArray() {
        DataBuffer dataBuffer = ByteDecoArrowSerde.fromArrowBuffer(doubleArray.values(),arrowDataType());
        INDArray ret =  Nd4j.create(dataBuffer);
        return ret;
    }

    @Override
    public Double elementAtRow(int rowNumber) {
        return doubleArray.Value(rowNumber);
    }

    @Override
    public ColumnType type() {
        return ColumnType.Double;
    }

    @Override
    public DataType arrowDataType() {
        return float64();
    }


    @Override
    public int compare(Double o1, Double o2) {
        return Double.compare(o1,o2);
    }
}
