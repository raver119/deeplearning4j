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

import org.bytedeco.arrow.ChunkedArray;
import org.bytedeco.arrow.FlatArray;
import org.datavec.arrow.table.DataVecArrowUtils;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static org.nd4j.arrow.Nd4jArrowOpRunner.runOpOn;

/**
 * Abstract class for the column.
 * @param <T> the type of the class
 *
 * @author Adam Gibson
 */
public abstract class BaseDataVecColumn<T> implements DataVecColumn<T>  {

    protected String name;
    protected FlatArray values;
    protected ChunkedArray chunkedArray;
    protected  long length;

    public BaseDataVecColumn(String name,List<T> input)  {
        setValues(input);
        this.name = name;
    }

    public BaseDataVecColumn(String name,T[] input)  {
        setValues(input);
        this.name = name;
    }

    public BaseDataVecColumn(String name,ChunkedArray chunkedArray) {
        this.name = name;
        this.chunkedArray = chunkedArray;
    }

    public BaseDataVecColumn(String name, FlatArray values) {
        this.name = name;
        this.chunkedArray = new ChunkedArray(values);
        this.values = values;
    }

    @Override
    public long rows() {
        return length;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public FlatArray values() {
        return values;
    }

    @Override
    public DataVecColumn[] op(String opName, DataVecColumn[] columnParams, String[] outputColumnNames, Object... otherArgs) {
        FlatArray[] primitiveArrays = new FlatArray[columnParams.length];
        for(int i = 0; i < columnParams.length; i++) {
            primitiveArrays[i] = columnParams[i].values();
        }

        return DataVecArrowUtils.convertPrimitiveArraysToColumns(runOpOn(primitiveArrays, opName, otherArgs),outputColumnNames);
    }


    @Override
    public boolean contains(T input) {
        for(int i = 0; i < rows(); i++) {
            if(elementAtRow(i).equals(input))
                return true;
        }
        return false;
    }

    @Override
    public ChunkedArray chunkedValues() {
        return chunkedArray;
    }

    @Override
    public Iterator<T> iterator() {
        return new ColumnIterator<>(this);
    }

    @Override
    public List<T> toList() {
        List<T> ret = new ArrayList<>();
        for(T item : this) {
            ret.add(item);
        }

        return ret;
    }

    /**
     * Set the values for this column using
     * the specified array
     * @param values the array of values to use
     */
    public abstract void setValues(List<T> values);

    /**
     * Set the values for this column using
     * the specified array
     * @param values the array of values to use
     */
    public abstract void setValues(T[] values);

}
