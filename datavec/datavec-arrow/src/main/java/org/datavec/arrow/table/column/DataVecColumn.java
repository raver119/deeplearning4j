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
import org.bytedeco.arrow.DataType;
import org.bytedeco.arrow.FlatArray;
import org.datavec.api.transform.ColumnType;

import java.util.Comparator;

/**
 * A column abstraction on top of {@link org.nd4j.linalg.api.ndarray.INDArray}
 * @param <T>
 *
 * @author Adam Gibson
 */
public interface DataVecColumn<T> extends Iterable<T>, Comparator<T> {


    /**
     * Returns the element at the given row number
     * @param rowNumber the element at the given row number
     * @return
     */
    T elementAtRow(int rowNumber);

    /**
     * The column type
     * @return
     */
    ColumnType type();

    /**
     * The arrow representation
     * of the values for the column
     * @return the values
     */
    FlatArray values();

    /**
     *
     * @return
     */
    ChunkedArray chunkedValues();

    /**
     *
     * @return
     */
    DataType arrowDataType();

    /**
     * The column name
     * @return
     */
    String name();

    DataVecColumn[] op(String opName, DataVecColumn[] columnParams, String[] outputColumnNames, Object... otherArgs);

    /**
     * Returns true if the input is contained in the
     * column or not
     * @param input the input to test for
     * @return
     */
    boolean contains(T input);

    /**
     * Returns true if the given row is null
     * @param row the row to test for
     * @return
     */
    default boolean rowIsNull(int row) {
        return values().IsNull(row);
    }

    /**
     * Returns the number of missing values
     * @return
     */
    default long numValuesMissing() {
        return values().null_count();
    }

    /**
     * Returns the number of rows in the column
     * @return
     */
    default long rows() {
        return values().data().length();
    }
}
