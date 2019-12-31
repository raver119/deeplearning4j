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

import org.bytedeco.arrow.ArrayVisitor;
import org.bytedeco.arrow.DataType;
import org.bytedeco.arrow.PrimitiveArray;
import org.datavec.api.transform.ColumnType;

import java.util.Comparator;

public interface DataVecColumn<T> extends Iterable<T>, Comparator<T> {

    ColumnType type();

    PrimitiveArray values();

    DataType arrowDataType();

    String name();

    DataVecColumn op(String name, DataVecColumn[] columnParams, ColumnType outputType, Object... otherArgs);

    boolean contains(T input);

    default boolean rowIsNull(int row) {
        return values().IsNull(row);
    }

    default long numValuesMissing() {
        return values().null_count();
    }
}
