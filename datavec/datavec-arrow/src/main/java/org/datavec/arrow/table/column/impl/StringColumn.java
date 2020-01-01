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
import org.datavec.api.transform.sequence.comparator.StringComparator;
import org.datavec.arrow.table.DataVecArrowUtils;
import org.datavec.arrow.table.column.BaseDataVecColumn;
import org.datavec.arrow.table.column.DataVecColumn;

import java.util.Iterator;

import static org.bytedeco.arrow.global.arrow.utf8;
import static org.nd4j.arrow.Nd4jArrowOpRunner.runOpOn;

public class StringColumn extends BaseDataVecColumn<String> {

    public StringColumn(String name, ChunkedArray chunkedArray) {
        super(name, chunkedArray);
    }

    public StringColumn(String name, PrimitiveArray values) {
        super(name, values);
    }

    public StringColumn(String name, String[] input) {
        super(name, input);
    }

    @Override
    public void setValues(String[] values) {
        this.values = DataVecArrowUtils.convertStringArray(values);
        this.chunkedArray = new ChunkedArray(this.values);
    }

    @Override
    public ColumnType type() {
        return ColumnType.String;
    }


    @Override
    public DataType arrowDataType() {
        return utf8();
    }

    @Override
    public boolean contains(String input) {
        return false;
    }


    @Override
    public Iterator<String> iterator() {
        return null;
    }

    @Override
    public int compare(String o1, String o2) {
        return o1.compareTo(o2);
    }
}
