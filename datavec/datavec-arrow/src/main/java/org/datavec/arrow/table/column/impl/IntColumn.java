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
import org.datavec.arrow.table.column.BaseDataVecColumn;

import static org.bytedeco.arrow.global.arrow.int32;

public class IntColumn extends BaseDataVecColumn {

    public IntColumn(String name, ChunkedArray chunkedArray) {
        super(name, chunkedArray);
    }

    public IntColumn(String name, PrimitiveArray values) {
        super(name, values);
    }

    @Override
    public ColumnType type() {
        return ColumnType.Integer;
    }

    @Override
    public DataType arrowDataType() {
        return int32();
    }
}
