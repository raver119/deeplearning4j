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
import org.bytedeco.arrow.PrimitiveArray;
import org.datavec.api.transform.ColumnType;

import static org.nd4j.arrow.Nd4jArrowOpRunner.runOpOn;

public abstract class BaseDataVecColumn implements DataVecColumn  {

    protected String name;
    protected PrimitiveArray values;
    protected ChunkedArray chunkedArray;

    public BaseDataVecColumn(String name,ChunkedArray chunkedArray) {
        this.name = name;
        this.chunkedArray = chunkedArray;
    }

    public BaseDataVecColumn(String name, PrimitiveArray values) {
        this.name = name;
        this.values = values;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public PrimitiveArray values() {
        return values;
    }

    @Override
    public DataVecColumn op(String name, DataVecColumn[] columnParams, ColumnType outputType, Object... otherArgs) {
        PrimitiveArray[] primitiveArrays = new PrimitiveArray[columnParams.length];
        for(int i = 0; i < columnParams.length; i++) {
            primitiveArrays[i] = columnParams[i].values();
        }

        PrimitiveArray[] primitiveArrays1 = runOpOn(primitiveArrays, name, otherArgs);

        return null;
    }

}
