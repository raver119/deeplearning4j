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

import org.datavec.arrow.table.DataVecTable;
import org.datavec.arrow.table.column.DataVecColumn;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class ColumnTests {

    @Test
    public void testBooleanColumn() {
        assertColumnInput(new Boolean[]{true,false});
    }

    @Test
    public void testDoubleColumn() {
        assertColumnInput(new Double[]{1.0,2.0});

    }

    @Test
    public void testFloatColumn() {
        assertColumnInput(new Float[]{1.0f,2.0f});

    }


    @Test
    public void testIntColumn() {
        assertColumnInput(new Integer[]{1,2});

    }

    @Test
    public void testLongColumn() {
        assertColumnInput(new Long[]{1L,2L});

    }

    @Test
    public void testStringColumn() {
        assertColumnInput(new String[]{"12","22"});

    }


    private <T>  void assertColumnInput(T[] inputData) {
        DataVecColumn<T> column = DataVecTable.createColumnOfType("test",inputData);
        assertEquals(inputData.length,column.rows());
        for(int i = 0; i < inputData.length; i++) {
            assertEquals(inputData[i],column.elementAtRow(i));
        }

        column.op("reduce_sum",new DataVecColumn[]{column},new String[]{"test"},null);

    }

}
