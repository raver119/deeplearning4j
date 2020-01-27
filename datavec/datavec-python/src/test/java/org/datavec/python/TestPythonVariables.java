/*
 *
 *  * ******************************************************************************
 *  *  * Copyright (c) 2015-2019 Skymind Inc.
 *  *  * Copyright (c) 2019 Konduit AI.
 *  *  *
 *  *  * This program and the accompanying materials are made available under the
 *  *  * terms of the Apache License, Version 2.0 which is available at
 *  *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *  *
 *  *  * Unless required by applicable law or agreed to in writing, software
 *  *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  * License for the specific language governing permissions and limitations
 *  *  * under the License.
 *  *  *
 *  *  * SPDX-License-Identifier: Apache-2.0
 *  *  *****************************************************************************
 *
 *
 */

package org.datavec.python;

import org.bytedeco.javacpp.BytePointer;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collections;

import static junit.framework.TestCase.assertNotNull;
import static junit.framework.TestCase.assertNull;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestPythonVariables {

    @Test
    public void testDataAssociations() {
        PythonVariables pythonVariables = new PythonVariables();
        PythonVariables.Type[] types = {
                PythonVariables.Type.INT,
                PythonVariables.Type.FLOAT,
                PythonVariables.Type.STR,
                PythonVariables.Type.BOOL,
                PythonVariables.Type.DICT,
                PythonVariables.Type.LIST,
                PythonVariables.Type.LIST,
                PythonVariables.Type.FILE,
                PythonVariables.Type.NDARRAY,
                PythonVariables.Type.BYTES
        };

        NumpyArray npArr = new NumpyArray(Nd4j.scalar(1.0));
        ByteBuffer bb = ByteBuffer.allocateDirect(3);
        bb.put((byte)'a');
        bb.put((byte)'b');
        bb.put((byte)'c');
        bb.rewind();
        Object[] values = {
                1L,1.0,"1",true, Collections.singletonMap("1",1),
                new Object[]{1}, Arrays.asList(1),"type", npArr,
                bb
        };

        Object[] expectedValues = {
                1L,1.0,"1",true, Collections.singletonMap("1",1),
                new Object[]{1}, new Object[]{1},"type", npArr, new BytePointer(bb)
        };

        for(int i = 0; i < types.length; i++) {
            testInsertGet(pythonVariables,types[i].name() + i,values[i],types[i],expectedValues[i]);
        }

        assertEquals(types.length,pythonVariables.getVariables().length);

    }

    private void testInsertGet(PythonVariables pythonVariables,String key,Object value,PythonVariables.Type type,Object expectedValue) {
        pythonVariables.add(key, type);
        assertNull(pythonVariables.getValue(key));
        pythonVariables.setValue(key,value);
        assertNotNull(pythonVariables.getValue(key));
        Object actualValue = pythonVariables.getValue(key);
        if (expectedValue instanceof Object[]){
            assertTrue(actualValue instanceof Object[]);
            Object[] actualArr = (Object[])actualValue;
            Object[] expectedArr = (Object[])expectedValue;
            assertArrayEquals(expectedArr, actualArr);
        }
        else{
            assertEquals(expectedValue,pythonVariables.getValue(key));
        }

    }


}
