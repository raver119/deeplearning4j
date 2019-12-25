/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.python;


import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

@javax.annotation.concurrent.NotThreadSafe
public class TestPythonContextManager {

    @Test
    public void testInt(){
        PythonContextManager.setContext("context1");
        FastPythonExecutioner.exec("a = 1");
        PythonContextManager.setContext("context2");
        FastPythonExecutioner.exec("a = 2");
        PythonContextManager.setContext("context3");
        FastPythonExecutioner.exec("a = 3");


        PythonContextManager.setContext("context1");
        Assert.assertEquals(1, FastPythonExecutioner.getVariable("a").toInt());

        PythonContextManager.setContext("context2");
        Assert.assertEquals(2, FastPythonExecutioner.getVariable("a").toInt());

        PythonContextManager.setContext("context3");
        Assert.assertEquals(3, FastPythonExecutioner.getVariable("a").toInt());

        PythonContextManager.deleteNonMainContexts();
    }

    @Test
    public void testNDArray(){
        PythonContextManager.setContext("context1");
        FastPythonExecutioner.exec("import numpy as np");
        FastPythonExecutioner.exec("a = np.zeros((3,2)) + 1");

        PythonContextManager.setContext("context2");
        FastPythonExecutioner.exec("import numpy as np");
        FastPythonExecutioner.exec("a = np.zeros((3,2)) + 2");

        PythonContextManager.setContext("context3");
        FastPythonExecutioner.exec("import numpy as np");
        FastPythonExecutioner.exec("a = np.zeros((3,2)) + 3");

        PythonContextManager.setContext("context1");
        FastPythonExecutioner.exec("a += 1");

        PythonContextManager.setContext("context2");
        FastPythonExecutioner.exec("a += 2");

        PythonContextManager.setContext("context3");
        FastPythonExecutioner.exec("a += 3");

        INDArray arr = Nd4j.create(DataType.DOUBLE, 3, 2);
        PythonContextManager.setContext("context1");
        Assert.assertEquals(arr.add(2), FastPythonExecutioner.getVariable("a").toNumpy().getNd4jArray());

        PythonContextManager.setContext("context2");
        Assert.assertEquals(arr.add(4), FastPythonExecutioner.getVariable("a").toNumpy().getNd4jArray());

        PythonContextManager.setContext("context3");
        Assert.assertEquals(arr.add(6), FastPythonExecutioner.getVariable("a").toNumpy().getNd4jArray());
    }

}
