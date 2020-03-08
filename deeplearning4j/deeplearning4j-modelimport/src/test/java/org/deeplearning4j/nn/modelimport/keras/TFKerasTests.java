/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

package org.deeplearning4j.nn.modelimport.keras;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

public class TFKerasTests extends BaseDL4JTest{

    @Test
    public void testModelWithTFOp() throws Exception{
        File f = new File("C:\\Users\\fariz\\desktop\\model.h5");
       ComputationGraph graph = KerasModelImport.importKerasModelAndWeights(f.getAbsolutePath());
        INDArray out = graph.outputSingle(Nd4j.zeros(12, 3, 2));
        Assert.assertArrayEquals(new long[]{12, 3}, out.shape());
    }

}
