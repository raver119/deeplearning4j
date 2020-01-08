
/* ******************************************************************************
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
package org.nd4j.linalg.api.ops.random.custom;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

@NoArgsConstructor
public class RandomMultinomial extends DynamicCustomOp {

    private DataType outputType;

    public RandomMultinomial(INDArray logits, INDArray num_samples, INDArray output) {
        if (num_samples != null)
            addInputArgument(logits, num_samples);
        else
            addInputArgument(logits);
        addOutputArgument(output);
    }

    public RandomMultinomial(SameDiff sameDiff, SDVariable logits, SDVariable num_samples) {
        super(sameDiff, new SDVariable[]{logits, num_samples});
    }

    @Override
    public String opName() {
        return "random_multinomial";
    }

    @Override
    public String tensorflowName() {
        return "Multinomial";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        if(attributesForNode.containsKey("output_dtype")) {
            outputType = TFGraphMapper.convertType(attributesForNode.get("output_dtype").getType());
        } else {
            outputType = DataType.FLOAT;
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        return Collections.singletonList(outputType);
    }
}
