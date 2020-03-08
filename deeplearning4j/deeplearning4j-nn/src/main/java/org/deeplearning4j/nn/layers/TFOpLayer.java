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

package org.deeplearning4j.nn.layers;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.tensorflow.conversion.TensorDataType;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;
import com.google.gson.Gson;
import org.nd4j.shade.protobuf.Message;
import org.nd4j.shade.protobuf.TextFormat;

import java.util.*;
import java.util.List;


@Slf4j
public class TFOpLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.TFOpLayer> {


    private Map nodeDef;
    private INDArray[] inputs;
    private GraphRunner graphRunner;
    private List<String> inputNames;
    private List<TensorDataType> inputDtypes;


    public TFOpLayer(Map nodeDef, Map constants, NeuralNetConfiguration conf, DataType dtype){
        super(conf, dtype);
        this.nodeDef = nodeDef;
        setGraphRunner(nodeDef);
        INDArray[] inputs = new INDArray[constants.size() + 1];
        for (int i = 1; i < inputs.length; i++){
            List<Number> list = (List<Number>)constants.get(String.valueOf(i));
            if (list == null){
                throw new RuntimeException("Invalid constants map.");
            }
            inputs[i] = Nd4j.create(list);
        }
        this.inputs = inputs;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr){
        throw new RuntimeException("Backprop through TFOplayer is not supported yet.");
    }

    /**
     * Converts a Map representation of Nodedef to a singleton TF Graph and instantiates a GraphRunner.
     * @param nodedefMap
     */
    private void setGraphRunner(Map<String, Object> nodedefMap) {
        try{
            String json = new Gson().toJson(nodedefMap);
            NodeDef.Builder builder = NodeDef.newBuilder();
            org.nd4j.shade.protobuf.util.JsonFormat.parser().merge(json, builder);
            NodeDef nodeDef = builder.build();
            List<String> inputNames = new ArrayList<>();
            List<String> outputNames = Arrays.asList(nodeDef.getName());
            this.inputNames = inputNames;
            for (int i = 0; i < nodeDef.getInputCount(); i++){
                inputNames.add(nodeDef.getInput(i));
            }
            inputDtypes = new ArrayList<>();
            List<String> inputDataTypeNames = new ArrayList<>();
            Map<String, AttrValue> attrMap = nodeDef.getAttrMap();
            for (Map.Entry<String, AttrValue> e: attrMap.entrySet()){
                String dtName = e.getValue().getType().toString();
                inputDataTypeNames.add(dtName);
                inputDtypes.add(TensorDataType.fromProtoValue(dtName));
            }
            String graph = "node{\n" + nodeDef.toString() + "\n}\nversions {\n producer: 22\n}";
            for (int i = 0; i < inputNames.size(); i++){
                String inpName = inputNames.get(i);
                String dtype = inputDataTypeNames.get(i);
                graph = "node{\nname: \"" + inpName + "\"\nop: \"Placeholder\"\nattr{\nkey: \"dtype\"\n value {\n type: " + dtype + "}\n}\n}\n" + graph;
            }
            log.info(graph);
            GraphDef.Builder graphDefBuilder = GraphDef.newBuilder();
            TextFormat.getParser().merge(graph, graphDefBuilder);
            GraphDef graphDef = graphDefBuilder.build();
            org.nd4j.shade.protobuf.ByteString serialized = graphDef.toByteString();
            byte[] binaryString = serialized.toByteArray();
            Map<String, TensorDataType> dtypeMap = new HashMap<>();
            for (int i = 0; i < inputNames.size(); i++){
                dtypeMap.put(inputNames.get(i), inputDtypes.get(i));
            }
            graphRunner = GraphRunner.builder().inputNames(inputNames).outputNames(outputNames).graphBytes(binaryString).inputDataTypes(dtypeMap).build();
        }
        catch (Exception e){
            throw new RuntimeException("Error parsing protobuf", e);
        }

    }

    private INDArray runGraph(INDArray input){
        if (input.rank() == 3){
            // TODO make this a preprocessor
            input = input.permute(1, 0, 2);
        }
        inputs[0] = input;
        Map<String, INDArray> inputMap = new HashMap<>();
        for (int i = 0; i < inputs.length; i++){
            inputMap.put(inputNames.get(i), inputs[i]);
        }
        INDArray out = graphRunner.run(inputMap).values().toArray(new INDArray[0])[0];
        log.debug(out.toString());
        if (out.rank() == 3){
            out = out.permute(1, 0, 2); // TODO post-processing?
        }
        return out;
    }

    public long[] getOutputShape(long[] inputShape){
        long[] shape = ArrayUtils.clone(inputShape);
        for(int i = 0; i < shape.length; i++){
            if (shape[i] < 0){
                shape[i] = 1;
            }
        }
        INDArray dummyArr = Nd4j.zeros(shape);
        return runGraph(dummyArr).shape();
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr){
        return runGraph(input);
    }


    @Override
    public boolean isPretrainLayer(){
        return false;
    }

    @Override
    public void clearNoiseWeightParams(){

    }

}
