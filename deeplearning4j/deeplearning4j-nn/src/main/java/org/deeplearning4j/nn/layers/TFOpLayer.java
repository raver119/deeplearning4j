package org.deeplearning4j.nn.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
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
    private List<String> inputDtypes;


    public TFOpLayer(Map nodeDef, Map constants, NeuralNetConfiguration conf, DataType dtype){
        super(conf, dtype);
        this.nodeDef = nodeDef;
        setGraphRunner(nodeDef);
        INDArray[] consts = new INDArray[constants.size() + 1];
        for (int i = 0; i < consts.length - 1; i++){
            List<Number> list = (List<Number>)constants.get(String.valueOf(i + 1));
            if (list == null){
                throw new RuntimeException("Invalid constants map.");
            }
            consts[i] = Nd4j.create(list);
        }
        this.inputs = consts;
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
            Map<String, AttrValue> attrMap = nodeDef.getAttrMap();
            for (Map.Entry<String, AttrValue> e: attrMap.entrySet()){
                inputDtypes.add(e.getValue().getType().toString());
            }
            String graph = "node{\n" + nodeDef.toString() + "\n}\nversions {\n producer: 22\n}";
            for (int i = 0; i < inputNames.size(); i++){
                String inpName = inputNames.get(i);
                String dtype = inputDtypes.get(i);
                graph = "node{\nname: \"" + inpName + "\"\nop: \"Placeholder\"\nattr{\nkey: \"dtype\"\n value {\n type: " + dtype + "}\n}\n}\n" + graph;
            }
            log.info(graph);
            GraphDef.Builder graphDefBuilder = GraphDef.newBuilder();
            TextFormat.getParser().merge(graph, graphDefBuilder);
            GraphDef graphDef = graphDefBuilder.build();
            org.nd4j.shade.protobuf.ByteString serialized = graphDef.toByteString();
            byte[] binaryString = serialized.toByteArray();
            graphRunner = GraphRunner.builder().inputNames(inputNames).outputNames(outputNames).graphBytes(binaryString).build();
        }
        catch (Exception e){
            throw new RuntimeException("Error parsing protobuf", e);
        }

    }
    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr){

        // TODO resolve protobuff from nodeDef and Constants
        inputs[0] = input;
        Map<String, INDArray> inputMap = new HashMap<>();
        for (int i = 0; i < inputs.length; i++){
            inputMap.put(inputNames.get(i), inputs[i]);
        }
        INDArray out = graphRunner.run(inputMap).values().toArray(new INDArray[0])[0];
        log.debug(out.toString());
        return out;

    }


    @Override
    public boolean isPretrainLayer(){
        return false;
    }

    @Override
    public void clearNoiseWeightParams(){

    }

}
