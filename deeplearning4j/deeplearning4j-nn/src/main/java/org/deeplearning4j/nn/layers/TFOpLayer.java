package org.deeplearning4j.nn.layers;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TFOpLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.TFOpLayer> {


    private Map nodeDef;
    private Map constants;

    public TFOpLayer(Map nodeDef, Map constants, NeuralNetConfiguration conf, DataType dtype){
        super(conf, dtype);
        this.nodeDef = nodeDef;
        this.constants = constants;
        System.out.println(nodeDef);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr){
        throw new RuntimeException("Backprop through TFOplayer is not supported yet.");
    }

    private static String pbtxtFromMap(Map<String, Object> map){
        String ret = "";
        ret += "node{\n";
        for (Map.Entry<String, Object> e: map.entrySet()){
            if (e.getValue() instanceof String){
                ret += e.getKey() + ": " + e.getValue() + '\n';
            }
            else if (e.getValue() instanceof List){
                List listVal = (List)e.getValue();
                for(Object item:listVal){
                    Map subMap = new HashMap();
                    subMap.put(e.getKey(), item);
                    //TODO
                }
            }
        }
    }
    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr){

        // TODO resolve protobuff from nodeDef and Constants
        return input.reshape(-1, 6);
    }


    @Override
    public boolean isPretrainLayer(){
        return false;
    }

    @Override
    public void clearNoiseWeightParams(){

    }

}
