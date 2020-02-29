package org.deeplearning4j.nn.conf.layers;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.AbstractSameDiffLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.regularization.Regularization;

import java.util.Collection;
import java.util.List;
import java.util.Map;


public class TFOpLayer extends Layer{

    private Map nodeDef;
    private Map constants;
    public TFOpLayer(Map nodeDef, Map constants){
        super();
        this.nodeDef = nodeDef;
        this.constants = constants;
    }

    @Override
    public ParamInitializer initializer() {
        return new ParamInitializer() {
            @Override
            public long numParams(NeuralNetConfiguration conf) {
                return 0;
            }

            @Override
            public long numParams(Layer layer) {
                return 0;
            }

            @Override
            public List<String> paramKeys(Layer layer) {
                return null;
            }

            @Override
            public List<String> weightKeys(Layer layer) {
                return null;
            }

            @Override
            public List<String> biasKeys(Layer layer) {
                return null;
            }

            @Override
            public boolean isWeightParam(Layer layer, String key) {
                return false;
            }

            @Override
            public boolean isBiasParam(Layer layer, String key) {
                return false;
            }

            @Override
            public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
                return null;
            }

            @Override
            public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
                return null;
            }
        };
    }
    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        //Default implementation: no-op
        return null;
    }

    @Override
    public boolean isPretrainParam(String param){
        return false;
    }

    @Override
    public InputType getOutputType(int idx, InputType inputType){
        InputType.Type type = inputType.getType();
        long[] shape = inputType.getShape();
        if (type == InputType.Type.RNN){
           long t = shape[0];
           shape[0] = shape[1];
           shape[1] = t;
        }
        org.deeplearning4j.nn.layers.TFOpLayer tempLayer = new org.deeplearning4j.nn.layers.TFOpLayer(nodeDef, constants, null, null);
        long[] outputShape = tempLayer.getOutputShape(shape);
        if (outputShape.length == 3){
            long t = outputShape[1];
            outputShape[1] = outputShape[0];
            outputShape[0] = t;
        }
        System.out.println(outputShape);
        return InputType.inferInputType(Nd4j.create(outputShape));

    }

    @Override
    public  void setNIn(InputType inputType, boolean override){}


    @Override
    public GradientNormalization getGradientNormalization(){return null;}


    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                                Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
                                                                boolean initializeParams, DataType networkDataType) {

        org.deeplearning4j.nn.layers.TFOpLayer tfOpLayer = new org.deeplearning4j.nn.layers.TFOpLayer(nodeDef, constants, conf, networkDataType);
        tfOpLayer.setListeners(trainingListeners);
        tfOpLayer.setIndex(layerIndex);
        return tfOpLayer;
    }

    @Override
    public double getGradientNormalizationThreshold(){return 0.;}

    @Override
    public List<Regularization> getRegularizationByParam(String paramName){return null;}

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return new LayerMemoryReport(); //TODO
    }





}
