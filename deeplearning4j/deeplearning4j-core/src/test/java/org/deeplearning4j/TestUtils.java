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

package org.deeplearning4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import lombok.NonNull;
import org.apache.commons.compress.utils.IOUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.samediff.AbstractSameDiffLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.layers.LossLayer;
import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;
import org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer;
import org.deeplearning4j.nn.layers.normalization.BatchNormalization;
import org.deeplearning4j.nn.layers.normalization.LocalResponseNormalization;
import org.deeplearning4j.nn.layers.recurrent.LSTM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.regularization.L1Regularization;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.learning.regularization.WeightDecay;
import org.nd4j.linalg.lossfunctions.ILossFunction;

public class TestUtils {

    public static MultiLayerNetwork testModelSerialization(MultiLayerNetwork net){

        MultiLayerNetwork restored;
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ModelSerializer.writeModel(net, baos, true);
            byte[] bytes = baos.toByteArray();

            ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
            restored = ModelSerializer.restoreMultiLayerNetwork(bais, true);

            assertEquals(net.getLayerWiseConfigurations(), restored.getLayerWiseConfigurations());
            assertEquals(net.params(), restored.params());
        } catch (IOException e){
            //Should never happen
            throw new RuntimeException(e);
        }

        //Also check the MultiLayerConfiguration is serializable (required by Spark etc)
        MultiLayerConfiguration conf = net.getLayerWiseConfigurations();
        serializeDeserializeJava(conf);

        return restored;
    }

    public static ComputationGraph testModelSerialization(ComputationGraph net){
        ComputationGraph restored;
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ModelSerializer.writeModel(net, baos, true);
            byte[] bytes = baos.toByteArray();

            ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
            restored = ModelSerializer.restoreComputationGraph(bais, true);

            assertEquals(net.getConfiguration(), restored.getConfiguration());
            assertEquals(net.params(), restored.params());
        } catch (IOException e){
            //Should never happen
            throw new RuntimeException(e);
        }

        //Also check the ComputationGraphConfiguration is serializable (required by Spark etc)
        ComputationGraphConfiguration conf = net.getConfiguration();
        serializeDeserializeJava(conf);

        return restored;
    }

    public static boolean SKIP_UNIMPLEMENTED = true;
    public static boolean FAIL_FAST = true;

    private static Set<String> failures = new HashSet<>();

    public static void testToSameDiff(@NonNull MultiLayerNetwork network, INDArray input, INDArray labels){

        SameDiff sameDiff;
        try{
            sameDiff = network.toSameDiff(null, true);
        } catch (UnsupportedOperationException e){
            if(!SKIP_UNIMPLEMENTED)
                throw e;
            else
                return;
        } catch (IllegalStateException e){
            if((e.getMessage().contains(" convert to SameDiff with different regularizations") ||
                    e.getMessage().contains(" convert to SameDiff with different IUpdaters")) && SKIP_UNIMPLEMENTED)
                return;
            else
                throw e;
        }

        if(input == null){
            long[] inputShape = sameDiff.getVariable("input").placeholderShape();
            for(int i = 0 ; i < inputShape.length ; i++){
                if(inputShape[i] == -1)
                    inputShape[i] = 1;
            }

            input = Nd4j.rand(inputShape);
        }

        List<INDArray> activations = network.feedForward(input);
        activations.remove(0);

        List<String> sdActivationVariables = new ArrayList<>();

        Map<String, Integer> numLayers = new HashMap<>();

        List<String> layerNames = new ArrayList<>();
        for(int i = 0 ; i < network.getnLayers() ; i++){
            org.deeplearning4j.nn.conf.layers.Layer config = network.getLayerWiseConfigurations().getConf(i).getLayer();
            String baseName = config.getLayerName() == null ? config.getClass().getSimpleName() : config.getLayerName();

            int layerNum = 0;

            if (numLayers.containsKey(baseName)) {
                layerNum = numLayers.get(baseName);
                numLayers.put(baseName, ++layerNum);
            } else {
                numLayers.put(baseName, 0);
            }

            String scope = baseName + (layerNum == 0 ? "" : "_" + layerNum);
            List<SDVariable> scopeVars = sameDiff.getVariablesInScope(scope);
            layerNames.add(config.getClass().getSimpleName());
            if(scopeVars.size() > 0)
                sdActivationVariables.add(scopeVars.get(scopeVars.size() - 1).name());
            else
                sdActivationVariables.add(sdActivationVariables.get(sdActivationVariables.size() - 1));
        }

        Map<String, INDArray> sdActivations = sameDiff.batchOutput()
                .output(sdActivationVariables.toArray(new String[0]))
                .input("input", input)
                .output();


        //TODO remove
        System.out.println("Failures to date: " + failures);

        assertEquals("Sizes of DL4J activations and found SameDiff activations differ", activations.size(), sdActivationVariables.size());

        List<Pair<String, String>> messages = new ArrayList<>();
        for(int i = 0 ; i < sdActivationVariables.size() ; i++){
            INDArray sd = sdActivations.get(sdActivationVariables.get(i));
            INDArray dl4j = activations.get(i);

            if(! sd.equalsWithEps(dl4j, 1e-3)) {
                failures.add(layerNames.get(i));
                if(FAIL_FAST)
                    fail("DL4J activation and SameDiff activation not equal for Layer " + layerNames.get(i) +  " and SDVariable " + sdActivationVariables.get(i));
                else
                    messages.add(new Pair<>(layerNames.get(i), sdActivationVariables.get(i)));
            }
        }

        StringBuilder message = new StringBuilder("DL4J activation and SameDiff activation not equal for ");

        for(Pair<String, String> pair : messages)
            message.append("Layer ").append(pair.getFirst()).append(" and SDVariable ").append(pair.getSecond())
                    .append(", ");

        assertEquals(message.toString(), 0, messages.size());

        if(labels != null){

            INDArray output = network.output(input).dup();
            network.setLabels(labels);
            network.computeGradientAndScore();
            double score = network.score();

            Map<String, INDArray> sdOutputs = sameDiff.batchOutput()
                    .output(sameDiff.outputs().get(0), sameDiff.getLossVariables().get(0))
                    .input("input", input)
                    .input("labels", labels)
                    .output();

            INDArray sdLoss = sdOutputs.get(sameDiff.getLossVariables().get(0));
            INDArray sdOutput = sdOutputs.get(sameDiff.outputs().get(0));
            double sdScore = sdLoss.sumNumber().doubleValue();

            ILossFunction lossFn = null;
            Layer lastLayer = network.getLayer(network.getnLayers() - 1);
            if(lastLayer instanceof LossLayer){
                lossFn = ((LossLayer) lastLayer).layerConf().getLossFn();
            } else if(lastLayer instanceof BaseOutputLayer){
                lossFn = ((BaseOutputLayer<?>) lastLayer).layerConf().getLossFn();
            } else if(lastLayer instanceof org.deeplearning4j.nn.layers.convolution.Cnn3DLossLayer){
                lossFn = ((org.deeplearning4j.nn.layers.convolution.Cnn3DLossLayer) lastLayer).layerConf().getLossFn();
            }

            assertTrue("Outputs don't match for original network and SameDiff version", sdOutput.equalsWithEps(output, 1e-3));
            assertEquals("Losses don't match for original network and SameDiff version" + (lossFn != null ? " for loss function " + lossFn.getClass().getSimpleName() : ""),
                    sdScore, score, 1e-3);
        }

    }

    public static void testToSameDiff(@NonNull ComputationGraph graph, @NonNull INDArray inputs, INDArray labels){
        INDArray[] labelsArray = null;
        if(labels != null)
            labelsArray = new INDArray[]{labels};
        
        testToSameDiff(graph, new INDArray[]{inputs}, labelsArray);
    }
    
    public static void testToSameDiff(@NonNull ComputationGraph graph, @NonNull INDArray[] inputs, INDArray[] labels){
        Preconditions.checkArgument(inputs.length == graph.getConfiguration().getNetworkInputs().size(),
                "Didn't supply the right number of inputs: expected " + graph.getConfiguration().getNetworkInputs().size() + ", got " + inputs.length);

        Map<String, InputType> inputTypes = new HashMap<>();
        Map<String, INDArray> inputsMap = new HashMap<>();

        for(int i = 0 ; i < inputs.length ; i++){
            String name = graph.getConfiguration().getNetworkInputs().get(i);
            inputsMap.put(name, inputs[i]);
            inputTypes.put(name, InputType.inferInputType(inputs[i]));
        }

        SameDiff sameDiff;
        try{
            sameDiff = graph.toSameDiff(inputTypes, true);
        } catch (UnsupportedOperationException e){
            if(!SKIP_UNIMPLEMENTED)
                throw e;
            else
                return;
        } catch (IllegalStateException e){
            if((e.getMessage().contains(" convert to SameDiff with different regularizations") ||
                    e.getMessage().contains(" convert to SameDiff with different IUpdaters")) && SKIP_UNIMPLEMENTED)
                return;
            else
                throw e;
        }

        Map<String, INDArray> activations = graph.feedForward(inputs, false);

        for(String inputName : inputsMap.keySet())
            activations.remove(inputName);

        Map<String, SDVariable> sdActivationVariables = new HashMap<>();

        for(String vertexName : new ArrayList<>(activations.keySet())){
            List<SDVariable> scopeVars = sameDiff.getVariablesInScope(vertexName);
            if(!scopeVars.isEmpty()){
                sdActivationVariables.put(vertexName, scopeVars.get(scopeVars.size() - 1));
            }
        }

        Map<String, INDArray> sdActivations = sameDiff.batchOutput()
                .inputs(inputsMap)
                .output(sdActivationVariables.values().toArray(new SDVariable[0]))
                .output();

        System.out.println("Failures to date: " + failures);

        assertEquals("Sizes of DL4J activations and found SameDiff activations differ", activations.size(), sdActivationVariables.size());


        List<Pair<String, String>> messages = new ArrayList<>();
        for(String vertexName : activations.keySet()){
            INDArray dl4j = activations.get(vertexName);
            INDArray sd = sdActivations.get(sdActivationVariables.get(vertexName).name());

            if(! sd.equalsWithEps(dl4j, 1e-3)) {
                GraphVertex vertex = graph.getVertex(vertexName);
                String vertexStr = vertexName + "[" + vertex.getClass().getSimpleName();

                if(vertex.hasLayer())
                    vertexStr += "(" + vertex.getLayer().conf().getLayer().getClass().getSimpleName() + ")";

                vertexStr += "]";


                failures.add(vertexStr);
                if(FAIL_FAST)
                    fail("DL4J activation and SameDiff activation not equal for Vertex " + vertexStr +  " and SDVariable " + sdActivationVariables.get(vertexName).name());
                else
                    messages.add(new Pair<>(vertexStr, sdActivationVariables.get(vertexName).name()));
            }

        }

        StringBuilder message = new StringBuilder("DL4J activation and SameDiff activation not equal for ");

        for(Pair<String, String> pair : messages)
            message.append("Layer ").append(pair.getFirst()).append(" and SDVariable ").append(pair.getSecond())
                    .append(", ");

        assertEquals(message.toString(), 0, messages.size());

        if(sameDiff.getTrainingConfig() != null && labels != null) {
            List<String> labelNames = sameDiff.getTrainingConfig().getDataSetLabelMapping();
            Map<String, INDArray> inputAndLabelMap = new HashMap<>(inputsMap);
            Preconditions.checkArgument(labels.length == labelNames.size(),
                    "Didn't supply the right number of labels: expected " + labelNames.size() + ", got "
                            + labels.length);

            for (int i = 0; i < labels.length; i++) {
                inputAndLabelMap.put(labelNames.get(i), labels[i]);
            }

            graph.computeGradientAndScore();
            double score = graph.score();

            Map<String, INDArray> sdLosses = sameDiff.batchOutput()
                    .inputs(inputAndLabelMap)
                    .output(sameDiff.getLossVariables().toArray(new String[0]))
                    .output();

            double sdScore = 0;
            for(INDArray scoreArr : sdLosses.values())
                sdScore += scoreArr.sumNumber().doubleValue();

            assertEquals("Losses don't match for original network and SameDiff version",
                    sdScore, score, 1e-3);
        }
    }

    private static <T> T serializeDeserializeJava(T object){
        byte[] bytes;
        try(ByteArrayOutputStream baos = new ByteArrayOutputStream(); ObjectOutputStream oos = new ObjectOutputStream(baos)){
            oos.writeObject(object);
            oos.close();
            bytes = baos.toByteArray();
        } catch (IOException e){
            //Should never happen
            throw new RuntimeException(e);
        }

        T out;
        try(ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(bytes))){
            out = (T)ois.readObject();
        } catch (IOException | ClassNotFoundException e){
            throw new RuntimeException(e);
        }

        assertEquals(object, out);
        return out;
    }

    public static INDArray randomOneHot(long examples, long nOut){
        return randomOneHot(examples, nOut, new Random(12345));
    }

    public static INDArray randomOneHot(DataType dataType, long examples, long nOut){
        return randomOneHot(dataType, examples, nOut, new Random(12345));
    }

    public static INDArray randomOneHot(long examples, long nOut, long rngSeed){
        return randomOneHot(examples, nOut, new Random(rngSeed));
    }

    public static INDArray randomOneHot(long examples, long nOut, Random rng) {
        return randomOneHot(Nd4j.defaultFloatingPointType(), examples,nOut, rng);
    }

    public static INDArray randomOneHot(DataType dataType, long examples, long nOut, Random rng){
        INDArray arr = Nd4j.create(dataType, examples, nOut);
        for( int i=0; i<examples; i++ ){
            arr.putScalar(i, rng.nextInt((int) nOut), 1.0);
        }
        return arr;
    }

    public static INDArray randomOneHotTimeSeries(int minibatch, int outSize, int tsLength){
        return randomOneHotTimeSeries(minibatch, outSize, tsLength, new Random());
    }

    public static INDArray randomOneHotTimeSeries(int minibatch, int outSize, int tsLength, long rngSeed){
        return randomOneHotTimeSeries(minibatch, outSize, tsLength, new Random(rngSeed));
    }

    public static INDArray randomOneHotTimeSeries(int minibatch, int outSize, int tsLength, Random rng) {
        return randomOneHotTimeSeries(RNNFormat.NCW, minibatch, outSize, tsLength, rng);
    }

    public static INDArray randomOneHotTimeSeries(RNNFormat format, int minibatch, int outSize, int tsLength, Random rng){
        boolean ncw = format == RNNFormat.NCW;
        long[] shape = ncw ? new long[]{minibatch, outSize, tsLength} : new long[]{minibatch, tsLength, outSize};
        char order = ncw ? 'f' : 'c';
        INDArray out = Nd4j.create(DataType.FLOAT, shape, order);
        for( int i=0; i<minibatch; i++ ){
            for( int j=0; j<tsLength; j++ ){
                if(ncw){
                    out.putScalar(i, rng.nextInt(outSize), j, 1.0);
                } else {
                    out.putScalar(i, j, rng.nextInt(outSize), 1.0);
                }
            }
        }
        return out;
    }

    public static INDArray randomBernoulli(int... shape) {
        return randomBernoulli(0.5, shape);
    }

    public static INDArray randomBernoulli(double p, int... shape){
        INDArray ret = Nd4j.createUninitialized(shape);
        Nd4j.getExecutioner().exec(new BernoulliDistribution(ret, p));
        return ret;
    }

    public static void writeStreamToFile(File out, InputStream is) throws IOException {
        byte[] b = IOUtils.toByteArray(is);
        try (OutputStream os = new BufferedOutputStream(new FileOutputStream(out))) {
            os.write(b);
        }
    }

    public static L1Regularization getL1Reg(List<Regularization> l){
        for(Regularization r : l){
            if(r instanceof L1Regularization){
                return (L1Regularization) r;
            }
        }
        return null;
    }

    public static L2Regularization getL2Reg(BaseLayer baseLayer){
        return getL2Reg(baseLayer.getRegularization());
    }

    public static L2Regularization getL2Reg(List<Regularization> l){
        for(Regularization r : l){
            if(r instanceof L2Regularization){
                return (L2Regularization) r;
            }
        }
        return null;
    }

    public static WeightDecay getWeightDecayReg(BaseLayer bl){
        return getWeightDecayReg(bl.getRegularization());
    }

    public static WeightDecay getWeightDecayReg(List<Regularization> l){
        for(Regularization r : l){
            if(r instanceof WeightDecay){
                return (WeightDecay) r;
            }
        }
        return null;
    }

    public static double getL1(BaseLayer layer) {
        List<Regularization> l = layer.getRegularization();
        return getL1(l);
    }

    public static double getL1(List<Regularization> l){
        L1Regularization l1Reg = null;
        for(Regularization reg : l){
            if(reg instanceof L1Regularization)
                l1Reg = (L1Regularization) reg;
        }
        assertNotNull(l1Reg);
        return l1Reg.getL1().valueAt(0,0);
    }

    public static double getL2(BaseLayer layer) {
        List<Regularization> l = layer.getRegularization();
        return getL2(l);
    }

    public static double getL2(List<Regularization> l){
        L2Regularization l2Reg = null;
        for(Regularization reg : l){
            if(reg instanceof L2Regularization)
                l2Reg = (L2Regularization) reg;
        }
        assertNotNull(l2Reg);
        return l2Reg.getL2().valueAt(0,0);
    }

    public static double getL1(AbstractSameDiffLayer layer){
        return getL1(layer.getRegularization());
    }

    public static double getL2(AbstractSameDiffLayer layer){
        return getL2(layer.getRegularization());
    }

    public static double getWeightDecay(BaseLayer layer) {
        return getWeightDecayReg(layer.getRegularization()).getCoeff().valueAt(0,0);
    }

    public static void removeHelper(Layer layer) throws Exception {
        removeHelpers(new Layer[]{layer});
    }

    public static void removeHelpers(Layer[] layers) throws Exception {
        for(Layer l : layers){

            if(l instanceof ConvolutionLayer){
                Field f1 = ConvolutionLayer.class.getDeclaredField("helper");
                f1.setAccessible(true);
                f1.set(l, null);
            } else if(l instanceof SubsamplingLayer){
                Field f2 = SubsamplingLayer.class.getDeclaredField("helper");
                f2.setAccessible(true);
                f2.set(l, null);
            } else if(l instanceof BatchNormalization) {
                Field f3 = BatchNormalization.class.getDeclaredField("helper");
                f3.setAccessible(true);
                f3.set(l, null);
            } else if(l instanceof LSTM){
                Field f4 = LSTM.class.getDeclaredField("helper");
                f4.setAccessible(true);
                f4.set(l, null);
            } else if(l instanceof LocalResponseNormalization){
                Field f5 = LocalResponseNormalization.class.getDeclaredField("helper");
                f5.setAccessible(true);
                f5.set(l, null);
            }


            if(l.getHelper() != null){
                throw new IllegalStateException("Did not remove helper for layer: " + l.getClass().getSimpleName());
            }
        }
    }

    public static void assertHelperPresent(Layer layer){

    }

    public static void assertHelpersPresent(Layer[] layers) throws Exception {
        for(Layer l : layers){
            //Don't use instanceof here - there are sub conv subclasses
            if(l.getClass() == ConvolutionLayer.class || l instanceof SubsamplingLayer || l instanceof BatchNormalization || l instanceof LSTM){
                Preconditions.checkNotNull(l.getHelper(), l.conf().getLayer().getLayerName());
            }
        }
    }

    public static void assertHelpersAbsent(Layer[] layers) throws Exception {
        for(Layer l : layers){
            Preconditions.checkState(l.getHelper() == null, l.conf().getLayer().getLayerName());
        }
    }
}
