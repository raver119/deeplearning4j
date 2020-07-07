/*
 * ******************************************************************************
 *  * Copyright (c) 2020 Konduit K.K.
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.samediff;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.reflect.ClassPath;
import com.google.common.reflect.ClassPath.ClassInfo;
import java.io.IOException;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.dropout.IDropout;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Cnn3DLossLayer;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.Convolution1D;
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.LayerWithLoss;
import org.deeplearning4j.nn.conf.layers.Pooling1D;
import org.deeplearning4j.nn.conf.layers.Pooling2D;
import org.deeplearning4j.nn.conf.layers.RnnLossLayer;
import org.deeplearning4j.nn.conf.layers.Subsampling1DLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.impl.LayerVertex;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.layers.LossLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ToSameDiffUtils;
import org.junit.runner.Result;
import org.junit.runner.notification.RunListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;


@Slf4j
public class ToSameDiffTests extends RunListener {

    public static boolean SKIP_UNIMPLEMENTED = true;
    public static boolean FAIL_FAST = true;
    public static boolean FAIL_IF_MISSING = false;
    // makes it show up in IDEA test runs
    public static boolean PRINT_AFTER_EVERY = true;

    private static final Set<String> failurePointLayers = new HashSet<>();
    private static final Set<String> failurePointVertices = new HashSet<>();
    private static final Set<String> failureLosses = new HashSet<>();

    private static void cleanupLayers(Set<Class<? extends Layer>> layers){
        if(layers.remove(Convolution1D.class))
            layers.add(Convolution1DLayer.class);

        if(layers.remove(Convolution2D.class))
            layers.add(ConvolutionLayer.class);

        if(layers.remove(Pooling1D.class))
            layers.add(Subsampling1DLayer.class);

        if(layers.remove(Pooling2D.class))
            layers.add(SubsamplingLayer.class);
    }

    private static void cleanupLosses(Set<Class<? extends ILossFunction>> layers){

    }

    private static void cleanupDropouts(Set<Class<? extends IDropout>> layers){

    }

    private static void cleanupActivations(Set<Class<? extends IActivation>> layers){

    }

    private static void cleanupPreprocessors(Set<Class<? extends InputPreProcessor>> layers){

    }

    private static void cleanupVertices(Set<Class<? extends GraphVertex>> layers){

    }

    private static <T> Set<Class<? extends T>> findClasses(Class<T> superClass, String topPackage) {

        Set<ClassInfo> infos;
        try {
            infos = ClassPath.from(superClass.getClassLoader()).getTopLevelClassesRecursive(topPackage);
        } catch (IOException e) {
            infos = new HashSet<>();
        }

        Set<Class<? extends T>> classes = new HashSet<>();
        for(ClassInfo ci : infos){
            Class<?> c = ci.load();
            if(superClass.isAssignableFrom(c) && 
                    !Modifier.isAbstract(c.getModifiers()) && 
                    !c.isInterface() && 
                    !c.getSimpleName().toLowerCase().contains("custom"))
                classes.add(c.asSubclass(superClass));
        }
        return classes;
    }
    
    private static Set<Class<? extends Layer>> findLayers() {
        Set<Class<? extends Layer>> ret = findClasses(Layer.class, "org.deeplearning4j.nn.conf.layers");
        cleanupLayers(ret);
        return ret;
    }

    private static Set<Class<? extends ILossFunction>> findLosses() {
        Set<Class<? extends ILossFunction>> ret = findClasses(ILossFunction.class, "org.nd4j.linalg.lossfunctions");
        cleanupLosses(ret);
        return ret;
    }

    private static Set<Class<? extends IDropout>> findDropouts() {
        Set<Class<? extends IDropout>> ret = findClasses(IDropout.class, "org.deeplearning4j.nn.conf.dropout");
        cleanupDropouts(ret);
        return ret;
    }

    private static Set<Class<? extends IActivation>> findActivations() {
        Set<Class<? extends IActivation>> ret = findClasses(IActivation.class, "org.nd4j.linalg.activations");
        cleanupActivations(ret);
        return ret;
    }

    private static Set<Class<? extends InputPreProcessor>> findPreprocessors() {
        Set<Class<? extends InputPreProcessor>> ret = findClasses(InputPreProcessor.class, "org.deeplearning4j.nn.conf.preprocessor");
        cleanupPreprocessors(ret);
        return ret;
    }

    private static Set<Class<? extends GraphVertex>> findVertices() {
        Set<Class<? extends GraphVertex>> ret = findClasses(GraphVertex.class, "org.deeplearning4j.nn.graph.vertex.impl");
        cleanupVertices(ret);
        return ret;
    }

    private enum Stage{
        Conversion, Output, Loss;

        public Set<Class<? extends Layer>> testedLayers = new HashSet<>();
        public Set<Class<? extends ILossFunction>> testedLosses = new HashSet<>();
        public Set<Class<? extends IDropout>> testedDropouts = new HashSet<>();
        public Set<Class<? extends IActivation>> testedActivations = new HashSet<>();
        public Set<Class<? extends InputPreProcessor>> testedPreprocessors = new HashSet<>();
        public Set<Class<? extends GraphVertex>> testedVertices = new HashSet<>();

        public void cleanup(){
            cleanupLayers(testedLayers);
            cleanupLosses(testedLosses);
            cleanupDropouts(testedDropouts);
            cleanupActivations(testedActivations);
            cleanupPreprocessors(testedPreprocessors);
            cleanupVertices(testedVertices);
        }
        
        private static <T> Set<String> minusStr(Set<Class<? extends T>> a, Set<Class<? extends T>> b){
            Set<String> ret = new HashSet<>();
            for(Class<? extends T> c : a){
                if(!b.contains(c))
                    ret.add(c.getSimpleName());
            }
            return ret;
        }

        public int check(Set<Class<? extends Layer>> foundLayers,
                Set<Class<? extends ILossFunction>> foundLosses,
                Set<Class<? extends IDropout>> foundDropouts,
                Set<Class<? extends IActivation>> foundActivations,
                Set<Class<? extends InputPreProcessor>> foundPreprocessors,
                Set<Class<? extends GraphVertex>> foundVertices
                ){

            if(this == Stage.Loss){
                // only care about losses & output/loss layers here
                foundDropouts.clear();
                foundActivations.clear();
                foundPreprocessors.clear();
                foundVertices.clear();

                Set<Class<? extends Layer>> old = foundLayers;
                foundLayers = new HashSet<>();
                for(Class<? extends Layer> layer : old){
                    if(LayerWithLoss.class.isAssignableFrom(layer))
                        foundLayers.add(layer);
                }
            }

            Set<String> missingLayers = minusStr(foundLayers, testedLayers);
            Set<String> missingLosses = minusStr(foundLosses, testedLosses);
            Set<String> missingDropouts = minusStr(foundDropouts, testedDropouts);
            Set<String> missingActivations = minusStr(foundActivations, testedActivations);
            Set<String> missingPreprocessors = minusStr(foundPreprocessors, testedPreprocessors);
            Set<String> missingVertices = minusStr(foundVertices, testedVertices);

            if(this != Stage.Loss)
                log.info(" --- ToSameDiff {} Tests --- ", name());
            else
                log.info(" --- ToSameDiff Loss Tests (only layers that define losses and loss functions are shown) --- ");

            log.info("Missing Layers: {}", missingLayers);

            if(this != Stage.Loss) {
                log.info("Missing Activations: {}", missingActivations);
            }

            log.info("Missing Losses: {}", missingLosses);

            if(this != Stage.Loss) {
                log.info("Missing Preprocessors: {}", missingPreprocessors);
                log.info("Missing Dropouts: {}", missingDropouts);
                log.info("Missing Vertices: {}", missingVertices);
            }

            return missingLayers.size() + missingLosses.size() + missingDropouts.size() +
                    missingActivations.size() + missingPreprocessors.size() + missingVertices.size();
        }
        
        public void record(InputPreProcessor preProcessor){
            if(preProcessor != null) {
                testedPreprocessors.add(preProcessor.getClass());
            }
        }

        public void record(IActivation activation){
            if(activation != null){
                testedActivations.add(activation.getClass());
            }
        }

        public void record(ILossFunction lossFunction){
            if(lossFunction != null){
                testedLosses.add(lossFunction.getClass());
            }
        }

        public void record(IDropout dropout){
            if(dropout != null){
                testedDropouts.add(dropout.getClass());
            }
        }

        public void record(Layer layer){
            if(layer == null)
                return;

            testedLayers.add(layer.getClass());
            record(layer.getIDropout());

            if(layer instanceof BaseWrapperLayer){
                record(((BaseWrapperLayer) layer).getUnderlying());
            } else if(layer instanceof FrozenLayer) {
                record(((FrozenLayer) layer).getLayer());
            } else if(layer instanceof Bidirectional){
                record(((Bidirectional) layer).getFwd());
                record(((Bidirectional) layer).getBwd());
            }

            if(layer instanceof FeedForwardLayer){
                record(((FeedForwardLayer) layer).getActivationFn());
            }

            if(layer instanceof org.deeplearning4j.nn.conf.layers.BaseOutputLayer){
                record(((org.deeplearning4j.nn.conf.layers.BaseOutputLayer) layer).getLossFn());
            } else if(layer instanceof CnnLossLayer){
                record(((CnnLossLayer) layer).getLossFn());
            } else if(layer instanceof RnnLossLayer){
                record(((RnnLossLayer) layer).getLossFn());
            } else if(layer instanceof org.deeplearning4j.nn.conf.layers.LossLayer){
                record(((org.deeplearning4j.nn.conf.layers.LossLayer) layer).getLossFn());
            } else if(layer instanceof Cnn3DLossLayer){
                record(((Cnn3DLossLayer) layer).getLossFn());
            }
        }

        public void record(GraphVertex vertex){
            if(vertex == null)
                return;

            testedVertices.add(vertex.getClass());
            if(vertex.hasLayer()){
                record(vertex.getLayer().conf().getLayer());

            }

            if(vertex instanceof LayerVertex)
                record(((LayerVertex) vertex).getLayerPreProcessor());

        }
    }

    public static void testToSameDiff(@NonNull MultiLayerNetwork network, INDArray input, INDArray labels){
        testToSameDiff(network, null, input, labels);
    }

    private static ILossFunction getLossFn(org.deeplearning4j.nn.api.Layer layer){
        ILossFunction lossFn = null;
        if(layer instanceof BaseOutputLayer){
            lossFn = ((BaseOutputLayer<?>) layer).getLossFn();
        } else if(layer instanceof LossLayer){
            lossFn = ((LossLayer) layer).getLossFn();
        } else if(layer instanceof org.deeplearning4j.nn.layers.convolution.Cnn3DLossLayer){
            lossFn = ((org.deeplearning4j.nn.layers.convolution.Cnn3DLossLayer) layer).getLossFn();
        } else if(layer instanceof org.deeplearning4j.nn.layers.convolution.CnnLossLayer){
            lossFn = ((org.deeplearning4j.nn.layers.convolution.CnnLossLayer) layer).getLossFn();
        } else if(layer instanceof org.deeplearning4j.nn.layers.recurrent.RnnLossLayer){
            lossFn = ((org.deeplearning4j.nn.layers.recurrent.RnnLossLayer) layer).getLossFn();
        }
        return lossFn;
    }

    public static void testToSameDiff(@NonNull MultiLayerNetwork network, InputType inputType, INDArray input, INDArray labels){

        for(int i = 0 ; i < network.getnLayers() ; i++){
            Layer layer = network.getLayer(i).conf().getLayer();
            Stage.Conversion.record(layer);
            Stage.Conversion.record(network.getLayerWiseConfigurations().getInputPreProcess(i));
        }

        SameDiff sameDiff;
        try{
            sameDiff = network.toSameDiff(inputType, true, true);
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

        for(int i = 0 ; i < network.getnLayers() ; i++){
            Layer layer = network.getLayer(i).conf().getLayer();
            Stage.Output.record(layer);
            Stage.Output.record(network.getLayerWiseConfigurations().getInputPreProcess(i));
        }

        List<INDArray> activations = network.feedForward(input);
        activations.remove(0);

        List<String> sdActivationVariables = new ArrayList<>();


        List<String> namesByLayer = ToSameDiffUtils.getScopeNames(network.getLayers());

        List<String> layerClassNames = new ArrayList<>();
        for(int i = 0 ; i < network.getnLayers() ; i++){
            org.deeplearning4j.nn.conf.layers.Layer config = network.getLayerWiseConfigurations().getConf(i).getLayer();

            String scope = namesByLayer.get(i);
            List<SDVariable> scopeVars = sameDiff.getVariablesInScope(scope);
            layerClassNames.add(config.getClass().getSimpleName());
            if(scopeVars.size() > 0) {

                SDVariable lastVar = null;
                for(int j = scopeVars.size() - 1 ; j >= 0 ; j--){
                    SDVariable variable = scopeVars.get(j);

                    if(!variable.name().contains("/loss/") && !variable.name().endsWith("loss") && !variable.name().endsWith("labels")){
                        lastVar = variable;
                        break;
                    }

                }

                if(lastVar != null)
                    sdActivationVariables.add(lastVar.name());
                else
                    sdActivationVariables.add(sdActivationVariables.get(sdActivationVariables.size() - 1));
            } else
                sdActivationVariables.add(sdActivationVariables.get(sdActivationVariables.size() - 1));
        }

        Map<String, INDArray> sdActivations = sameDiff.batchOutput()
                .output(sdActivationVariables.toArray(new String[0]))
                .input("input", input)
                .output();


        assertEquals("Sizes of DL4J activations and found SameDiff activations differ", activations.size(), sdActivationVariables.size());

        List<Pair<String, String>> messages = new ArrayList<>();
        boolean failed = false;
        for(int i = 0 ; i < sdActivationVariables.size() ; i++){
            INDArray sd = sdActivations.get(sdActivationVariables.get(i));
            INDArray dl4j = activations.get(i);

            if(! sd.equalsWithEps(dl4j, 1e-3)) {

                if(!failed)
                    failurePointLayers.add(network.getLayer(i).conf().getLayer().getClass().getSimpleName());

                failed = true;
                if(FAIL_FAST)
                    fail("DL4J activation and SameDiff activation not equal for Layer " + layerClassNames.get(i) +  " and SDVariable " + sdActivationVariables.get(i));
                else
                    messages.add(new Pair<>(layerClassNames.get(i), sdActivationVariables.get(i)));
            }
        }

        StringBuilder message = new StringBuilder("DL4J activation and SameDiff activation not equal for ");

        for(Pair<String, String> pair : messages)
            message.append("Layer ").append(pair.getFirst()).append(" and SDVariable ").append(pair.getSecond())
                    .append(", ");

        assertEquals(message.toString(), 0, messages.size());

        if(labels != null){

            for(int i = 0 ; i < network.getnLayers() ; i++){
                Layer layer = network.getLayer(i).conf().getLayer();
                Stage.Loss.record(layer);
                Stage.Loss.record(network.getLayerWiseConfigurations().getInputPreProcess(i));
            }

            INDArray output = network.output(input).dup();
            network.setLabels(labels);
            network.computeGradientAndScore();
            double score = network.score() - network.calcRegularizationScore(true);

            Map<String, INDArray> sdOutputs = sameDiff.batchOutput()
                    .output(sameDiff.outputs().get(0), sameDiff.getLossVariables().get(0))
                    .input("input", input)
                    .input("labels", labels)
                    .output();

            INDArray sdLoss = sdOutputs.get(sameDiff.getLossVariables().get(0));
            INDArray sdOutput = sdOutputs.get(sameDiff.outputs().get(0));


            assertTrue("Outputs don't match for original network and SameDiff version", sdOutput.equalsWithEps(output, 1e-3));

            double sdScore = sdLoss.sumNumber().doubleValue();

            ILossFunction lossFn = getLossFn(network.getOutputLayer());
            try {
                assertEquals("Losses don't match for original network and SameDiff version" + (lossFn != null ?
                                " for loss function " + lossFn.getClass().getSimpleName() : ""),
                        sdScore, score, 1e-3);
            } catch (AssertionError ae){
                if(ae.getMessage().contains("Losses don't match") && lossFn != null){
                    failureLosses.add(lossFn.getClass().getSimpleName());
                }
                throw ae;
            }
        }

        if(PRINT_AFTER_EVERY) {
            printResults();
        }

    }

    public static void testToSameDiff(@NonNull ComputationGraph graph, @NonNull INDArray inputs, INDArray labels){
        INDArray[] labelsArray = null;
        if(labels != null)
            labelsArray = new INDArray[]{labels};

        testToSameDiff(graph, new INDArray[]{inputs}, labelsArray);
    }

    public static void testToSameDiff(@NonNull ComputationGraph graph, @NonNull INDArray[] inputs, INDArray[] labels){
        testToSameDiff(graph, inputs, labels, null);
    }

    public static void testToSameDiff(@NonNull ComputationGraph graph, @NonNull INDArray[] inputs, INDArray[] labels, InputType[] inputTypes){
        Preconditions.checkArgument(inputs.length == graph.getConfiguration().getNetworkInputs().size(),
                "Didn't supply the right number of inputs: expected %s, got %s", graph.getConfiguration().getNetworkInputs().size(), inputs.length);

        Map<String, InputType> inputTypesMap = new HashMap<>();
        Map<String, INDArray> inputsMap = new HashMap<>();

        for(int i = 0 ; i < inputs.length ; i++){
            String name = graph.getConfiguration().getNetworkInputs().get(i);
            inputsMap.put(name, inputs[i]);

            if(inputTypes != null && inputTypes.length > i && inputTypes[i] != null)
                inputTypesMap.put(name, inputTypes[i]);
            else
                inputTypesMap.put(name, InputType.inferInputType(inputs[i]));
        }

        InputType[] inputVertTypes = new InputType[inputTypesMap.size()];
        int j = 0;
        for(String inputName : graph.getConfiguration().getNetworkInputs()){
            inputVertTypes[j] = inputTypesMap.get(inputName);
            j++;
        }

        try {
            graph.getConfiguration().getLayerActivationTypes(true, inputVertTypes);
        } catch (Exception e){
            log.warn("Error getting activation types and adding preprocessors for graph", e);
        }

        for(GraphVertex vertex : graph.getVertices()){
            Stage.Conversion.record(vertex);
        }

        SameDiff sameDiff;
        try{
            sameDiff = graph.toSameDiff(inputTypesMap, true, true);
        } catch (UnsupportedOperationException e){
            if(!SKIP_UNIMPLEMENTED)
                throw e;
            else
                return;
        } catch (IllegalStateException e){
            if((e.getMessage().contains(" convert to SameDiff with different regularizations") ||
                    e.getMessage().contains(" convert to SameDiff with different IUpdaters") ||
                    e.getMessage().equals("Dimension must be set for toSameDiff conversion.")) &&
                    SKIP_UNIMPLEMENTED)
                return;
            else
                throw e;
        }

        for(GraphVertex vertex : graph.getVertices()){
            Stage.Output.record(vertex);
        }

        Map<String, INDArray> activations = graph.feedForward(inputs, false);

        for(String inputName : inputsMap.keySet())
            activations.remove(inputName);

        List<String> activationKeys = new ArrayList<>();
        for(String n : graph.getConfiguration().getTopologicalOrderStr()){
            if(activations.containsKey(n))
                activationKeys.add(n);
        }

        Map<String, SDVariable> sdActivationVariables = new HashMap<>();
        for(String vertexName : activationKeys){
            List<SDVariable> scopeVars = sameDiff.getVariablesInScope(vertexName);
            if(!scopeVars.isEmpty()){
                SDVariable lastVar = null;
                for(int i = scopeVars.size() - 1 ; i >= 0 ; i--){
                    SDVariable variable = scopeVars.get(i);

                    if(!variable.name().contains("/loss/") && !variable.name().endsWith("loss") && !variable.name().endsWith("labels")){
                        lastVar = variable;
                        break;
                    }

                }

                if(lastVar != null)
                    sdActivationVariables.put(vertexName, lastVar);
                else {
                    List<String> vertexInputs = graph.getConfiguration().getVertexInputs().get(vertexName);
                    if(vertexInputs.size() == 1){
                        sdActivationVariables.put(vertexName, sdActivationVariables.get(vertexInputs.get(0)));
                    }
                }
            }
        }

        Map<String, INDArray> sdActivations = sameDiff.batchOutput()
                .inputs(inputsMap)
                .output(sdActivationVariables.values().toArray(new SDVariable[0]))
                .output();

        assertEquals("Sizes of DL4J activations and found SameDiff activations differ", activations.size(), sdActivationVariables.size());


        List<Pair<String, String>> messages = new ArrayList<>();
        boolean failed = false;
        for(String vertexName : activations.keySet()){
            INDArray dl4j = activations.get(vertexName);
            INDArray sd = sdActivations.get(sdActivationVariables.get(vertexName).name());

            if(! sd.equalsWithEps(dl4j, 1e-3)) {
                GraphVertex vertex = graph.getVertex(vertexName);

                if(!failed){
                    if(vertex instanceof LayerVertex)
                        failurePointLayers.add(vertex.getLayer().conf().getLayer().getClass().getSimpleName());
                    else
                        failurePointVertices.add(vertex.getClass().getSimpleName());
                }

                failed = true;

                String vertexStr = vertexName + "[" + vertex.getClass().getSimpleName();

                if(vertex.hasLayer())
                    vertexStr += "(" + vertex.getLayer().conf().getLayer().getClass().getSimpleName() + ")";

                vertexStr += "]";

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

            for(GraphVertex vertex : graph.getVertices()){
                Stage.Loss.record(vertex);
            }

            List<String> labelNames = sameDiff.getTrainingConfig().getDataSetLabelMapping();
            Map<String, INDArray> inputAndLabelMap = new HashMap<>(inputsMap);
            Preconditions.checkArgument(labels.length == labelNames.size(),
                    "Didn't supply the right number of labels: expected %s, got %s", labelNames.size(), labels.length);

            for (int i = 0; i < labels.length; i++) {
                inputAndLabelMap.put(labelNames.get(i), labels[i]);
            }

            graph.setLabels(labels);
            graph.computeGradientAndScore();
            double score = graph.score() - graph.calcRegularizationScore(true);

            Map<String, INDArray> sdLosses = sameDiff.batchOutput()
                    .inputs(inputAndLabelMap)
                    .output(sameDiff.getLossVariables().toArray(new String[0]))
                    .output();

            double sdScore = 0;
            for(INDArray scoreArr : sdLosses.values())
                sdScore += scoreArr.sumNumber().doubleValue();

            Set<String> lossFunctions = new HashSet<>();
            for(String name : graph.getConfiguration().getNetworkOutputs()){
                GraphVertex vertex = graph.getVertex(name);
                if(vertex.hasLayer()){
                    ILossFunction lossFn = getLossFn(vertex.getLayer());
                    if(lossFn != null)
                        lossFunctions.add(lossFn.getClass().getSimpleName());
                }
            }

            try {
                assertEquals("Losses don't match for original network and SameDiff version, with loss functions " + lossFunctions,
                        sdScore, score, 1e-3);
            } catch (AssertionError ae){
                if(ae.getMessage().contains("Losses don't match") && !lossFunctions.isEmpty()){
                    failureLosses.addAll(lossFunctions);
                }
                throw ae;
            }
        }

        if(PRINT_AFTER_EVERY) {
            printResults();
        }
    }

    private static final Set<Class<? extends Layer>> foundLayers = findLayers();
    private static final Set<Class<? extends ILossFunction>> foundLosses = findLosses();
    private static final Set<Class<? extends IDropout>> foundDropouts = findDropouts();
    private static final Set<Class<? extends IActivation>> foundActivations = findActivations();
    private static final Set<Class<? extends InputPreProcessor>> foundPreprocessors = findPreprocessors();
    private static final Set<Class<? extends GraphVertex>> foundVertices = findVertices();

    public static int printResults() {
        int conversion = Stage.Conversion.check(foundLayers, foundLosses, foundDropouts, foundActivations, foundPreprocessors, foundVertices);
        int output = Stage.Output.check(foundLayers, foundLosses, foundDropouts, foundActivations, foundPreprocessors, foundVertices);
        int loss = Stage.Loss.check(foundLayers, foundLosses, foundDropouts, foundActivations, foundPreprocessors, foundVertices);

        if(!(failurePointVertices.isEmpty() && failureLosses.isEmpty() && failurePointLayers.isEmpty())){
            log.info(" --- ToSameDiff Failure Points --- ");
        }

        if(!failurePointLayers.isEmpty()){
            log.info("Failure point layers: {}", failurePointLayers);
        }

        if(!failurePointVertices.isEmpty()){
            log.info("Failure point vertices: {}", failurePointVertices);
        }

        if(!failureLosses.isEmpty()){
            log.info("Failed losses: {}", failureLosses);
        }

        return conversion + output + loss;
    }

    @Override
    public void testRunFinished(Result result) throws Exception {
        int failCount = printResults();

        if(FAIL_IF_MISSING){
            assertEquals("There were missing ToSameDiff tests", 0, failCount);
        } else if(failCount > 0){
            log.warn("There were {} missing ToSameDiff tests", failCount);
        }
    }
}
