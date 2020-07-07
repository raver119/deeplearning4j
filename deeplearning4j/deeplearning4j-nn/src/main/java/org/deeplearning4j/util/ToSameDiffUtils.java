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

package org.deeplearning4j.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Trainable;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.updater.BaseMultiLayerUpdater;
import org.deeplearning4j.nn.updater.UpdaterBlock;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.learning.regularization.Regularization;

/**
 * Utilities for use in {@link org.deeplearning4j.nn.graph.ComputationGraph#toSameDiff(SameDiff, Map, boolean, boolean)} and {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork#toSameDiff(SameDiff, InputType, boolean, boolean)}.
 */
@Slf4j
public class ToSameDiffUtils {


    /**
     * Get the updater for a network.  If updaters aren't the same on all layers, throws an exception or returns null depending on skipErrors.
     * @param layers The layers of the network.
     * @param skipErrors If true, returns null if updaters aren't the same for all layers.  Otherwise, throws an error.
     */
    public static IUpdater getUpdater(Layer[] layers, boolean skipErrors){
        IUpdater iUpdater = null;
        for(Layer l : layers) {
            org.deeplearning4j.nn.conf.layers.Layer conf = l.conf().getLayer();
            if (conf instanceof BaseLayer) {
                IUpdater u = ((BaseLayer) conf).getIUpdater();
                if (iUpdater == null) {
                    iUpdater = u;
                } else {
                    if (u != null && !u.equals(iUpdater)) {
                        if (skipErrors) {
                            log.warn("Ignoring updater config: Can not convert to SameDiff with different IUpdaters. Expected {}, but was {} for {}", iUpdater, u, conf);
                            return null;
                        } else {
                            throw new IllegalStateException(
                                    "Can not convert to SameDiff with different IUpdaters.  Ensure all layers have the same updater.  Expected "
                                            + iUpdater + ", but was " + u + " different for " + conf);
                        }
                    }
                }

                u = ((BaseLayer) conf).getBiasUpdater();
                if (iUpdater == null) {
                    iUpdater = u;
                } else {
                    if (u != null && !u.equals(iUpdater)) {
                        if (skipErrors) {
                            log.warn("Ignoring updater config: Can not convert to SameDiff when layers have different IUpdaters. Expected {}, but was {} for {}", iUpdater, u, conf);
                            return null;
                        } else {
                            throw new IllegalStateException(
                                    "Can not convert to SameDiff with different IUpdaters.  Ensure all layers have the same updater.  Expected "
                                            + iUpdater + ", but was " + u + " for " + conf);
                        }
                    }
                }
            }
        }
        return iUpdater;
    }

    /**
     * Get the regularizations of a network.  If regularizations aren't the same on all layers, throws an exception or returns null depending on skipErrors.
     * @param layers The layers of the network.
     * @param skipErrors If true, returns null if regularizations aren't the same for all layers.  Otherwise, throws an error.
     */
    public static List<Regularization> getRegularizations(Layer[] layers, boolean skipErrors){
        List<Regularization> regularizations = null;

        for(Layer l : layers){
            org.deeplearning4j.nn.conf.layers.Layer conf = l.conf().getLayer();
            if(conf instanceof BaseLayer){
                if(regularizations == null){
                    regularizations = ((BaseLayer) conf).getRegularization();
                } else {
                    if(!((BaseLayer) conf).getRegularization().equals(regularizations)) {
                        if(skipErrors){
                            log.warn("Ignoring regularization config: Can not convert to SameDiff when layers have different regularizations. Expected {}, but was {} for {}",
                                    regularizations, ((BaseLayer) conf).getRegularization(), conf);
                            return null;
                        } else {
                            throw new IllegalStateException(
                                    "Can not convert to SameDiff with different regularizations.  Ensure all layers have the same regularizations, and that bias and weight regularizations are the same.  "
                                            + "Expected " + regularizations + ", but was " + ((BaseLayer) conf)
                                            .getRegularization() + " for " + conf);
                        }
                    }
                }

                if(regularizations == null){
                    regularizations = ((BaseLayer) conf).getRegularizationBias();
                } else {
                    if(!((BaseLayer) conf).getRegularizationBias().equals(regularizations)) {
                        if(skipErrors){
                            log.warn("Ignoring regularization config: Can not convert to SameDiff when layers have different regularizations. Expected {}, but was {} for {}",
                                    regularizations, ((BaseLayer) conf).getRegularization(), conf);
                            return null;
                        } else {
                            throw new IllegalStateException(
                                    "Can not convert to SameDiff with different regularizations.  Ensure all layers have the same regularizations, and that bias and weight regularizations are the same.  "
                                            + "Expected " + regularizations + ", but was " + ((BaseLayer) conf)
                                            .getRegularizationBias() + " for bias in " + conf);
                        }
                    }
                }
            }
        }
        return regularizations;
    }

    /**
     * Define the parameters of a layer, transforming them if necessary using {@link org.deeplearning4j.nn.conf.layers.Layer#transformParamsForSameDiff(Map)}.
     *
     * @param sameDiff The SameDiff to define the parameters in.
     * @param layer The layer whose parameters we are defining.
     * @param useView Whether to use the param view directly (if true) or dup it.
     * @return The SDVariable parameters of the layer.
     */
    public static Map<String, SDVariable> defineParams(SameDiff sameDiff, Layer layer, boolean useView){
        Map<String, INDArray> params = new HashMap<>(layer.paramTable(false));
        layer.conf().getLayer().transformParamsForSameDiff(params);
        return defineTransformedParams(sameDiff, params, (int) layer.numParams(), useView);
    }


    /**
     * Define the parameters of a vertex, transforming them if necessary using {@link GraphVertex#transformParamsForSameDiff(Map)}.
     *
     * @param sameDiff The SameDiff to define the parameters in.
     * @param vertex The vertex whose parameters we are defining.
     * @param useView Whether to use the param view directly (if true) or dup it.
     * @return The SDVariable parameters of the vertex.
     */
    public static Map<String, SDVariable> defineParams(SameDiff sameDiff, GraphVertex vertex, boolean useView){
        Map<String, INDArray> params = new HashMap<>(vertex.paramTable(false));
        vertex.transformParamsForSameDiff(params);
        return defineTransformedParams(sameDiff, params, (int) vertex.numParams(), useView);
    }

    /**
     * A helper for parameter definition.
     */
    private static Map<String, SDVariable> defineTransformedParams(SameDiff sameDiff, Map<String, INDArray> params, int numParams, boolean useView){
        Map<String, SDVariable> newParams = new HashMap<>(numParams);
        for (Map.Entry<String, INDArray> entry : params.entrySet()) {
            INDArray value = entry.getValue();
            if (!useView) {
                value = value.dup();
            }
            newParams.put(entry.getKey(), sameDiff.var(entry.getKey(), value));
        }
        return newParams;
    }

    public static List<String> getScopeNames(Layer[] layers){
        List<String> names = new ArrayList<>();
        Map<String, Integer> numLayers = new HashMap<>();

        for (Layer layer : layers) {
            org.deeplearning4j.nn.conf.layers.Layer config = layer.conf().getLayer();
            String baseName = config.getLayerName() == null ? config.getClass().getSimpleName() : config.getLayerName();

            int layerNum = 0;

            if (numLayers.containsKey(baseName)) {
                layerNum = numLayers.get(baseName);
                numLayers.put(baseName, ++layerNum);
            } else {
                numLayers.put(baseName, 0);
            }
            names.add(baseName + (layerNum == 0 ? "" : "_" + layerNum));
        }

        return names;
    }

    /**
     * Copy the state from a MultiLayerNetwork or ComputationGraph updater to a SameDiff instance.
     * @param sameDiff The SameDiff to copy to.
     * @param updater The updater to copy from.
     * @param layers The layers of the network or graph.
     */
    public static void copyUpdaterState(@NonNull SameDiff sameDiff, BaseMultiLayerUpdater<?> updater, Layer[] layers){
        if(updater == null)
            return;

        List<Layer> layerList = null;
        List<String> layerNames = null;
        if(layers != null) {
            layerNames = getScopeNames(layers);
            layerList = Arrays.asList(layers);
        }

        Map<String, Map<String, INDArray>> stateViewsPerParam = new HashMap<>();
        for(UpdaterBlock ub : updater.getUpdaterBlocks()){
            List<UpdaterBlock.ParamState> params = ub.getLayersAndVariablesInBlock();
            int blockPStart = ub.getParamOffsetStart();
            int blockPEnd = ub.getParamOffsetEnd();

            int blockUStart = ub.getUpdaterViewOffsetStart();
            int blockUEnd = ub.getUpdaterViewOffsetEnd();

            int paramsMultiplier = (blockUEnd-blockUStart)/(blockPEnd-blockPStart);     //Updater state length should be exactly 0, 1, 2 or 3x number of params

            INDArray updaterView = ub.getUpdaterView();
            long nParamsInBlock = blockPEnd - blockPStart;

            long soFar = 0;
            for( int sub=0; sub<paramsMultiplier; sub++) {
                //subsetUpdaterView: [m0, m1, m2] etc
                INDArray subsetUpdaterView = updaterView.get(
                        NDArrayIndex.interval(0, 0, true), NDArrayIndex.interval(soFar, soFar + nParamsInBlock));

                Map<String, INDArray> state = ub.getGradientUpdater().getState();

                long offsetWithinSub = 0;
                for (UpdaterBlock.ParamState ps : params) {

                    String namespace;
                    if(ps.getLayer() instanceof GraphVertex){
                        namespace = ((GraphVertex) ps.getLayer()).getVertexName();
                    } else {
                        Layer layer = (Layer) ps.getLayer();
                        namespace = layerNames.get(layerList.indexOf(layer));
                    }

                    String paramName = namespace + "/" + ps.getParamName();
//                    int idx = getId(ps.getLayer());
//                    String paramName = idx + "_" + ps.getParamName();

                    INDArray pv = ps.getParamView();
                    long nParamsThisParam = pv.length();



//                    INDArray currSplit = subsetUpdaterView.get(NDArrayIndex.interval(0, 0, true), NDArrayIndex.interval(offsetWithinSub, offsetWithinSub + nParamsThisParam));
//                    if(!stateViewsPerParam.containsKey(paramName))
//                        stateViewsPerParam.put(paramName, new ArrayList<INDArray>());
//                    stateViewsPerParam.get(paramName).add(currSplit);
//                    offsetWithinSub += nParamsThisParam;
                        stateViewsPerParam.put(paramName, state);
                }

                soFar += nParamsInBlock;
            }
        }

        if (sameDiff.getTrainingConfig() == null) {
            throw new ND4JIllegalStateException("Please specify a training config with setTrainingConfig");
        }

        Map<String, GradientUpdater<?>> updaterMap = new HashMap<>();
        for (Variable v : sameDiff.getVariables().values()) {
            if (v.getVariable().getVariableType() != VariableType.VARIABLE || !v.getVariable().dataType().isFPType()) {
                //Skip non-trainable parameters
                continue;
            }

            INDArray arr = v.getVariable().getArr();
            long stateSize = sameDiff.getTrainingConfig().getUpdater().stateSize(arr.length());

            Map<String, INDArray> params;
            if(stateSize > 0) {
                if (stateViewsPerParam.containsKey(v.getVariable().name())) {
                    params = stateViewsPerParam.get(v.getVariable().name());
                } else {
                    throw new IllegalStateException("No updater state found for variable " + v.getVariable().name());
                }
            } else {
                params = new HashMap<>();
            }

            GradientUpdater gu = sameDiff.getTrainingConfig().getUpdater().instantiate(params, false);
            gu.setState(params, false);
//            gu.setStateViewArray(params, arr.shape(), arr.ordering(), false);
            updaterMap.put(v.getName(), gu);
        }

        sameDiff.initializeTraining(updaterMap);

    }

}
