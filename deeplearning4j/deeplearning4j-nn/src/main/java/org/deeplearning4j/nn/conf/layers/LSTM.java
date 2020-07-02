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

package org.deeplearning4j.nn.conf.layers;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.ToString;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional.Mode;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.layers.recurrent.LSTMHelpers;
import org.deeplearning4j.nn.params.LSTMParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationELU;
import org.nd4j.linalg.activations.impl.ActivationHardSigmoid;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationSoftPlus;
import org.nd4j.linalg.activations.impl.ActivationSoftSign;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.activations.impl.ActivationThresholdedReLU;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMActivations;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDataFormat;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDirectionMode;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMLayerConfig;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMLayerWeights;
import org.nd4j.linalg.factory.Nd4j;

/**
 * LSTM recurrent neural network layer without peephole connections. Supports CuDNN acceleration - see <a
 * href="https://deeplearning4j.konduit.ai/config/backends/config-cudnn">https://deeplearning4j.konduit.ai/config/backends/config-cudnn</a>
 * for details
 *
 * @author Alex Black
 * @see GravesLSTM GravesLSTM class for an alternative LSTM (with peephole connections)
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class LSTM extends AbstractLSTM {

    private double forgetGateBiasInit;
    private IActivation gateActivationFn = new ActivationSigmoid();

    private LSTM(Builder builder) {
        super(builder);
        this.forgetGateBiasInit = builder.forgetGateBiasInit;
        this.gateActivationFn = builder.gateActivationFn;
        initializeConstraints(builder);
    }

    @Override
    protected void initializeConstraints(org.deeplearning4j.nn.conf.layers.Layer.Builder<?> builder) {
        super.initializeConstraints(builder);
        if (((Builder) builder).recurrentConstraints != null) {
            if (constraints == null) {
                constraints = new ArrayList<>();
            }
            for (LayerConstraint c : ((Builder) builder).recurrentConstraints) {
                LayerConstraint c2 = c.clone();
                c2.setParams(Collections.singleton(LSTMParamInitializer.RECURRENT_WEIGHT_KEY));
                constraints.add(c2);
            }
        }
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
            int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
        LayerValidation.assertNInNOutSet("LSTM", getLayerName(), layerIndex, getNIn(), getNOut());
        org.deeplearning4j.nn.layers.recurrent.LSTM ret = new org.deeplearning4j.nn.layers.recurrent.LSTM(conf,
                networkDataType);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return LSTMParamInitializer.getInstance();
    }

    private static LSTMActivations toLSTMActivation(IActivation activationFn){
        if(activationFn instanceof ActivationTanH)
            return LSTMActivations.TANH;
        else if(activationFn instanceof ActivationReLU) {
            ActivationReLU relu = (ActivationReLU) activationFn;
            if(relu.getThreshold() != 0 || relu.getNegativeSlope() != 0)
                throw new UnsupportedOperationException("LSTM toSameDiff doesn't support ReLU activation with threshold and negative slope.");

            if(relu.getMax() != 0)
                throw new UnsupportedOperationException("LSTM toSameDiff doesn't support ReLU activation with max.");

            //TODO no way to pass parms to libnd4j
//            if(relu.getNegativeSlope() != 0)
//                return LSTMActivations.LEAKY_RELU;
//
//            if(relu.getThreshold() != 0)
//                return LSTMActivations.THRESHHOLD_RELU;

            return LSTMActivations.RELU;
        } else if(activationFn instanceof ActivationSigmoid)
            return LSTMActivations.SIGMOID;
        else if(activationFn instanceof ActivationLReLU)
//            return LSTMActivations.LEAKY_RELU;
            //TODO no way to pass parms to libnd4j
            throw new UnsupportedOperationException("LSTM toSameDiff doesn't support activation ActivationLReLU");
        else if(activationFn instanceof ActivationThresholdedReLU)
//            return LSTMActivations.THRESHHOLD_RELU;
            //TODO no way to pass parms to libnd4j
            throw new UnsupportedOperationException("LSTM toSameDiff doesn't support activation ActivationThresholdedReLU");
        else if(activationFn instanceof ActivationHardSigmoid)
            return LSTMActivations.HARD_SIGMOID;
        else if(activationFn instanceof ActivationELU)
            return LSTMActivations.ELU;
        else if(activationFn instanceof ActivationSoftSign)
            return LSTMActivations.SOFTSIGN;
        else if(activationFn instanceof ActivationSoftPlus)
            return LSTMActivations.SOFTPLUS;
        else
            //TODO add ActivationThresholdedReLU and ActivationLReLU to list once supported
            throw new UnsupportedOperationException("Unsupported activation for LSTM toSameDiff: " + activationFn.getClass().getSimpleName() +
                    ".  Should be one of ActivationTanH, ActivationReLU, ActivationSigmoid, "
                    + "ActivationHardSigmoid, ActivationELU, ActivationSoftSign, or ActivationSoftPlus.");
    }

    @Override
    public void transformParamsForSameDiff(@NonNull Map<String, INDArray> params) {
        INDArray bias = params.get(LSTMParamInitializer.BIAS_KEY);
        params.put(LSTMParamInitializer.BIAS_KEY, Nd4j.squeeze(bias, 0));
    }

    @Override
    public SDVariable defineLayer(@NonNull SameDiff sameDiff, @NonNull SDVariable layerInput,
            SDVariable mask, @NonNull Map<String, SDVariable> paramTable) {

        SDVariable recurrentWeight = paramTable.get(LSTMParamInitializer.RECURRENT_WEIGHT_KEY);
        SDVariable inputWeight = paramTable.get(LSTMParamInitializer.INPUT_WEIGHT_KEY);
        SDVariable bias = paramTable.get(LSTMParamInitializer.BIAS_KEY);

        LSTMActivations gateActivation = toLSTMActivation(gateActivationFn);
        LSTMActivations recurrentActivation = toLSTMActivation(activationFn);


        return sameDiff.rnn.lstmLayer(layerInput, LSTMLayerWeights.builder()
                        .weights(inputWeight)
                        .rWeights(recurrentWeight)
                        .bias(bias)
                        .build(),
                LSTMLayerConfig.builder()
                        .gateAct(gateActivation)
                        .cellAct(recurrentActivation)
                        .retFullSequence(true)
                        .directionMode(LSTMDirectionMode.FWD)
                        .lstmdataformat(rnnDataFormat == RNNFormat.NCW ? LSTMDataFormat.NST : LSTMDataFormat.NTS)
                        .build())[0];
    }

    @Override
    public SDVariable defineBidirectional(SameDiff sameDiff, SDVariable layerInput, Map<String, SDVariable> paramTable,
            SDVariable mask, Mode mode) {
        SDVariable recurrentWeight = paramTable.get(LSTMParamInitializer.RECURRENT_WEIGHT_KEY);
        SDVariable inputWeight = paramTable.get(LSTMParamInitializer.INPUT_WEIGHT_KEY);
        SDVariable bias = paramTable.get(LSTMParamInitializer.BIAS_KEY);

        LSTMActivations gateActivation = toLSTMActivation(gateActivationFn);
        LSTMActivations recurrentActivation = toLSTMActivation(activationFn);

        LSTMDirectionMode directionMode;
        if(mode == Mode.ADD || mode == Mode.AVERAGE)
            directionMode = LSTMDirectionMode.BIDIR_SUM;
        else if(mode == Mode.CONCAT)
            directionMode = LSTMDirectionMode.BIDIR_CONCAT;
        else
            throw new UnsupportedOperationException("Bidirectional not supported for mode " + mode);

        LSTMDataFormat format = rnnDataFormat == RNNFormat.NCW ? LSTMDataFormat.NST : LSTMDataFormat.NTS;

        SDVariable output = sameDiff.rnn.lstmLayer(layerInput, LSTMLayerWeights.builder()
                        .weights(inputWeight)
                        .rWeights(recurrentWeight)
                        .bias(bias)
                        .build(),
                LSTMLayerConfig.builder()
                        .gateAct(gateActivation)
                        .cellAct(recurrentActivation)
                        .directionMode(directionMode)
                        .lstmdataformat(format)
                        .build())[0];

        if(mode == Mode.AVERAGE)
            return output.div(2);
        else
            return output;

    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        //TODO - CuDNN etc
        return LSTMHelpers.getMemoryReport(this, inputType);
    }

    @NoArgsConstructor
    public static class Builder extends AbstractLSTM.Builder<Builder> {


        @SuppressWarnings("unchecked")
        public LSTM build() {
            return new LSTM(this);
        }
    }

}
