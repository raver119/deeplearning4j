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

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.ToString;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.Collection;
import java.util.Map;

/**
 * A version of {@link OutputLayer} for recurrent neural networks. Expects inputs of size [minibatch,nIn,sequenceLength]
 * and labels of shape [minibatch,nOut,sequenceLength]. It also supports mask arrays.
 * <br>
 * Note that RnnOutputLayer can also be used for 1D CNN layers, which also have [minibatch,nOut,sequenceLength]
 * activations/labels shape.
 *
 * See also: {@link RnnLossLayer}
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class RnnOutputLayer extends BaseOutputLayer {

    private RNNFormat rnnDataFormat = RNNFormat.NCW;
    private RnnOutputLayer(Builder builder) {
        super(builder);
        initializeConstraints(builder);
        this.rnnDataFormat = builder.rnnDataFormat;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
        LayerValidation.assertNInNOutSet("RnnOutputLayer", getLayerName(), layerIndex, getNIn(), getNOut());

        org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer ret =
                        new org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer(conf, networkDataType);
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
        return DefaultParamInitializer.getInstance();
    }

    @Override
    public SDVariable defineLayer(@NonNull SameDiff sameDiff, @NonNull SDVariable layerInput,
            @NonNull Map<String, SDVariable> paramTable, SDVariable mask) {
        SDVariable b = paramTable.get(DefaultParamInitializer.BIAS_KEY);
        SDVariable  W = paramTable.get(DefaultParamInitializer.WEIGHT_KEY);

        if(rnnDataFormat == RNNFormat.NWC)
            layerInput = layerInput.permute(0, 2, 1);

        SDVariable batch = sameDiff.sizeAt(layerInput, 0);
        SDVariable sequenceLength;

        if(rnnDataFormat == RNNFormat.NCW) {
            sequenceLength = sameDiff.sizeAt(layerInput, 2);
            layerInput = layerInput.permute(0, 2, 1);
        } else if(rnnDataFormat == RNNFormat.NWC)
            sequenceLength = sameDiff.sizeAt(layerInput, 1);
        else
            throw new UnsupportedOperationException("Unknown RNN data format " + rnnDataFormat);

        SDVariable distributedShape = sameDiff.concat(0, batch.mul(sequenceLength), sameDiff.constant(-1).castTo(batch.dataType()));
        SDVariable distributedInput = layerInput.reshape(distributedShape);

        SDVariable distributedOutput = distributedInput.mmul(W);
        if(hasBias)
            distributedOutput = distributedOutput.add(b);

        distributedOutput = doActivation(distributedOutput);

        SDVariable temp = distributedOutput.reshape(sameDiff.concat(0, batch, sequenceLength, sameDiff.constant(-1).castTo(batch.dataType())));

        if(rnnDataFormat == RNNFormat.NCW)
            return temp.permute(0, 2, 1);
        else
            return temp;
    }

    @Override
    public SDVariable defineLoss(@NonNull SameDiff sameDiff, @NonNull SDVariable input, SDVariable labels,
            boolean average) {
        SDVariable batch = sameDiff.sizeAt(input, 0);
        SDVariable channels;

        if(rnnDataFormat == RNNFormat.NCW){
            channels = sameDiff.sizeAt(input, 1);
            input = input.permute(0, 2, 1);
            labels = labels.permute(0, 2, 1);
        } else if(rnnDataFormat == RNNFormat.NWC){
            channels = sameDiff.sizeAt(input, 2);
        }  else
            throw new UnsupportedOperationException("Unknown CNN data format " + rnnDataFormat);

        SDVariable newShape = sameDiff.concat(0, sameDiff.constant(-1).castTo(batch.dataType()), channels);
        return lossFn.defineLoss(sameDiff, input.reshape(newShape), labels.reshape(newShape), average);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input type for RnnOutputLayer (layer index = " + layerIndex
                            + ", layer name=\"" + getLayerName() + "\"): Expected RNN input, got " + inputType);
        }
        InputType.InputTypeRecurrent itr = (InputType.InputTypeRecurrent) inputType;

        return InputType.recurrent(nOut, itr.getTimeSeriesLength(), itr.getFormat());
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input type for RnnOutputLayer (layer name=\"" + getLayerName()
                            + "\"): Expected RNN input, got " + inputType);
        }

        InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent) inputType;
        this.rnnDataFormat = r.getFormat();
        if (nIn <= 0 || override) {
            this.nIn = r.getSize();
        }
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, rnnDataFormat, getLayerName());
    }


    public static class Builder extends BaseOutputLayer.Builder<Builder> {

        private RNNFormat rnnDataFormat = RNNFormat.NCW;
        public Builder() {
            //Set default activation function to softmax (to match default loss function MCXENT)
            this.setActivationFn(new ActivationSoftmax());
        }

        /**
         * @param lossFunction Loss function for the output layer
         */
        public Builder(LossFunction lossFunction) {
            lossFunction(lossFunction);
            //Set default activation function to softmax (for consistent behaviour with no-arg constructor)
            this.setActivationFn(new ActivationSoftmax());
        }

        /**
         * @param lossFunction Loss function for the output layer
         */
        public Builder(ILossFunction lossFunction) {
            this.setLossFn(lossFunction);
            //Set default activation function to softmax (for consistent behaviour with no-arg constructor)
            this.setActivationFn(new ActivationSoftmax());
        }

        @Override
        @SuppressWarnings("unchecked")
        public RnnOutputLayer build() {
            return new RnnOutputLayer(this);
        }

        /**
         * @param rnnDataFormat Data format expected by the layer. NCW = [miniBatchSize, size, timeSeriesLength],
         * NWC = [miniBatchSize, timeSeriesLength, size]. Defaults to NCW.
         */
        public Builder dataFormat(RNNFormat rnnDataFormat){
            this.rnnDataFormat = rnnDataFormat;
            return this;
        }
    }
}
