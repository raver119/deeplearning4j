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
package org.nd4j.linalg.lossfunctions;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.common.primitives.Pair;
import java.util.HashMap;
import java.util.Map;

/**
 * SameDiff loss function.
 *
 * This class can be extended to create Deeplearning4j loss functions by defining one single method only:
 * {@link #defineLoss(SameDiff, SDVariable, SDVariable, boolean)}. This method is used to define the loss function on a
 * <i>per example</i> basis - i.e., the output should be an array with shape [minibatch].<br>
 * <br>
 * For example, the mean squared error (MSE) loss function can be defined using:<br>
 * {@code return labels.squaredDifference(layerInput).mean(1);}
 *
 */
public abstract class BaseSameDiffLoss extends BaseLossFunction {
    protected transient SameDiff sumSD;
    protected transient SameDiff averageSD;
    protected static final String LOSS_VAR_NAME = "loss";

    protected BaseSameDiffLoss() {

    }

//    /**
//     * Define the loss function.<br>
//     * <b>NOTE</b>: The score on a *per example* basis - should return a SDVariable with shape [minibatch], where out[i]
//     * is the score for the ith minibatch
//     *
//     * @param sameDiff         SameDiff instance to define the loss on
//     * @param layerInput Input to the SameDiff loss function
//     * @param labels     Labels placeholder
//     * @return The score on a per example basis (SDVariable with shape [minibatch])
//     */
//    public abstract SDVariable defineLoss(@NonNull SameDiff sameDiff, @NonNull SDVariable layerInput, @NonNull SDVariable labels);

    protected void createSameDiffInstance(DataType dataType, boolean average){
        SameDiff sd;
        if(average) {
            averageSD = SameDiff.create();
            sd = averageSD;
        } else {
            sumSD = SameDiff.create();
            sd = sumSD;
        }

        SDVariable layerInput = sd.placeHolder("layerInput", dataType, -1);
        SDVariable labels = sd.placeHolder("labels", dataType, -1);
        SDVariable loss = this.defineLoss(sd, layerInput, labels, average);
        loss.rename(LOSS_VAR_NAME);
        loss.markAsLoss();
        sd.createGradFunction("layerInput");
    }

    protected void createSameDiffInstanceIfRequired(DataType dataType, boolean average){
        if(average){
            if(averageSD == null)
                createSameDiffInstance(dataType, average);
        } else {
            if(sumSD == null)
                createSameDiffInstance(dataType, average);
        }
    }

    protected SameDiff getSameDiffInstance(boolean average){
        return average ? averageSD : sumSD;
    }

    /**
     * Compute the score (loss function value) for the given inputs.
     *
     * @param labels       Label/expected preOutput
     * @param preOutput    Output of the model (neural network)
     * @param activationFn Activation function that should be applied to preOutput
     * @param mask         Mask array; may be null
     * @param average      Whether the score should be averaged (divided by number of rows in labels/preOutput) or not   @return Loss function value
     */
    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        createSameDiffInstanceIfRequired(preOutput.dataType(), average);

        Preconditions.checkArgument((labels.size(1) == preOutput.size(1)), "Labels array numColumns (size(1) = %s) does not match output layer number of outputs (nOut = %s)", labels.size(1), preOutput.size(1));

        INDArray output = activationFn.getActivation(preOutput.dup(), true);

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("labels", labels);
        inputs.put("layerInput", output);

        INDArray score = getSameDiffInstance(average).outputSingle(inputs, LOSS_VAR_NAME);

        return score.sumNumber().doubleValue();
    }


    /**
     * Compute the score (loss function value) for each example individually.
     * For input [numExamples,nOut] returns scores as a column vector: [numExamples,1]
     *
     * @param labels       Labels/expected output
     * @param preOutput    Output of the model (neural network)
     * @param activationFn Activation function that should be applied to preOutput
     * @param mask         @return Loss function value for each example; column vector
     */
    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        throw new UnsupportedOperationException("Can't calculate per-example loss when using SameDiff loss functions");
    }


    /**
     * Compute the gradient of the loss function with respect to the inputs: dL/dOutput
     *
     * @param labels       Label/expected output
     * @param preOutput    Output of the model (neural network), before the activation function is applied
     * @param activationFn Activation function that should be applied to preOutput
     * @param mask         Mask array; may be null
     * @return Gradient dL/dPreOut
     */
    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        createSameDiffInstanceIfRequired(preOutput.dataType(), false);


        Map<String, INDArray> m = new HashMap<>();
        INDArray output = activationFn.getActivation(preOutput.dup(), true);
        m.put("labels", labels);
        m.put("layerInput", output);

        Map<String, INDArray> grads = getSameDiffInstance(false).calculateGradients(m, "layerInput");

        INDArray gradAtActivationOutput = grads.get("layerInput");
        INDArray gradAtInput = activationFn.backprop(preOutput.dup(), gradAtActivationOutput).getFirst();

        //TODO no mask application in forward pass yet
//        if (mask != null) {
//            LossUtil.applyMask(gradAtInput, mask);
//        }
        return gradAtInput;
    }

    /**
     * Compute both the score (loss function value) and gradient. This is equivalent to calling {@link #computeScore(INDArray, INDArray, IActivation, INDArray, boolean)}
     * and {@link #computeGradient(INDArray, INDArray, IActivation, INDArray)} individually
     *
     * @param labels       Label/expected output
     * @param preOutput    Output of the model (neural network)
     * @param activationFn Activation function that should be applied to preOutput
     * @param mask         Mask array; may be null
     * @param average      Whether the score should be averaged (divided by number of rows in labels/output) or not
     * @return The score (loss function value) and gradient
     */
    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn,
                                                          INDArray mask, boolean average) {

        Pair<Double, INDArray> GradientAndScore = new Pair<>();
        GradientAndScore.setFirst(this.computeScore(labels, preOutput, activationFn, mask, average));
        GradientAndScore.setSecond(this.computeGradient(labels, preOutput, activationFn, mask));

        return GradientAndScore;
    }

    @Override
    public String name() {
        return getClass().getSimpleName();
    }
}




