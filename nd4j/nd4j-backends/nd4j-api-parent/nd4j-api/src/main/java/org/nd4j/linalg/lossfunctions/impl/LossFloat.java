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

package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.primitives.Pair;

/**
 * Implementation of "Do We Need Zero Training Loss After Achieving Zero Training Error?"
 * https://arxiv.org/abs/2002.08709
 *
 * Wraps any other Loss Function that is optimized towards zero and adds a "float value" to it.
 *
 * Intuitively this means that your model will not be able to get below a specific loss value. If the loss on an example
 * gets below this float value, the gradient for this example gets inverted, i.e. instead of descending on this gradient
 * we will ascent on it instead.
 *
 * The effect of using this Loss function is regularization: It makes it harder for the model to over-fit on the
 * training data. The float level is a hyper parameter, and selecting a good one, can take a few tries.
 *
 * A float value of 0 is the same as not using this loss function at all, everything above it is a valid value, but you
 * should likely stay below 0.3. Overall, a good starting point is to use the loss score of your best performing model
 * when using early stopping.
 *
 * @author Paul Dubs
 */
@Getter
@Setter
@EqualsAndHashCode
public class LossFloat implements ILossFunction {

    private ILossFunction wrapped;
    private double floatLevel;

    // For (De-)Serialization
    private LossFloat(){}

    public LossFloat(ILossFunction wrapped, double floatLevel) {
        this.wrapped = wrapped;
        this.floatLevel = floatLevel;
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
                               boolean average) {
        INDArray scoreArr = computeScoreArray(labels, preOutput, activationFn, mask);

        double score = scoreArr.sumNumber().doubleValue();

        if (average)
            score /= scoreArr.size(0);

        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        return Nd4j.math.abs(wrapped.computeScoreArray(labels, preOutput, activationFn, mask).subi(floatLevel)).addi(floatLevel);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray gradient = wrapped.computeGradient(labels, preOutput, activationFn, mask);
        return gradient.muli(Nd4j.math.sign(wrapped.computeScoreArray(labels, preOutput, activationFn, mask).sub(floatLevel)));
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels,
                                                          INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        return new Pair<>(computeScore(labels, preOutput, activationFn, mask, average),
            computeGradient(labels, preOutput, activationFn, mask));
    }

    @Override
    public String name() {
        return toString();
    }

    @Override
    public String toString() {
        return "LossFloatWrapper("+wrapped.name()+", "+floatLevel+")";
    }
}
