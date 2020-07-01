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

package org.nd4j.linalg.lossfunctions;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.ops.SDLoss;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * This class can be extended in two ways: <br>
 *     <ul>
 *         <li>Implement {@code defineLoss}.  It will be called to define the loss function.  You must handle averaging yourself.</li>
 *         <li>Implement {@code defineLossArray}.  It will be called from the default implementation of {@code defineLoss}, which automatically handles averaging.
 *         The use of {@link SDLoss} ops or any ops that don't accept output gradients is <b>FORBIDDEN</b>.
 *         If you want to use those ops you must use the other extension method.</li>
 *     </ul>
 */
public abstract class BaseLossFunction implements ILossFunction {

    /**
     * Helper function to sum or average a loss array depending on the parameter. <br>
     *
     * Should not be used from {@link #defineLossArray(SameDiff, SDVariable, SDVariable)} baring special circumstances,
     * as the averaging is handled automatically (using this function).
     *
     * @param output The loss array.
     * @param labels The labels, used to find the batch size.
     * @param average Whether to average the array (sums if false).
     * @return The scalar average or sum, depending on the parameter.
     */
    protected static SDVariable reduceLossArray(SDVariable output, SDVariable labels, boolean average){
//        SameDiff sameDiff = output.getSameDiff();
//        SDVariable batchSize = sameDiff.sizeAt(labels, 0);
//        SDVariable newShape = sameDiff.concat(0, batchSize, sameDiff.constant(Nd4j.scalar(batchSize.dataType(), -1)));
        output = output.sum();
        if(average)
            return output.div(output.getSameDiff().sizeAt(labels, 0));
        else
            return output;
    }

    /**
     * Helper function to apply a weight to the loss. <br>
     * Should only be used from {@link #defineLossArray(SameDiff, SDVariable, SDVariable)} baring special circumstances.
     *
     * @param loss The loss array.
     * @param weight The weight.
     */
    protected static SDVariable multiplyWeight(@NonNull SDVariable loss, INDArray weight){
        if(weight == null){
            return loss;
        } else {
            return loss.mul(loss.getSameDiff().constant(weight));
        }
    }

    /**
     * Define the loss function for a {@link SameDiff} instance by defining a per-example score array, which is averaged automatically if necessary. <br>
     *
     * The default implementation of {@link #defineLoss(SameDiff, SDVariable, SDVariable, boolean)} will call this method,
     * so a subclass of {@link BaseLossFunction} can define only this method. <br>
     *
     * However, when using this method, the use of {@link SDLoss} ops and any other ops that don't accept an output gradient is <b>FORBIDDEN</b>.
     * This is due to the fact that we have to do sum/average ops to the output of this function.
     * If those ops are nessecary, implement {@link #defineLoss(SameDiff, SDVariable, SDVariable, boolean)} instead.<br>
     *
     * @param sameDiff The {@link SameDiff} instance
     * @param input The input to the loss function, typically the output of the previous layer.
     * @param labels The labels to compare the output to.  Should be the same shape as input.
     * @return The score array.  The first dimension should be the batch, so it has shape {@code [batchSize, ...]}.
     */
    protected SDVariable defineLossArray(@NonNull SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable labels){
        throw new UnsupportedOperationException("defineLossArray not implemented for " + this.getClass().getSimpleName());
    }

    /**
     *
     * Define the loss function for a {@link SameDiff} instance.  Should return a scalar. <br>
     *
     * If average is true, should be the batchwise average, otherwise the sum.<br>
     *
     * The default implementation of this method calls {@link #defineLossArray(SameDiff, SDVariable, SDVariable)} and then averages it if necessary.
     * If you are not using {@link SDLoss} ops often it is easier to implement {@link #defineLossArray(SameDiff, SDVariable, SDVariable)} when extending this class.
     * However, if you are, you must implement this method instead.  See {@link BaseLossFunction}. <br>
     *
     * Note that using a {@link org.nd4j.autodiff.samediff.ops.SDLoss} function with {@link org.nd4j.autodiff.loss.LossReduce} MEAN_BY_NONZERO_WEIGHT_COUNT
     * will result in the loss values for DL4J and SameDiff being slightly different, but is as close as you can get.
     * DL4J gets the loss with average=false and then averages it itself, while SameDiff will work with what you pass it.
     *
     * @see BaseLossFunction
     * @param sameDiff The {@link SameDiff} instance
     * @param input The input to the loss function, typically the output of the previous layer.
     * @param labels The labels to compare the output to.  Should be the same shape as input.
     * @param average Whether to average the loss per example.
     * @return The scalar score (loss function value).
     */
    @Override
    public SDVariable defineLoss(@NonNull SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable labels, boolean average) {
        try{
            return reduceLossArray(defineLossArray(sameDiff, input, labels), labels, average);
        } catch (UnsupportedOperationException e){
            throw new UnsupportedOperationException("SameDiff conversion has not been implemented for " + this.getClass().getSimpleName(), e);
        }
    }
}