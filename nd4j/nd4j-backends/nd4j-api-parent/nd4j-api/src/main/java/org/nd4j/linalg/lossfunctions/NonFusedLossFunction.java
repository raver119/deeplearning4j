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
import org.nd4j.linalg.factory.Nd4j;

/**
 * A loss function whose defineLoss method does not use {@link SDLoss} ops.
 */
public abstract class NonFusedLossFunction extends BaseLossFunction implements INonFusedLossFunction {

    /**
     * Define the loss array calculation.
     *
     * DO NOT USE {@link SDLoss} METHODS!
     *
     * @return Loss array of shape [batch, ...]
     */
    @Override
    public abstract SDVariable defineLossArray(SameDiff sameDiff, SDVariable input, SDVariable labels);

    protected SDVariable batchAverage(SDVariable output, SDVariable labels, boolean average){
        if(average)
            return output.div(labels.shape().get(SDIndex.point(0)));
        else
            return output;
    }

    protected SDVariable reduce(SDVariable output, SDVariable labels, boolean average){
        SameDiff sameDiff = output.getSameDiff();
        SDVariable batchSize = sameDiff.sizeAt(labels, 0);
        SDVariable newShape = sameDiff.concat(0, batchSize, sameDiff.constant(Nd4j.scalar(batchSize.dataType(), -1)));
        output = output.reshape(newShape).sum();
        return batchAverage(output, labels, average);
    }

    @Override
    public final SDVariable defineLoss(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
            @NonNull SDVariable labels, boolean average) {
        SDVariable output = defineLossArray(sameDiff, input, labels);
        return reduce(output, labels, average);
    }
}
