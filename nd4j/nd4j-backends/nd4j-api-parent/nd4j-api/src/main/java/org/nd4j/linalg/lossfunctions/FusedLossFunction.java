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
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.ops.SDLoss;

/**
 * A loss function whose defineLoss method can use {@link SDLoss} ops.
 */
public abstract class FusedLossFunction extends BaseLossFunction {
    /**
     * Define the loss array calculation.
     *
     * Should probably use {@link SDLoss} methods.
     *
     * @return Loss array of shape [batch, ...]
     */
    protected abstract SDVariable defineLoss(SameDiff sameDiff, SDVariable input, SDVariable labels, LossReduce reduction);

    @Override
    public final SDVariable defineLoss(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
            @NonNull SDVariable labels, boolean average) {
        return defineLoss(sameDiff, input, labels, average ? LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT : LossReduce.SUM);
    }
}
