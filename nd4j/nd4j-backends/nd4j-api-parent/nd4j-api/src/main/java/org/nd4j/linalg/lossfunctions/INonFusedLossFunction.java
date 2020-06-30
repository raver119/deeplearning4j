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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.ops.SDLoss;

/**
 * A loss function whose defineLoss method does not use {@link SDLoss} ops, and defines the loss as a [batch, ...] array.
 *
 * You most likely want to extend {@link NonFusedLossFunction} instead of implementing this directly.
 */
public interface INonFusedLossFunction  extends ILossFunction {

    /**
     * Define the loss array calculation.
     *
     * DO NOT USE {@link SDLoss} METHODS!
     *
     * @return Loss array of shape [batch, ...]
     */
    SDVariable defineLossArray(SameDiff sameDiff, SDVariable input, SDVariable labels);
}
