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

package org.deeplearning4j.nn.conf.layers;

import lombok.NonNull;
import org.deeplearning4j.nn.layers.ocnn.OCNNOutputLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Any loss or output layers that support SameDiff conversion must implement this.
 */
public interface LayerWithLoss {

    /**
     * Define the layer's loss function.  Should return a scalar. <br>
     *
     * If average is true, should be the batchwise average, otherwise the sum.
     * @param sameDiff The {@link SameDiff} instance
     * @param input The input to the loss function, the output (activations) of this layer.
     * @param labels The labels to compare the output to.  The placeholder will be created with the shape of the output (activations) of this layer.  May be null if the implementation layer doesn't require labels (e.g. {@link OCNNOutputLayer}.
     * @param average Whether to average the loss per example.  Most of the time this should be passed to the {@link org.nd4j.linalg.lossfunctions.ILossFunction}.
     * @return The loss scalar.
     */
    SDVariable defineLoss(@NonNull SameDiff sameDiff, @NonNull SDVariable input, SDVariable labels, boolean average);
}
