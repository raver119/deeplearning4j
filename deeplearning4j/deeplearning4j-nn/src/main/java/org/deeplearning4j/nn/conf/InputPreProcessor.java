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

package org.deeplearning4j.nn.conf;


import java.util.Map;
import lombok.NonNull;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * Input pre processor used
 * for pre processing input before passing it
 * to the neural network.
 *
 * You will most likely want to extend BaseInputPreProcessor when creating a custom preprocessor,
 * as it supplies default exception-throwing define* methods.
 *
 * @author Adam Gibson
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface InputPreProcessor extends Serializable, Cloneable {

    /**
     * Pre preProcess input/activations for a multi layer network
     * @param input the input to pre preProcess
     * @param miniBatchSize Minibatch size
     * @param workspaceMgr Workspace manager
     * @return the processed input. Note that the returned array should be placed in the
     *         {@link org.deeplearning4j.nn.workspace.ArrayType#ACTIVATIONS} workspace via the workspace manager
     */
    INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr);

    /**Reverse the preProcess during backprop. Process Gradient/epsilons before
     * passing them to the layer below.
     * @param output which is a pair of the gradient and epsilon
     * @param miniBatchSize Minibatch size
     * @param workspaceMgr Workspace manager
     * @return the reverse of the pre preProcess step (if any). Note that the returned array should be
     *         placed in {@link org.deeplearning4j.nn.workspace.ArrayType#ACTIVATION_GRAD} workspace via the
     *         workspace manager
     */
    INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr);

    InputPreProcessor clone();

    /**
     * For a given type of input to this preprocessor, what is the type of the output?
     *
     * @param inputType    Type of input for the preprocessor
     * @return             Type of input after applying the preprocessor
     */
    InputType getOutputType(InputType inputType);


    Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize);

    /**
     * Define the InputPreProcessor's input transformation in a {@link SameDiff} instance.
     * @param sameDiff The {@link SameDiff} instance.
     * @param input The input to transform.
     * @return The transformed input.
     */
    @NonNull SDVariable definePreProcess(@NonNull SameDiff sameDiff, @NonNull SDVariable input);
}
