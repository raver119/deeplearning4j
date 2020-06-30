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

package org.deeplearning4j.nn.conf.preprocessor;

import lombok.NonNull;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.common.primitives.Pair;

/**
 * @author Adam Gibson
 */

public abstract class BaseInputPreProcessor implements InputPreProcessor {
    @Override
    public BaseInputPreProcessor clone() {
        try {
            BaseInputPreProcessor clone = (BaseInputPreProcessor) super.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }


    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                    int minibatchSize) {
        //Default: pass-through, unmodified
        return new Pair<>(maskArray, currentMaskState);
    }

    public SDVariable definePreProcess(@NonNull SameDiff sameDiff, @NonNull SDVariable input){
        throw new UnsupportedOperationException("SameDiff conversion has not been implemented for " + this.getClass().getSimpleName());
    }
}
