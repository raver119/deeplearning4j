/*******************************************************************************
 * Copyright (c) 2019-2020 Konduit K.K.
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
package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

@NoArgsConstructor
public class Qr extends DynamicCustomOp {

    public Qr(INDArray input) {
        this(input, false);
    }
    public Qr(INDArray input, boolean fullMatrices) {
        addInputArgument(input);
        addBArgument(fullMatrices);
    }

    public Qr(SameDiff sameDiff, SDVariable input, boolean fullMatrices) {
        super(sameDiff, new SDVariable[]{input});
        addBArgument(fullMatrices);
    }

    @Override
    public String opName() {
        return "qr";
    }
}
