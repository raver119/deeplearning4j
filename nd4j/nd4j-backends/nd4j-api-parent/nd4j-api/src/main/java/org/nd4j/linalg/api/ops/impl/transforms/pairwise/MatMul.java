package org.nd4j.linalg.api.ops.impl.transforms.pairwise;

/*******************************************************************************
 * Copyright (c) 2019 - 2020 Konduit K. K.
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

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;

import java.util.List;

@NoArgsConstructor
public class MatMul  extends BaseDynamicTransformOp {

    public MatMul( SameDiff sameDiff, SDVariable first, SDVariable second)  {
        super(sameDiff, new SDVariable[]{first, second}, false);
    }

    public MatMul( SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(sameDiff, args, inPlace);
    }

    public MatMul(INDArray first, INDArray second, INDArray result){
        this(new INDArray[]{first, second}, result == null ? null : new INDArray[]{result});
    }

    public MatMul( INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }


    @Override
    public String opName() {
        return "matmul";
    }


    @Override
    public String onnxName() {
        return "MatMul";
    }

    @Override
    public String tensorflowName() {
        return "MatMul";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return f().mulBp(larg(), rarg(), i_v.get(0));
    }
}

