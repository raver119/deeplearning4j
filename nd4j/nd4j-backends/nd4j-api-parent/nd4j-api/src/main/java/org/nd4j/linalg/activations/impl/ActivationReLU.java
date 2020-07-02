/* *****************************************************************************
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

package org.nd4j.linalg.activations.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.LeakyReLUBp;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * f(x) = max(0, x)
 */
@EqualsAndHashCode(callSuper = false)
@Getter
public class ActivationReLU extends BaseActivationFunction {

    private Double max;
    private Double threshold;
    private Double negativeSlope;

    public ActivationReLU(){
        this(null, null, null);
    }

    public ActivationReLU(Double maxValue, Double threshold, Double negativeSlope){
        this.max = maxValue;
        this.threshold = threshold;
        this.negativeSlope = negativeSlope;
    }

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        if(negativeSlope != null || threshold != null){
            double t = threshold == null ? 0.0 : threshold;
            double ns = negativeSlope == null ? 0.0 : negativeSlope;
            if(t == 0.0) {
                Nd4j.getExecutioner().execAndReturn(new LeakyReLU(in, ns));
            } else {
                //Non-zero threshold, and non-zero slope
                //TODO optimize this... but, extremely rare case in practice?
                INDArray oneGte = in.gte(t).castTo(in.dataType());
                INDArray oneLt = in.lt(t).castTo(in.dataType());
                INDArray lower = oneLt.muli(ns).muli(in.sub(threshold));
                INDArray upper = oneGte.muli(in);
                in.assign(lower.addi(upper));
            }
        } else {
            Nd4j.getExecutioner().exec(new RectifiedLinear(in, in));
        }
        if(max != null){
            Nd4j.exec(new ScalarMin(in, null, in, max));
        }
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);

        INDArray dLdz;
        INDArray maxMask = (max == null || max == 0.0 ? null : in.lt(max));
        if(negativeSlope != null || threshold != null){
            double t = threshold == null ? 0.0 : threshold;
            double ns = negativeSlope == null ? 0.0 : negativeSlope;
            if(t == 0.0) {
                dLdz = Nd4j.getExecutioner().exec(new LeakyReLUBp(in, epsilon, in.ulike(), ns))[0];
            } else {
                //Non-zero threshold, and non-zero slope
                //TODO optimize this... but, extremely rare case in practice?
                INDArray oneGte = in.gte(t).castTo(in.dataType());
                INDArray oneLt = in.lt(t).castTo(in.dataType());
                INDArray lower = oneLt.muli(ns);
                INDArray upper = oneGte;
                dLdz = in.assign(lower.addi(upper)).muli(epsilon);
            }
        } else {
            dLdz = Nd4j.getExecutioner().exec(new RectifiedLinearDerivative(in, epsilon, in.ulike(), threshold == null ? 0.0 : threshold))[0];
        }

        if(maxMask != null){
            dLdz.muli(maxMask);
        }
        return new Pair<>(dLdz, null);
    }

    @Override
    public SDVariable defineActivation(@NonNull SameDiff sameDiff, @NonNull SDVariable input) {
        SDVariable temp;
        double thresh = threshold == null ? 0.0 : threshold;
        double ns = negativeSlope == null ? 0.0 : negativeSlope;
        if(ns == 0){
            temp = sameDiff.nn.relu(input, thresh);
        } else {
            if(thresh == 0)
                temp = sameDiff.nn.leakyRelu(input, negativeSlope);
            else {
                //TODO optimize this
                SDVariable t = sameDiff.constant(Nd4j.scalar(input.dataType(), thresh));
                SDVariable oneGte = input.gte(t).castTo(input.dataType());
                SDVariable oneLt = input.lt(t).castTo(input.dataType());
                SDVariable lower = oneLt.mul(ns).mul(input.sub(threshold));
                SDVariable upper = oneGte.mul(input);
                temp = lower.add(upper);
            }
        }

        if(max != null)
            temp = sameDiff.math.max(sameDiff.constant(Nd4j.scalar(temp.dataType(), max)), temp);

        return temp;
    }

    @Override
    public String toString() {
        return "relu";
    }

}
