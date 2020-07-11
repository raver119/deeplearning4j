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

package org.nd4j.imports.keras.layers;

import java.util.Map;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.nd4j.imports.keras.activations.IKerasActivation;
import org.nd4j.imports.keras.constraints.KerasConstraint;
import org.nd4j.imports.keras.initalizers.KerasWeightInitializer;
import org.nd4j.imports.keras.regularizers.KerasRegularizer;

@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Getter
@NoArgsConstructor
public class Dense extends KerasLayer{
    protected int units;
    protected IKerasActivation activation;
    protected boolean useBias;

    protected KerasWeightInitializer kernelInitializer;
    protected KerasWeightInitializer biasInitializer;
    protected KerasRegularizer kernelRegularizer;
    protected KerasRegularizer biasRegularizer;
    protected KerasRegularizer activityRegularizer;
    protected KerasConstraint kernelConstraint;
    protected KerasConstraint biasConstraint;
}
