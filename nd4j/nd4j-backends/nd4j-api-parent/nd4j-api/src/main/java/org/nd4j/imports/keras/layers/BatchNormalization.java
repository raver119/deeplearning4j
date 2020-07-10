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
import lombok.Value;

@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Getter
@NoArgsConstructor
public class BatchNormalization extends KerasLayer {
    protected int axis = -1;
    protected double momentum = 0.0;
    protected double epsilon = 0.0;
    protected boolean center = false;
    protected boolean scale = false;

    protected Map<String, Object> betaInitializer = null;
    protected Map<String, Object> gammaInitializer = null;
    protected Map<String, Object> movingMeanInitializer = null;
    protected Map<String, Object> movingVarianceInitializer = null;
    protected Map<String, Object> betaRegularizer = null;
    protected Map<String, Object> gammaRegularizer = null;
    protected Map<String, Object> betaConstraint = null;
    protected Map<String, Object> gammaConstraint = null;

    protected boolean renorm = false;
    protected Map<String, Double> renormClipping = null;
    protected double renormMomentum = 0.0;
    protected int virtualBatchSize = 0;

}
