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

package org.nd4j.imports.keras.layers.convolutional;

import java.util.Collections;
import java.util.Map;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.ToString;
import lombok.Value;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.enums.DataFormat;
import org.nd4j.imports.keras.deserialize.KerasDataFormatDeserializer;
import org.nd4j.imports.keras.deserialize.KerasWrappedJson;
import org.nd4j.imports.keras.layers.KerasLayer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Getter
@ToString(callSuper = true)
@NoArgsConstructor
public abstract class Conv extends KerasLayer {



    abstract int rank();

    protected @NonNull int filters;
    protected @NonNull int[] kernelSize;
    protected @NonNull int[] strides = ArrayUtil.toArray(Collections.nCopies(rank(), 1));
    protected @NonNull PaddingMode padding = PaddingMode.VALID;
    protected @NonNull DataFormat dataFormat = DataFormat.NHWC;

    @JsonProperty("dilation_rate")
    protected int[] dilation = ArrayUtil.toArray(Collections.nCopies(rank(), 1));

    protected int groups = 1;

    protected String activation;

    protected boolean useBias = true;

    protected Map<String, Object> kernelInitializer;
    protected Map<String, Object> biasInitializer;
    protected Map<String, Object> kernelRegularizer;
    protected Map<String, Object> biasRegularizer;
    protected Map<String, Object> activityRegularizer;
    protected Map<String, Object> kernelConstraint;
    protected Map<String, Object> biasConstraint;
}
