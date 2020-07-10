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

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.ToString;
import org.nd4j.imports.keras.deserialize.KerasShapeDeserializer;
import org.nd4j.imports.keras.deserialize.KerasWrappedJson;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo.As;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo.Id;
import org.nd4j.shade.jackson.databind.PropertyNamingStrategy.SnakeCaseStrategy;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonNaming;

@KerasWrappedJson
@Getter
@NoArgsConstructor
@ToString
public abstract class KerasLayer {
    @NonNull protected String name;
    protected boolean trainable = true;
    @JsonDeserialize(using = KerasShapeDeserializer.class)
    protected int[] batchInputShape = null;
    protected DataType dtype = null;
}
