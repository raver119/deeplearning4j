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

package org.nd4j.imports.keras.models;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.nd4j.imports.keras.deserialize.KerasShapeDeserializer;
import org.nd4j.imports.keras.deserialize.KerasWrappedJson;
import org.nd4j.imports.keras.layers.KerasSingleLayer;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonSubTypes.Type;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;

@Getter
@NoArgsConstructor
@ToString
@EqualsAndHashCode
@KerasWrappedJson
@JsonSubTypes({
        @Type(Sequential.class),
        @Type(Functional.class)
})
public abstract class KerasModel {

    protected String kerasVersion;
    protected String backend;
    protected String name;
    @JsonDeserialize(using = KerasShapeDeserializer.class)
    protected int[] buildInputShape;
}
