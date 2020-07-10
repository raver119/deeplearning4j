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

package org.nd4j.imports.keras.deserialize;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import org.nd4j.shade.jackson.annotation.JacksonAnnotationsInside;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo.As;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo.Id;
import org.nd4j.shade.jackson.databind.PropertyNamingStrategy.SnakeCaseStrategy;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonNaming;

@Retention(RetentionPolicy.RUNTIME)
@JacksonAnnotationsInside
@JsonTypeInfo(use = Id.NAME, include = As.EXTERNAL_PROPERTY, property = "class_name")
@JsonNaming(SnakeCaseStrategy.class)
//@JsonDeserialize(using = KerasWrapperDeserializer.class)
public @interface KerasWrappedJson {

}
