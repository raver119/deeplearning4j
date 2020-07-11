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

import java.io.IOException;
import org.nd4j.imports.keras.activations.ELU;
import org.nd4j.imports.keras.activations.Exponential;
import org.nd4j.imports.keras.activations.IKerasActivation;
import org.nd4j.imports.keras.activations.Linear;
import org.nd4j.imports.keras.activations.ReLU;
import org.nd4j.imports.keras.activations.Selu;
import org.nd4j.imports.keras.activations.Sigmoid;
import org.nd4j.imports.keras.activations.Softmax;
import org.nd4j.imports.keras.activations.Softplus;
import org.nd4j.imports.keras.activations.Softsign;
import org.nd4j.imports.keras.activations.Tanh;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.deser.std.StdDeserializer;

public class KerasActivationDeserializer extends StdDeserializer<IKerasActivation> {

    public KerasActivationDeserializer(){
        super(IKerasActivation.class);
    }

    @Override
    public IKerasActivation deserialize(JsonParser jsonParser, DeserializationContext deserializationContext)
            throws IOException, JsonProcessingException {
        String activation = deserializationContext.readValue(jsonParser, String.class);
        switch (activation){
            case "relu":
//                return new ReLU();
            case "sigmoid":
                return new Sigmoid();
            case "softmax":
//                return new Softmax();
            case "softplus":
                return new Softplus();
            case "softsign":
                return new Softsign();
            case "tanh":
                return new Tanh();
            case "selu":
                return new Selu();
            case "elu":
//                return new ELU();
            case "exponential":
                return new Exponential();
            case "linear":
                return new Linear();
            default:
                throw new IllegalStateException("Unknown activation function " + activation + ".  Custom activations are only supported by SavedModel format exports and SameDiff Tensorflow import.");
        }
    }
}
