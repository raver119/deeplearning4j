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
import org.nd4j.imports.keras.layers.KerasLayer;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.core.TreeNode;
import org.nd4j.shade.jackson.databind.BeanProperty;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JavaType;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonMappingException;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.deser.ContextualDeserializer;
import org.nd4j.shade.jackson.databind.deser.ResolvableDeserializer;
import org.nd4j.shade.jackson.databind.deser.std.StdDeserializer;
import org.nd4j.shade.jackson.databind.node.ObjectNode;

/**
 * Keras serializes config objects like
 * <pre>{@code
 * {
 *     "class_name": "Sequential",
 *     "config": {
 *         // .. Sequential's fields
 *     }
 * }
 * }</pre>
 */
public class KerasWrapperDeserializer extends StdDeserializer<Object> implements ContextualDeserializer,
        ResolvableDeserializer {

    public static final String CONFIG_PROP_NAME = "config";

    private JavaType innerType;
    private JsonDeserializer<?> defaultDeserializer;

    public KerasWrapperDeserializer(JsonDeserializer<?> defaultDeserializer, Class<?> klass){
        super(klass);
        this.defaultDeserializer = defaultDeserializer;
    }

    @Override
    public JsonDeserializer<?> createContextual(DeserializationContext deserializationContext,
            BeanProperty beanProperty) throws JsonMappingException {
        if(beanProperty != null)
            innerType = beanProperty.getType();
        else
            innerType = deserializationContext.getContextualType();

        return this;
    }

    @Override
    public void resolve(DeserializationContext deserializationContext) throws JsonMappingException {
        if(defaultDeserializer instanceof ResolvableDeserializer)
            ((ResolvableDeserializer) defaultDeserializer).resolve(deserializationContext);
    }

    @Override
    public Object deserialize(JsonParser jsonParser, DeserializationContext deserializationContext)
            throws IOException, JsonProcessingException {
        JsonNode innerNode = (JsonNode) jsonParser.readValueAsTree().get(CONFIG_PROP_NAME);

        // some versions of Keras set config to the layers array
        if(innerNode.isArray()){
            innerNode = deserializationContext.getNodeFactory()
                    .objectNode().set("layers", innerNode);
        }

        JsonParser inner = innerNode.traverse(jsonParser.getCodec());

        inner.nextToken();
        return defaultDeserializer.deserialize(inner, deserializationContext);
    }
}
