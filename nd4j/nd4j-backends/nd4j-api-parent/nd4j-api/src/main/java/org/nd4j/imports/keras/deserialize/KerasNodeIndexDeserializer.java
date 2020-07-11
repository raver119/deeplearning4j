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
import java.util.Map;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.imports.keras.KerasNodeIndex;
import org.nd4j.shade.jackson.core.JsonParseException;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.core.JsonToken;
import org.nd4j.shade.jackson.core.type.TypeReference;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.deser.std.StdDeserializer;

public class KerasNodeIndexDeserializer extends StdDeserializer<KerasNodeIndex> {

    public KerasNodeIndexDeserializer(){
        super(KerasNodeIndex.class);
    }

    @Override
    public KerasNodeIndex deserialize(JsonParser jsonParser, DeserializationContext deserializationContext)
            throws IOException, JsonProcessingException {
        if (jsonParser.currentToken() != JsonToken.START_ARRAY) {
            throw new JsonParseException("Start array expected", jsonParser.getCurrentLocation());
        }

        KerasNodeIndex node = new KerasNodeIndex();

        node.setLayerName(jsonParser.nextTextValue());

        jsonParser.nextToken();
        node.setNodeIndex(jsonParser.getIntValue());

        jsonParser.nextToken();
        node.setTensorIndex(jsonParser.getIntValue());

        JsonToken token = jsonParser.nextToken();

        if(token == JsonToken.END_ARRAY)
            return node;
        else{
            Map<String, Object> kwargs = deserializationContext.readValue(jsonParser,
                    deserializationContext.getTypeFactory().constructMapLikeType(Map.class, String.class, Object.class));
            node.setKwargs(kwargs);
        }

        token = jsonParser.nextToken();
        if(token != JsonToken.END_ARRAY)
            throw new JsonParseException("Expected 3 or 4 elements in the Keras Token array, got more", jsonParser.getCurrentLocation());

        return node;
    }
}
