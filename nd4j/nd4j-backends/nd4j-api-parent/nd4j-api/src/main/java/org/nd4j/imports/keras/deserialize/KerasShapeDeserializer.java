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
import java.util.ArrayList;
import java.util.List;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.shade.jackson.core.JsonParseException;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.core.JsonToken;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.deser.std.StdDeserializer;

public class KerasShapeDeserializer extends StdDeserializer<int[]> {

    protected KerasShapeDeserializer() {
        super(int[].class);
    }

    @Override
    public int[] deserialize(JsonParser jsonParser, DeserializationContext deserializationContext)
            throws IOException, JsonProcessingException {
        if (jsonParser.getCurrentToken() != JsonToken.START_ARRAY) {
            throw new JsonParseException("Start array expected", jsonParser.getCurrentLocation());
        }

        List<Integer> items = new ArrayList<>();
        JsonToken token = jsonParser.nextToken();
        while (token != JsonToken.END_ARRAY){

            if(token == JsonToken.VALUE_NULL)
                items.add(-1);
            else
                items.add(jsonParser.getIntValue());

            token = jsonParser.nextToken();
        }

        return ArrayUtil.toArray(items);
    }
}
