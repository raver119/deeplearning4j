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

package org.nd4j.imports.keras;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import org.nd4j.imports.keras.activations.IKerasActivation;
import org.nd4j.imports.keras.deserialize.KerasActivationDeserializer;
import org.nd4j.imports.keras.deserialize.KerasDatatypeDeserializer;
import org.nd4j.imports.keras.deserialize.KerasNames;
import org.nd4j.imports.keras.deserialize.KerasPaddingDeserializer;
import org.nd4j.imports.keras.deserialize.KerasWrappedJson;
import org.nd4j.imports.keras.deserialize.KerasWrapperDeserializer;
import org.nd4j.imports.keras.layers.KerasLayer;
import org.nd4j.imports.keras.layers.KerasSingleLayer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.shade.guava.reflect.ClassPath;
import org.nd4j.shade.jackson.databind.BeanDescription;
import org.nd4j.shade.jackson.databind.DeserializationConfig;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.deser.BeanDeserializerModifier;
import org.nd4j.shade.jackson.databind.jsontype.NamedType;
import org.nd4j.shade.jackson.databind.module.SimpleModule;

public class KerasImportUtils {

    private static final Set<Class<? extends KerasLayer>> layers = findClasses(KerasLayer.class, "org.nd4j.imports.keras.layers", "org.nd4j.imports.keras.activations");

    public static <T> Set<Class<? extends T>> findClasses(Class<T> baseClass, String... packageNames){
        Set<Class<? extends T>> classes = new HashSet<>();

        for(String packageName : packageNames){
            try {
                Set<ClassPath.ClassInfo> classInfos = ClassPath.from(baseClass.getClassLoader())
                        .getTopLevelClassesRecursive(packageName);
                for(ClassPath.ClassInfo info : classInfos){
                    Class<?> klass = info.load();
                    if(baseClass != klass && baseClass.isAssignableFrom(klass))
                        classes.add(klass.asSubclass(baseClass));
                }
            } catch (IOException ignored){

            }
        }

        return classes;
    }

    public static ObjectMapper kerasMapper(){
        SimpleModule module = new SimpleModule();
        module.setDeserializerModifier(new BeanDeserializerModifier() {
            @Override
            public JsonDeserializer<?> modifyDeserializer(DeserializationConfig config, BeanDescription beanDesc,
                    JsonDeserializer<?> deserializer) {

                Class<?> current = beanDesc.getBeanClass();

                while(current != null){
                    if(current.isAnnotationPresent(KerasWrappedJson.class))
                        return new KerasWrapperDeserializer(deserializer, beanDesc.getBeanClass());

                    current = current.getSuperclass();
                }

                return deserializer;
            }
        });

        module.addDeserializer(DataType.class, new KerasDatatypeDeserializer());
        module.addDeserializer(PaddingMode.class, new KerasPaddingDeserializer());
        // have to add this here instead of the annotation for some reason, otherwise Jackson doesn't use the KerasLayer serializer on the layer activations
        module.addDeserializer(IKerasActivation.class, new KerasActivationDeserializer());

        ObjectMapper mapper = new ObjectMapper();
        mapper.registerModule(module);

        for(Class<? extends KerasLayer> klass : layers){
            if(klass.isAnnotationPresent(KerasNames.class)) {
                for (String name : klass.getAnnotation(KerasNames.class).value())
                    mapper.registerSubtypes(new NamedType(klass, name));
            } else {
                mapper.registerSubtypes(new NamedType(klass, klass.getSimpleName()));
            }

        }

        mapper.configure(DeserializationFeature.FAIL_ON_IGNORED_PROPERTIES, false);
        mapper.configure(DeserializationFeature.ACCEPT_SINGLE_VALUE_AS_ARRAY, true);

        return mapper;
    }
}
