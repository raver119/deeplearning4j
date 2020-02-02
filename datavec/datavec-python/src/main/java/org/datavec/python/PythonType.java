/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.python;

import lombok.Data;
import org.bytedeco.javacpp.BytePointer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;

import static org.datavec.python.Python.importModule;


public abstract class PythonType<T> {

    public abstract T toJava(PythonObject pythonObject) throws PythonException;
    public abstract TypeName getName();
    public String toString(){
        return getName().name();
    }
    enum TypeName{
        STR,
        INT,
        FLOAT,
        BOOL,
        LIST,
        DICT,
        NDARRAY,
        BYTES
    }

    public static PythonType valueOf(String typeName) throws PythonException{
        try{
            typeName.valueOf(typeName);
        } catch (IllegalArgumentException iae){
            throw new PythonException("Invalid python type: " + typeName, iae);
        }
        try{
            return (PythonType)PythonType.class.getField(typeName).get(null); // shouldn't fail
        } catch (Exception e){
            throw new RuntimeException(e);
        }

    }
    public static PythonType valueOf(TypeName typeName){
        try{
            return valueOf(typeName.name()); // shouldn't fail
        }catch (PythonException pe){
            throw new RuntimeException(pe);
        }

    }
    
    public static final PythonType<String> STR = new PythonType<String>() {
        @Override
        public String toJava(PythonObject pythonObject)  throws PythonException{
            if (!Python.isinstance(pythonObject, Python.strType())){
                throw new PythonException("Expected variable to be float, but was " +  Python.type(pythonObject));
            }
            return pythonObject.toString();
        }
        @Override
        public TypeName getName(){
            return TypeName.STR;
        }
    };
    public static final PythonType<Long> INT = new PythonType<Long>() {
        @Override
        public Long toJava(PythonObject pythonObject) throws PythonException{
            if (!Python.isinstance(pythonObject, Python.intType())){
                throw new PythonException("Expected variable to be int, but was " +  Python.type(pythonObject));
            }
            return pythonObject.toLong();
        }
        @Override
        public TypeName getName(){
            return TypeName.INT;
        }
    };
    public static final PythonType<Double> FLOAT = new PythonType<Double>() {
        @Override
        public Double toJava(PythonObject pythonObject) throws PythonException {
            if (!Python.isinstance(pythonObject, Python.floatType())){
                throw new PythonException("Expected variable to be float, but was " +  Python.type(pythonObject));
            }
            return pythonObject.toDouble();
        }
        @Override
        public TypeName getName(){
            return TypeName.FLOAT;
        }
    };
    public static final PythonType<Boolean> BOOL = new PythonType<Boolean>() {
        @Override
        public Boolean toJava(PythonObject pythonObject) throws PythonException{
            if (!Python.isinstance(pythonObject, Python.boolType())){
                throw new PythonException("Expected variable to be float, but was " +  Python.type(pythonObject));
            }
            return pythonObject.toBoolean();
        }
        @Override
        public TypeName getName(){
            return TypeName.BOOL;
        }
    };

    public static final PythonType<List> LIST = new PythonType<List>() {
        @Override
        public List toJava(PythonObject pythonObject) throws PythonException{
            if (!Python.isinstance(pythonObject, Python.listType())){
                throw new PythonException("Expected variable to be float, but was " +  Python.type(pythonObject));
            }
            return pythonObject.toList();
        }
        @Override
        public TypeName getName(){
            return TypeName.LIST;
        }
    };

    public static final PythonType<Map> DICT = new PythonType<Map>() {
        @Override
        public Map toJava(PythonObject pythonObject) throws PythonException{
            if (!Python.isinstance(pythonObject, Python.dictType())){
                throw new PythonException("Expected variable to be float, but was " +  Python.type(pythonObject));
            }
            return pythonObject.toMap();
        }
        @Override
        public TypeName getName(){
            return TypeName.DICT;
        }
    };
    public static final PythonType<INDArray> NDARRAY = new PythonType<INDArray>() {
        @Override
        public INDArray toJava(PythonObject pythonObject) throws PythonException{
            PythonObject np = importModule("numpy");
            if (!Python.isinstance(pythonObject, np.attr("ndarray"), np.attr("generic"))){
                throw new PythonException("Expected variable to be numpy.ndarray, but was " +  Python.type(pythonObject));
            }
            return pythonObject.toNumpy().getNd4jArray();
        }
        @Override
        public TypeName getName(){
            return TypeName.NDARRAY;
        }
    };

    public static final PythonType<BytePointer> BYTES = new PythonType<BytePointer>() {
        @Override
        public BytePointer toJava(PythonObject pythonObject) throws PythonException {
           return pythonObject.toBytePointer();
        }

        @Override
        public TypeName getName() {
            return TypeName.BYTES;
        }
    };
}
