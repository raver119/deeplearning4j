/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

import org.bytedeco.arrow.*;
import org.datavec.arrow.table.DataVecTable;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.bytedeco.arrow.global.arrow.*;

public class PythonArrowUtils {

    static {
        try{
            new Field("x", int32());
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }
    public static PythonObject importPyArraow() throws PythonException{
        return Python.importModule("pyarrow");
    }

    public static PythonObject getPyArrowArrayFromINDArray(INDArray arr) throws PythonException{
        PythonObject pyarrow = importPyArraow();
        PythonObject npArr = new PythonObject(arr);
        PythonObject arrayF = pyarrow.attr("array");
        PythonObject ret = arrayF.call(npArr);
        pyarrow.del();
        npArr.del();
        arrayF.del();
        return ret;
    }

    public static INDArray getINDArrayFromPyArrowArray(PythonObject arr) throws PythonException{
        PythonObject pyArrow = Python.importModule("pyarrow");
        PythonObject arrayType = pyArrow.attr("Array");
        if (!Python.isinstance(arr, arrayType)){
            pyArrow.del();
            arrayType.del();
            throw new PythonException("Expected pyarrow.Array, received " + Python.type(arr));
        }
        PythonObject toNumpyF = arr.attr("to_numpy");
        PythonObject npArr = toNumpyF.call();
        pyArrow.del();
        arrayType.del();
        toNumpyF.del();
        INDArray ret = npArr.toNumpy().getNd4jArray();
        npArr.del();
        return ret;
    }
//    private static void installPyArrow() throws Exception{
//        try{
//            importPyArraow().del();
//        }catch (PythonException pe){
//            String python = Loader.load(org.bytedeco.cpython.python.class);
//            ProcessBuilder pb = new ProcessBuilder(python, "-m", "pip", "install", "pyarrow==0.15.1");
//            pb.inheritIO().start().waitFor();
//            importPyArraow().del();
//        }
//    }


    public static PythonObject getPyArrowField(Field field) throws PythonException{
        String name = field.name();
        org.bytedeco.arrow.DataType type = field.type();
       String typeName = type.name();
       String pyTypeFName;
       switch (typeName){
           case "list":
               throw new PythonException("Unsupported field type: list");
           case "bool":
               pyTypeFName = "bool_";
               break;
           case "halffloat":
               pyTypeFName = "float16";
               break;
           case "float":
               pyTypeFName = "float32";
               break;
           case "double":
               pyTypeFName = "float64";
               break;
           default:
               pyTypeFName = typeName;
       }
       PythonObject pyarrow = importPyArraow();
       PythonObject fieldF = pyarrow.attr("field");
       PythonObject pyArrowTypeF = pyarrow.attr(pyTypeFName);
       PythonObject pyArrowType = pyArrowTypeF.call();
       PythonObject pyArrowField = fieldF.call(name, pyArrowType);
       pyArrowType.del();
       pyArrowTypeF.del();
       fieldF.del();
       pyarrow.del();
       return pyArrowField;
    }

    public static Field getFieldFromPythonObject(PythonObject pyArrowField) throws PythonException{
        PythonObject pyarrow = importPyArraow();
        PythonObject fieldType = pyarrow.attr("Field");
        if(!Python.isinstance(pyArrowField, fieldType)){
            pyarrow.del();
            fieldType.del();
            throw new PythonException("Expected pyarrow.Field, received " + Python.type(pyArrowField));
        }
        PythonObject pyName = pyArrowField.attr("name");
        String name = pyName.toString();
        PythonObject pyTypeName = pyArrowField.attr("type");
        String typeName = pyTypeName.toString();
        DataType dt;
        switch (typeName){
            case "bool":
                dt  = _boolean();
                break;
            case "halffloat":
                dt = float16();
                break;
            case "float":
                dt = float32();
                break;
            case "double":
                dt = float64();
                break;
                default:
                    try{
                        dt = (DataType)org.bytedeco.arrow.global.arrow.class.getMethod(typeName).invoke(null);
                    }
                    catch (Exception e){
                        throw new PythonException("Unsupported type: " + typeName, e);
                    }
        }
        Field ret = new Field(name, dt);
        pyarrow.del();
        fieldType.del();
        pyName.del();
        pyTypeName.del();
        return ret;
    }

    public static PythonObject getPyArrowSchema(Schema schema) throws PythonException{
        Field[] fields = schema.fields().get();
        PythonObject[] pyFields = new PythonObject[fields.length];
        for (int i = 0; i < fields.length; i++){
            pyFields[i] = getPyArrowField(fields[i]);
        }
        PythonObject pyarrow = importPyArraow();
        PythonObject schemaF = pyarrow.attr("schema");
        PythonObject pySchema = schemaF.call(Python.list(pyFields));
        pyarrow.del();
        schemaF.del();
        return pySchema;
    }

    public static Schema getSchemaFromPythonObject(PythonObject pyArrowSchema) throws PythonException{
        PythonObject pyarrow = importPyArraow();
        PythonObject schemaType = pyarrow.attr("Schema");
        if(!Python.isinstance(pyArrowSchema, schemaType)){
            pyarrow.del();
            schemaType.del();
            throw new PythonException("Expected pyarrow.Field, received " + Python.type(pyArrowSchema));
        }
        PythonObject pySize = Python.len(pyArrowSchema);
        int size = pySize.toInt();
        Field[] fields = new Field[size];
        for(int i = 0; i < size; i++){
            PythonObject pyField = pyArrowSchema.get(i);
            fields[i] = getFieldFromPythonObject(pyField);
        }
        pySize.del();
        schemaType.del();
        pyarrow.del();
        return new Schema(new FieldVector(fields));
    }

    public static PythonObject getPyArrowTable(DataVecTable table) throws PythonException{
        PythonObject d = Python.dict();
        for(int i = 0; i < table.numColumns(); i++){
            PythonObject colName = new PythonObject(table.columnNameAt(i));
            PythonObject colArr = new PythonObject(table.column(i).toNdArray());
            d.set(colName, colArr);
            //colName.del();
            //colArr.del();
        }
        PythonObject pyarrow = importPyArraow();
        PythonObject tableF = pyarrow.attr("table");
        PythonObject pyTable = tableF.call(d);
        pyarrow.del();
        tableF.del();
        return pyTable;
    }

}
