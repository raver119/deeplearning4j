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

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.arrow.*;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.datavec.arrow.table.DataVecTable;
import org.nd4j.arrow.ByteDecoArrowSerde;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.UUID;

import static org.bytedeco.arrow.global.arrow.*;

@Slf4j
public class PythonArrowUtils {

    static {
        init();
    }
    private static String PYARROW = "pyarrow";
    private static String PANDAS = "pandas";
    private static String REQUIRED_PYARROW_VERSION = "0.15.1"; // TODO get version from pom.xml?
    private static boolean init = false;

    public static void init(){
        // TODO: Find out why this works
        INDArray dummyArr = Nd4j.rand(5);
        new DoubleArray(new ArrowBuffer(new BytePointer(dummyArr.data().pointer()), dummyArr.data().length()));
    }

    public static PythonObject importPyArraow() throws PythonException{
        try{
            if (!PythonProcess.isPackageInstalled(PYARROW)){
                log.info("PyArrow is not installed. Attempting to pip install pyarrow " + REQUIRED_PYARROW_VERSION);
                PythonProcess.pipInstall(PYARROW, REQUIRED_PYARROW_VERSION);
                PythonProcess.pipInstall(PANDAS);
            }
            else{
                String pkgVersion = PythonProcess.getPackageVersion(PYARROW);
                if (!pkgVersion.equals(REQUIRED_PYARROW_VERSION)) {
                    log.info("Required pyarrow version " + REQUIRED_PYARROW_VERSION + " but current version is " + pkgVersion + ". Attempting reinstall...");
                    PythonProcess.pipInstall(PYARROW, REQUIRED_PYARROW_VERSION);
                    PythonProcess.pipInstall(PANDAS);
                }
            }
        } catch(Exception e){
            throw new PythonException("Error verifying/installing pyarrow package.", e);
        }

        return Python.importModule(PYARROW);
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
        PythonObject pyArrow = Python.importModule(PYARROW);
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


    public static PythonObject getPyArrowTable(Table table) throws PythonException{
        PythonObject d = Python.dict();
        Schema schema =table.schema();
        Field[] fields = schema.fields().get();
        for (int i = 0; i < fields.length; i++){
            String colName = fields[i].name();
            PythonObject pyColName = new PythonObject(colName);
            ChunkedArray chunkedArray = table.column(i);
            INDArray arr = Nd4j.create(ByteDecoArrowSerde.fromArrowBuffer(chunkedArray.chunk(0).null_bitmap(), fields[i].type()));
            PythonObject pyArr = new PythonObject(arr);
            d.set(pyColName, pyArr);
        }
        PythonObject pyarrow = importPyArraow();
        PythonObject tableF = pyarrow.attr("table");
        PythonObject pyTable = tableF.call(d);
        pyarrow.del();
        tableF.del();
        return pyTable;
    }

    public static Table getTableFromPythonObject(PythonObject pyTable) throws PythonException{
        PythonObject pyarrow = importPyArraow();
        PythonObject tableType = pyarrow.attr("Table");
        if (!Python.isinstance(pyTable, tableType)){
            if (Python.isinstance(pyTable, Python.dictType())){
                PythonObject orig = pyTable;
                PythonObject tableF = pyarrow.attr("table");
                pyTable = tableF.call(pyTable);
                orig.del();
                tableF.del();
            }
            else {
                throw new PythonException("Expected pyarrow.lib.Table or dict, received " + Python.type(pyTable));
            }
        }
        PythonObject pySchema = pyTable.attr("schema");
        PythonObject pyShemaSize = Python.len(pySchema);
        Field[] fields  = new Field[pyShemaSize.toInt()];
        Array[] arrays = new FlatArray[fields.length];
        String origContext = Python.getCurrentContext();
        String tempContext = 'a' + UUID.randomUUID().toString().replace('-','_' );
        Python.setContext(tempContext);
        for (int i = 0; i < fields.length; i++){
            Python.setVariable("col", pyTable.get(i));
            Python.exec("arr=col.to_pandas().to_numpy()");
            INDArray indArray = Python.getVariable("arr").toNumpy().getNd4jArray();
            fields[i] = getFieldFromPythonObject(pySchema.get(i));
            arrays[i] = new DoubleArray(new ArrowBuffer(new BytePointer(indArray.data().pointer()), indArray.data().length()));
        }

        Python.setContext(origContext);
        Python.deleteContext(tempContext);
        FieldVector fieldVector = new FieldVector(fields);
        Schema schema = new Schema(fieldVector);
        ArrayVector arrayVector = new ArrayVector(arrays);
        Table ret = Table.Make(schema, arrayVector);
        pySchema.del();
        pyShemaSize.del();
        tableType.del();
        pyarrow.del();
        return ret;

    }

    public static PythonObject getPyArrowTable(DataVecTable table) throws PythonException{

        PythonObject d = Python.dict();
        for(int i = 0; i < table.numColumns(); i++){
            PythonObject colName = new PythonObject(table.columnNameAt(i));
            PythonObject colArr = new PythonObject(table.column(i).toNdArray());
            d.set(colName, colArr);
            colName.del();
            colArr.del();
        }
        PythonObject pyarrow = importPyArraow();
        PythonObject tableF = pyarrow.attr("table");
        PythonObject pyTable = tableF.call(d);
        pyarrow.del();
        tableF.del();
        return pyTable;
    }

}
