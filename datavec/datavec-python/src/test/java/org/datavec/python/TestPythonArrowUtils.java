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

import org.bytedeco.arrow.Field;
import org.bytedeco.arrow.FieldVector;
import org.bytedeco.arrow.Schema;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.bytedeco.arrow.global.arrow.*;

public class TestPythonArrowUtils {

    @Test
    public void testPyArrowImport() throws Exception{
        PythonArrowUtils.importPyArraow();
    }

    @Test
    public void testINDArrayConversion() throws PythonException{
        INDArray array = Nd4j.rand(10);
        PythonObject pyArrowArray = PythonArrowUtils.getPyArrowArrayFromINDArray(array);
        INDArray array2 = PythonArrowUtils.getINDArrayFromPyArrowArray(pyArrowArray);
        Assert.assertEquals(array, array2);

        // test no copy
        Assert.assertEquals(array.data().address(), array2.data().address());
        array.putScalar(0, Nd4j.rand(1).getDouble(0));
        Assert.assertEquals(array, array2);
    }

    @Test
    public void testFieldConversion() throws PythonException{
        Field[] fields = new Field[]{
                new Field("a", int8()),
                new Field("b", int16()),
                new Field("c", int32()),
                new Field("d", int64()),
                new Field("e", uint8()),
                new Field("f", uint16()),
                new Field("g", uint32()),
                new Field("h", uint64()),
                new Field("i", _boolean()),
                new Field("j", float16()),
                new Field("k", float32()),
                new Field("l", float64()),
        };
        for (Field field: fields){
            PythonObject pyArrowField = PythonArrowUtils.getPyArrowField(field);
            Field field2 = PythonArrowUtils.getFieldFromPythonObject(pyArrowField);
            Assert.assertEquals(field.name(), field2.name());
            Assert.assertEquals(field.type(), field.type());
        }
    }

    @Test
    public void testSchemaConversion() throws PythonException{
        Field[] fields = new Field[]{
                new Field("a", int8()),
                new Field("b", int16()),
                new Field("c", int32()),
                new Field("d", int64()),
                new Field("e", uint8()),
                new Field("f", uint16()),
                new Field("g", uint32()),
                new Field("h", uint64()),
                new Field("i", _boolean()),
                new Field("j", float16()),
                new Field("k", float32()),
                new Field("l", float64()),
        };
        Schema schema = new Schema(new FieldVector(fields));
        PythonObject pySchema = PythonArrowUtils.getPyArrowSchema(schema);
        Schema schema2 = PythonArrowUtils.getSchemaFromPythonObject(pySchema);
        Field[] fields2 = schema2.fields().get();
        Assert.assertEquals(fields.length, fields2.length);
        for(int i = 0;i < fields.length; i++){
            Assert.assertEquals(fields[i].name(), fields2[i].name());
            Assert.assertEquals(fields[i].type(), fields2[i].type());
        }
    }

}
