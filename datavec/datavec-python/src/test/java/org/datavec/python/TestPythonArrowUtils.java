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
import org.bytedeco.arrow.Table;
import org.datavec.api.transform.ColumnType;
import org.datavec.arrow.table.DataVecTable;
import org.datavec.arrow.table.column.DataVecColumn;
import org.datavec.arrow.table.column.impl.*;
import org.datavec.arrow.table.row.Row;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.bytedeco.arrow.global.arrow.*;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

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
                new Field("m", binary()),
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
                new Field("m", binary()),
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

    @Test
    public void testTableConversion() throws PythonException{
        new DoubleColumn("double", new Double[]{1.0});

    }

    @Test
    public void testTables()throws Exception{
        PythonArrowUtils.init();
        Map<String, INDArray> map = new HashMap<>();
        map.put("a", Nd4j.zeros(5));
        map.put("b", Nd4j.ones(5));
        map.put("c", Nd4j.rand(5));

        PythonObject d = new PythonObject(map);

        Table table = PythonArrowUtils.getTableFromPythonObject(d);
    }

  @Test
    public void testTable() throws Exception{
        PythonArrowUtils.init();
      ColumnType[] columnTypes = new ColumnType[] {
              ColumnType.Integer,
              ColumnType.Double,
              ColumnType.Float,
              ColumnType.Boolean,
              ColumnType.String
      };

      DataVecColumn[] dataVecColumns = new DataVecColumn[columnTypes.length];
      for (int i = 0; i < columnTypes.length; i++){
          ColumnType columnType = columnTypes[i];
          switch(columnType) {
              case Double:
                  dataVecColumns[i] = new DoubleColumn(columnType.name().toLowerCase(),new Double[]{1.0});
                  break;
              case Float:
                  dataVecColumns[i] = new FloatColumn(columnType.name().toLowerCase(),new Float[]{1.0f});
                  break;
              case Boolean:
                  dataVecColumns[i] = new BooleanColumn(columnType.name().toLowerCase(),new Boolean[]{true});
                  break;
              case String:
                  dataVecColumns[i] = new StringColumn(columnType.name().toLowerCase(),new String[]{"1.0"});
                  break;
              case Long:
                  dataVecColumns[i] = new LongColumn(columnType.name().toLowerCase(),new Long[]{1L});
                  break;
              case Integer:
                  dataVecColumns[i] = new IntColumn(columnType.name().toUpperCase(),new Integer[]{1});
                  break;

          }
      }
       DataVecTable dataVecTable = DataVecTable.create(dataVecColumns);
      PythonObject pyArrowTable = PythonArrowUtils.getPyArrowTable(dataVecTable);


  }

}
