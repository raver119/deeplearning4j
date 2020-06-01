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


import org.eclipse.python4j.PythonException;
import org.eclipse.python4j.PythonObject;
import org.eclipse.python4j.PythonTypes;
import org.junit.Assert;
import org.junit.Test;

public class PythonPrimitiveTypesTest {

    @Test
    public void testInt() throws PythonException {
        long j = 3;
        PythonObject p = PythonTypes.INT.toPython(j);
        long j2 = PythonTypes.INT.toJava(p);

        Assert.assertEquals(j, j2);

        PythonObject p2 = PythonTypes.convert(j);
        long j3 = PythonTypes.INT.toJava(p2);

        Assert.assertEquals(j, j3);
    }

    @Test
    public void testStr() throws PythonException{
        String s = "abcd";
        PythonObject p = PythonTypes.STR.toPython(s);
        String s2 = PythonTypes.STR.toJava(p);

        Assert.assertEquals(s, s2);

        PythonObject p2 = PythonTypes.convert(s);
        String s3 = PythonTypes.STR.toJava(p2);

        Assert.assertEquals(s, s3);
    }

    @Test
    public void testFloat() throws PythonException{
        double f = 7;
        PythonObject p = PythonTypes.FLOAT.toPython(f);
        double f2 = PythonTypes.FLOAT.toJava(p);

        Assert.assertEquals(f, f2, 1e-5);

        PythonObject p2 = PythonTypes.convert(f);
        double f3 = PythonTypes.FLOAT.toJava(p2);

        Assert.assertEquals(f, f3, 1e-5);
    }

    @Test
    public void testBool() throws PythonException{
        boolean b = true;
        PythonObject p = PythonTypes.BOOL.toPython(b);
        boolean b2 = PythonTypes.BOOL.toJava(p);

        Assert.assertEquals(b, b2);

        PythonObject p2 = PythonTypes.convert(b);
        boolean b3 = PythonTypes.BOOL.toJava(p2);

        Assert.assertEquals(b, b3);
    }

}
