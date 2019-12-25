package org.datavec.python;


import lombok.var;
import org.json.JSONArray;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

@javax.annotation.concurrent.NotThreadSafe
public class TestPythonList {

    @Test
    public void testPythonListFromIntArray(){
        PythonObject pyList = new PythonObject(new Integer[]{1, 2, 3, 4, 5});
        pyList.attr("append").call(6);
        pyList.attr("append").call(7);
        pyList.attr("append").call(8);
        assertEquals(8, Python.len(pyList).toInt());
        for (int i=0; i < 8; i++){
            assertEquals(i + 1, pyList.get(i).toInt());
        }

    }

    @Test
    public void testPythonListFromLongArray(){
        PythonObject pyList = new PythonObject(new Long[]{1L, 2L, 3L, 4L, 5L});
        pyList.attr("append").call(6);
        pyList.attr("append").call(7);
        pyList.attr("append").call(8);
        assertEquals(8, Python.len(pyList).toInt());
        for (int i=0; i < 8; i++){
            assertEquals(i + 1, pyList.get(i).toInt());
        }

    }

    @Test
    public void testPythonListFromDoubleArray(){
        PythonObject pyList = new PythonObject(new Double[]{1., 2., 3., 4., 5.});
        pyList.attr("append").call(6);
        pyList.attr("append").call(7);
        pyList.attr("append").call(8);
        assertEquals(8, Python.len(pyList).toInt());
        for (int i=0; i < 8; i++){
            assertEquals(i + 1, pyList.get(i).toInt());
            assertEquals((double) i + 1, pyList.get(i).toDouble(), 1e-5);
        }

    }
    @Test
    public void testPythonListFromStringArray(){
        PythonObject pyList = new PythonObject(new String[]{"abcd", "efg"});
        pyList.attr("append").call("hijk");
        pyList.attr("append").call("lmnop");
        assertEquals("abcdefghijklmnop", new PythonObject("").attr("join").call(pyList).toString());
    }

    @Test
    public void testPythonListFromMixedArray() {
        Map<Object, Object> map = new HashMap<>();
        map.put(1, "a");
        map.put("a", Arrays.asList("a", "b", "c"));
        Object[] objs = new Object[]{
                1, 2, "a", 3f, 4L, 5.0, Arrays.asList(10,
                20, "b", 30f, 40L, 50.0, map

        ), map
        };
        PythonObject pyList = new PythonObject(objs);
        String expectedStr = "[1, 2, 'a', 3.0, 4, 5.0, [10, 20, 'b', 30.0, 40, 50.0, {1: 'a', 'a': ['a', 'b', 'c']}], {1: 'a', 'a': ['a', 'b', 'c']}]";
        assertEquals(expectedStr, pyList.toString());
    }

}
