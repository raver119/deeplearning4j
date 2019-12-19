package org.datavec.python;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import static org.bytedeco.cpython.global.python.*;

/**
 * Swift like python wrapper for J
 *
 * @author Fariz Rahman
 */

public class Python {
    public final static PythonObject builtins = importModule("builtins");
    private final static PythonObject globals = builtins.attr("globals");
    public final static PythonObject None = dict().attr("get").call(new PythonObject(""));
    public final static PythonObject True = bool(new PythonObject(1));
    public final static PythonObject False = bool(new PythonObject(0));

    public static PythonObject importModule(String moduleName){
        return new PythonObject(PyImport_ImportModule(moduleName));
    }
    public static PythonObject attr(String attrName){
        return builtins.attr(attrName);
    }
    public static PythonObject len(PythonObject pythonObject){
        return attr("len").call(pythonObject);
    }
    public static PythonObject  str(PythonObject pythonObject){
        return attr("str").call(pythonObject);
    }
    public static PythonObject  str(){
        return attr("str").call();
    }

    public static PythonObject strType(){
        return attr("str");
    }
    public static PythonObject float_(PythonObject pythonObject){
        return attr("float").call(pythonObject);
    }
    public static PythonObject float_(){
        return attr("float").call();
    }
    public static PythonObject floatType(){
        return attr("float");
    }
    public static  PythonObject bool(PythonObject pythonObject){
        return attr("bool").call(pythonObject);
    }
    public static  PythonObject bool(){
        return attr("bool").call();
    }
    public static PythonObject boolType(){
        return attr("bool");
    }
    public static PythonObject int_(PythonObject pythonObject){
        return attr("int").call(pythonObject);
    }
    public static PythonObject int_(){
        return attr("int").call();
    }
    public static PythonObject intType(){
        return attr("int");
    }
    public static PythonObject list(PythonObject pythonObject){
        return attr("list").call(pythonObject);
    }
    public static PythonObject list(){
        return attr("list").call();
    }
    public static PythonObject listType(){
        return attr("list");
    }
    public static PythonObject dict(PythonObject pythonObject){
        return attr("dict").call(pythonObject);
    }
    public static PythonObject dict(){
        return attr("dict").call();
    }
    public static PythonObject dictType(){
        return attr("dict");
    }
    public static PythonObject set(PythonObject pythonObject){
        return attr("set").call(pythonObject);
    }
    public static PythonObject set(){
        return attr("set").call();
    }
    public static PythonObject tuple(PythonObject pythonObject){
        return attr("tuple").call(pythonObject);
    }
    public static PythonObject tuple(){
        return attr("tuple").call();
    }

    public static PythonObject globals(){
        return globals.call();
    }

    public static PythonObject type(PythonObject obj){
        return attr("type").call(obj);
    }

    public static PythonObject isinstance(PythonObject obj, PythonObject... type){
        return attr("isinstance").call(obj, type);
    }

    public static PythonObject eval(PythonObject code){
        return attr("eval").call(code);
    }

    public static PythonObject eval(String code){
        return eval(new PythonObject(code));
    }
}
