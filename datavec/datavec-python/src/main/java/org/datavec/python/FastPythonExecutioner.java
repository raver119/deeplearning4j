package org.datavec.python;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.bytedeco.cpython.PyThreadState;
import org.bytedeco.numpy.global.numpy;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.InputStream;
import java.util.Arrays;
import java.util.Map;

import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.cpython.global.python.PyThreadState_Get;

@Slf4j
public class FastPythonExecutioner {

    private static boolean init;
    private static PyThreadState mainThreadState;

    static{init();}
    private static synchronized  void init() {
        if(init) {
            return;
        }
        init = true;

        log.info("CPython: PyEval_InitThreads()");
        PyEval_InitThreads();
        log.info("CPython: Py_InitializeEx()");
        Py_InitializeEx(0);
        log.info("CPython: PyThreadState_Get()");
        mainThreadState = PyThreadState_Get();
        numpy._import_array();
    }

    private static synchronized void _exec(String code) {
        log.debug(code);
        log.info("CPython: PyRun_SimpleStringFlag()");

        int result = PyRun_SimpleStringFlags(code, null);
        if (result != 0) {
            log.info("CPython: PyErr_Print");
            PyErr_Print();
            throw new RuntimeException("exec failed");
        }
    }

    public static void setVariable(String varName, PythonVariables.Type varType, Object value){
        PythonObject pythonObject;
        switch(varType){
            case STR:
                pythonObject = new PythonObject((String)value);
                break;
            case INT:
                pythonObject = new PythonObject(((Number)value).longValue());
                break;
            case FLOAT:
                pythonObject = new PythonObject(((Number)value).floatValue());
                break;
            case BOOL:
                pythonObject = new PythonObject((boolean)value);
                break;
            case NDARRAY:
                if (value instanceof NumpyArray){
                    pythonObject = new PythonObject((NumpyArray) value);
                }
                else if (value instanceof INDArray){
                    pythonObject = new PythonObject((INDArray) value);
                }
                else{
                    throw new RuntimeException("Invalid value for type NDARRAY");
                }
                break;
            case LIST:
                pythonObject = new PythonObject(Arrays.asList((Object[])value));
                break;
            case DICT:
                pythonObject = new PythonObject((Map)value);
                break;
            default:
                throw new RuntimeException("Unsupported type: " + varType);

        }
        Python.globals().call().set(new PythonObject(varName), pythonObject);
    }

    public static void setVariables(PythonVariables pyVars){
        for (String varName: pyVars.getVariables()){
            setVariable(varName, pyVars.getType(varName), pyVars.getValue(varName));
        }
    }

    public static Object getVariable(String varName, PythonVariables.Type varType){
        PythonObject pythonObject = Python.globals().call().get(varName);
        switch (varType){
            case INT:
                return pythonObject.toLong();
            case FLOAT:
                return pythonObject.toDouble();
            case NDARRAY:
                return pythonObject.toNumpy();
            case STR:
                return pythonObject.toString();
            case BOOL:
                return pythonObject.toBoolean();
                default:
                    throw new RuntimeException("Unsupported type: " + varType);
        }
    }
    public static void getVariables(PythonVariables pyVars){
        for (String varName: pyVars.getVariables()){
            pyVars.setValue(varName, getVariable(varName, pyVars.getType(varName)));
        }
    }
    

}
