package org.datavec.python;

import java.util.HashMap;
import java.util.Map;

public class PythonJob {

    private String code;
    private String name;
    PythonObject runF;
    public PythonJob(String name, String code){
        this.name = name;
        this.code = code;
        setup();
    }

    public void clearState(){
        PythonContextManager.setContext("main");
        PythonContextManager.deleteContext(name);
        setup();
    }

    public void setup(){
        try (FastPythonExecutioner.GIL gil = FastPythonExecutioner.lock()){
            PythonContextManager.setContext(name);
            PythonObject runF = FastPythonExecutioner.getVariable("run");
            if (runF.isNone()){
                FastPythonExecutioner.exec(code);
            }
            runF = FastPythonExecutioner.getVariable("run");
            if (runF.isNone()){
                throw new RuntimeException("run() method not found!");
            }
            this.runF = runF;
            PythonObject setupF = FastPythonExecutioner.getVariable("setup");
            if (!setupF.isNone()){
                setupF.call();
            }
        }
    }
    public void exec(PythonVariables inputs, PythonVariables outputs){
        try(FastPythonExecutioner.GIL gil = FastPythonExecutioner.lock()){
            if(inputs != null)
            FastPythonExecutioner.setVariables(inputs);
            PythonObject inspect = Python.importModule("inspect");
            PythonObject argsList = inspect.attr("getfullargspec").call(runF).attr("args");
            PythonObject runargs = Python.dict();
            int argsCount = Python.len(argsList).toInt();
            for(int i=0; i <argsCount; i++){
                PythonObject arg = argsList.get(i);
                PythonObject val = Python.globals().get(arg);
                if(val.isNone()){
                    throw new RuntimeException("Input value not received for run() argument: " + arg.toString());
                }
                runargs.set(arg, val);
            }
            PythonObject outDict = runF.callWithKargs(runargs);
            Python.globals().attr("update").call(outDict);
            FastPythonExecutioner.getVariables(outputs);
        }
    }

    public PythonVariables execAndReturnAllVariables(PythonVariables inputs){
        try(FastPythonExecutioner.GIL gil = FastPythonExecutioner.lock()){
            if(inputs != null)
                FastPythonExecutioner.setVariables(inputs);
            PythonObject inspect = Python.importModule("inspect");
            PythonObject argsList = inspect.attr("getfullargspec").call(runF).attr("args");
            PythonObject runargs = Python.dict();
            int argsCount = Python.len(argsList).toInt();
            for(int i=0; i <argsCount; i++){
                PythonObject arg = argsList.get(i);
                PythonObject val = Python.globals().get(arg);
                if(val.isNone()){
                    throw new RuntimeException("Input value not received for run() argument: " + arg.toString());
                }
                runargs.set(arg, val);
            }
            PythonObject outDict = runF.callWithKargs(runargs);
            Python.globals().attr("update").call(outDict);
            return FastPythonExecutioner.getAllVariables();
        }
    }


}
