package org.datavec.python;

import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.HashMap;
import java.util.Map;


@Data
@NoArgsConstructor
public class PythonJob {

    private String code;
    private String name;
    private String context;
    private boolean setupRunMode;
    PythonObject runF;

    static {
        new PythonExecutioner();
    }

    public PythonJob(String name, String code, boolean setupRunMode) {
        this.name = name;
        this.code = code;
        this.setupRunMode = setupRunMode;
        context = "__job_" + name;
        if (PythonContextManager.hasContext(context)) {
            throw new RuntimeException("Unable to create python job " + name + ". Context " + context + " already exists!");
        }
        if (setupRunMode) setup();
    }

    public void clearState() {
        PythonContextManager.setContext("main");
        PythonContextManager.deleteContext(context);
        setup();
    }

    public void setup() {
        try (PythonGIL gil = PythonGIL.lock()) {
            PythonContextManager.setContext(context);
            PythonObject runF = PythonExecutioner.getVariable("run");
            if (runF.isNone() || !Python.callable(runF)) {
                PythonExecutioner.exec(code);
            }
            runF = PythonExecutioner.getVariable("run");
            if (runF.isNone() || !Python.callable(runF)) {
                throw new RuntimeException("run() method not found!");
            }
            this.runF = runF;
            PythonObject setupF = PythonExecutioner.getVariable("setup");
            if (!setupF.isNone()) {
                setupF.call();
            }
        }
    }

    public void exec(PythonVariables inputs, PythonVariables outputs) {
        try (PythonGIL gil = PythonGIL.lock()) {
            PythonContextManager.setContext(context);
            if (!setupRunMode) {
                PythonExecutioner.exec(code, inputs, outputs);
                return;
            }
            PythonExecutioner.setVariables(inputs);

            PythonObject inspect = Python.importModule("inspect");
            PythonObject getfullargspec = inspect.attr("getfullargspec");
            PythonObject argspec = getfullargspec.call(runF);
            PythonObject argsList = argspec.attr("args");
            PythonObject runargs = Python.dict();
            int argsCount = Python.len(argsList).toInt();
            for (int i = 0; i < argsCount; i++) {
                PythonObject arg = argsList.get(i);
                PythonObject val = Python.globals().get(arg);
                if (val.isNone()) {
                    throw new RuntimeException("Input value not received for run() argument: " + arg.toString());
                }
                runargs.set(arg, val);
            }
            PythonObject outDict = runF.callWithKwargs(runargs);
            Python.globals().attr("update").call(outDict);

            PythonExecutioner.getVariables(outputs);
        }
    }

    public PythonVariables execAndReturnAllVariables(PythonVariables inputs) {
        try (PythonGIL gil = PythonGIL.lock()) {
            if (!setupRunMode) {
                return PythonExecutioner.execAndReturnAllVariables(code, inputs);
            }
            PythonExecutioner.setVariables(inputs);
            PythonObject inspect = Python.importModule("inspect");
            PythonObject argsList = inspect.attr("getfullargspec").call(runF).attr("args");
            PythonObject runargs = Python.dict();
            int argsCount = Python.len(argsList).toInt();
            for (int i = 0; i < argsCount; i++) {
                PythonObject arg = argsList.get(i);
                PythonObject val = Python.globals().get(arg);
                if (val.isNone()) {
                    throw new RuntimeException("Input value not received for run() argument: " + arg.toString());
                }
                runargs.set(arg, val);
            }
            PythonObject outDict = runF.callWithKwargs(runargs);
            Python.globals().attr("update").call(outDict);
            return PythonExecutioner.getAllVariables();
        }
    }


}
