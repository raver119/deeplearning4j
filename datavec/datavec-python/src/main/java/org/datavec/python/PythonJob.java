package org.datavec.python;

import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.annotation.Nonnull;
import java.util.HashMap;
import java.util.Map;


@Data
@NoArgsConstructor
/**
 * PythonJob is the right abstraction for executing multiple python scripts
 * in a multi thread stateful environment. The setup-and-run mode allows your
 * "setup" code (imports, model loading etc) to be executed only once.
 */
public class PythonJob {

    private String code;
    private String name;
    private String context;
    private boolean setupRunMode;
    private PythonObject runF;

    static {
        new PythonExecutioner();
    }

    @Builder
    public PythonJob(@Nonnull String name, @Nonnull String code, boolean setupRunMode) throws Exception {
        this.name = name;
        this.code = code;
        this.setupRunMode = setupRunMode;
        context = "__job_" + name;
        if (PythonContextManager.hasContext(context)) {
            throw new Exception("Unable to create python job " + name + ". Context " + context + " already exists!");
        }
        if (setupRunMode) setup();
    }


    /**
     * Clears all variables in current context and calls setup()
     */
    public void clearState() throws Exception {
        String context = this.context;
        PythonContextManager.setContext("main");
        PythonContextManager.deleteContext(context);
        this.context = context;
        setup();
    }

    public void setup() throws Exception {
        try (PythonGIL gil = PythonGIL.lock()) {
            PythonContextManager.setContext(context);
            PythonObject runF = PythonExecutioner.getVariable("run");
            if (runF.isNone() || !Python.callable(runF)) {
                PythonExecutioner.exec(code);
                runF = PythonExecutioner.getVariable("run");
            }
            if (runF.isNone() || !Python.callable(runF)) {
                throw new Exception("run() method not found!");
            }
            this.runF = runF;
            PythonObject setupF = PythonExecutioner.getVariable("setup");
            if (!setupF.isNone()) {
                setupF.call();
            }
        }
    }

    public void exec(PythonVariables inputs, PythonVariables outputs) throws Exception {
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
                    throw new Exception("Input value not received for run() argument: " + arg.toString());
                }
                runargs.set(arg, val);
            }
            PythonObject outDict = runF.callWithKwargs(runargs);
            Python.globals().attr("update").call(outDict);

            PythonExecutioner.getVariables(outputs);
            inspect.del();
            getfullargspec.del();
            argspec.del();
            runargs.del();
        }
    }

    public PythonVariables execAndReturnAllVariables(PythonVariables inputs) throws Exception {
        try (PythonGIL gil = PythonGIL.lock()) {
            if (!setupRunMode) {
                return PythonExecutioner.execAndReturnAllVariables(code, inputs);
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
                    throw new Exception("Input value not received for run() argument: " + arg.toString());
                }
                runargs.set(arg, val);
            }
            PythonObject outDict = runF.callWithKwargs(runargs);
            Python.globals().attr("update").call(outDict);
            inspect.del();
            getfullargspec.del();
            argspec.del();
            runargs.del();
            return PythonExecutioner.getAllVariables();
        }
    }


}
