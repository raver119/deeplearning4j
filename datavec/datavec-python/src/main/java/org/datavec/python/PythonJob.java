package org.datavec.python;

public class PythonJob {

    private String code;
    private String name;
    PythonObject runF;
    public PythonJob(String name, String code){
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
        
    }




}
