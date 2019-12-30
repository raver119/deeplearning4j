package org.datavec.python;

import org.bytedeco.cpython.PyObject;

import java.util.*;

import static org.bytedeco.cpython.global.python.Py_DecRef;

public class PythonMemoryManager {

    private Map<PyObject, Set<PythonObject>> objectMap = new HashMap<>();
    private PythonSession currentSession = new PythonSession();
    private List<PyObject> gcQueue = new ArrayList<>();
    private static PythonMemoryManager instance = new PythonMemoryManager();
    private static Set<PyObject> borrowedReferences = new HashSet<>();

    private PythonMemoryManager(){
        // only single instance allowed!
    }

    public static PythonMemoryManager getInstance(){
        return instance;
    }

    public class PythonSession implements AutoCloseable{
        private PythonSession parentSession;

        private PythonSession(){

        }

        private Set<PythonObject> objects = new HashSet<>();


        private void registerObject(PythonObject pythonObject){
            objects.add(pythonObject);
        }
        @Override
        public void close(){
            for(PythonObject obj: objects){
                PythonMemoryManager.getInstance().deleteObject(obj);
            }
            PythonMemoryManager.getInstance().collect();
            PythonMemoryManager.getInstance().currentSession = parentSession;
        }

        public PythonSession forkSession(){
            PythonSession pythonSession = new PythonSession();
            pythonSession.parentSession = this;
            return pythonSession;
        }

        public void moveObjectToParentSession(PythonObject pythonObject){
            if (parentSession != null){
                objects.remove(pythonObject);
                parentSession.objects.add(pythonObject);
            }
        }

    }

    public void markAsBorrowedReference(PythonObject pythonObject){
        borrowedReferences.add(pythonObject.getNativePythonObject());
    }
    public  void registerObject(PythonObject pythonObject){
        PyObject nativeObject = pythonObject.getNativePythonObject();
        if (!objectMap.containsKey(nativeObject)){
            objectMap.put(nativeObject, new HashSet<PythonObject>());
        }
        objectMap.get(nativeObject).add(pythonObject);
        currentSession.registerObject(pythonObject);
    }

    public void deleteObject(PythonObject pythonObject){
        PyObject nativeObject = pythonObject.getNativePythonObject();
        Set<PythonObject> pythonObjects = objectMap.get(nativeObject);
        pythonObjects.remove(pythonObject);
        if (pythonObjects.size() == 0){
            gcQueue.add(nativeObject);
        }
    }

    public void collect(){
        for(PyObject pyObject: gcQueue){
            if (!borrowedReferences.contains(pyObject)){
                System.out.println("-----Py_DecRef----");
                Py_DecRef(pyObject);
            }
        }
        gcQueue.clear();
    }

    public static PythonSession getSession(){
        return instance.currentSession.forkSession();
    }

    public void clearAllObjects(){
        while(currentSession != null){
            currentSession.close();
        }
        collect();
    }

}
