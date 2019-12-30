package org.datavec.python;

import org.bytedeco.cpython.PyObject;

import java.util.*;

import static org.bytedeco.cpython.global.python.*;

public class PythonMemoryManager {

    private Map<PyObject, Set<PythonObject>> objectMap = new HashMap<>();
    private PythonSession currentSession = new PythonSession();
    private List<PyObject> gcQueue = new ArrayList<>();
    private static PythonMemoryManager instance = new PythonMemoryManager();
    private Set<PyObject> borrowedReferences = new HashSet<>();

    private PythonMemoryManager(){
        // only single instance allowed!
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
                PythonMemoryManager.deleteObject(obj);
            }
            PythonMemoryManager.collect();
            PythonMemoryManager.instance.currentSession = parentSession;
            System.out.println("-----close----");
        }

        private PythonSession forkSession(){
            System.out.println("----open----");
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

    public static void markAsBorrowedReference(PythonObject pythonObject){
        instance.borrowedReferences.add(pythonObject.getNativePythonObject());
    }
    public  static void  registerObject(PythonObject pythonObject){
        PyObject nativeObject = pythonObject.getNativePythonObject();
        if (!instance.objectMap.containsKey(nativeObject)){
            instance.objectMap.put(nativeObject, new HashSet<PythonObject>());
        }
        instance.objectMap.get(nativeObject).add(pythonObject);

        instance.currentSession.registerObject(pythonObject);
    }

    public static void deleteObject(PythonObject pythonObject){
        PyObject nativeObject = pythonObject.getNativePythonObject();
        Set<PythonObject> pythonObjects = instance.objectMap.get(nativeObject);
        pythonObjects.remove(pythonObject);
        if (pythonObjects.size() == 0){
            instance.gcQueue.add(nativeObject);
        }
    }

    public static void collect(){
        for(PyObject pyObject: instance.gcQueue){
            if (!instance.borrowedReferences.contains(pyObject)){
                PyObject repr = PyObject_Str(pyObject);
                PyObject str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
                String jstr = PyBytes_AsString(str).getString();
                Py_DecRef(repr);
                Py_DecRef(str);
                System.out.println(jstr);
                if (pyObject.ob_refcnt() > 0)
                Py_DecRef(pyObject);

            }
        }
        instance.gcQueue.clear();
    }

    public static PythonSession getSession(){
        PythonSession sess = instance.currentSession.forkSession();
        instance.currentSession = sess;
        return sess;
    }

    public static void clearAllObjects(){
        while(instance.currentSession != null){
            instance.currentSession.close();
        }
        collect();
    }

}
