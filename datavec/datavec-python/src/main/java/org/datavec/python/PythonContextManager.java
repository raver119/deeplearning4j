
/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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


package org.datavec.python;


import java.util.HashSet;
import java.util.Set;

/**
 * Emulates multiples interpreters in a single interpreter
 *
 * @author Fariz Rahman
 */


public class PythonContextManager {

    private static Set<String> contexts = new HashSet<>();
    private static boolean init = false;
    private static String currentContext;

    static {init();}
    private static void init(){
        if (init) return;
        new PythonExecutioner();
        init = true;
        currentContext = "main";
        contexts.add(currentContext);
    }

    public static void addContext(String contextName){
        if(!_validateContextName(contextName)){
            throw new RuntimeException("Invalid context name: " + contextName);
        }
        contexts.add(contextName);
    }

    public static boolean hasContext(String contextName){
        return contexts.contains(contextName);
    }


    public static boolean _validateContextName(String s) {
        if(s.length()==0)   return false;
        if(!Character.isJavaIdentifierStart(s.charAt(0)))   return false;
        for (int i = 1; i < s.length(); i++)
            if (!Character.isJavaIdentifierPart(s.charAt(i)))
                return false;
        return true;
    }

    private static String _getContextPrefix(String contextName){
        return "__collapsed__" + contextName + "__";
    }
    private static String _getCollapsedVarNameForContext(String varName, String contextName){
        return _getContextPrefix(contextName) + varName;
    }

    private static String _expandCollapsedVarName(String varName, String contextName){
        String prefix = "__collapsed__" + contextName + "__";
        if (!(varName.startsWith(prefix))){
            throw new RuntimeException("Invalid variable name.");
        }
        return varName.substring(prefix.length());

    }
    private static void _collapseContext(String contextName){
        PythonObject globals = Python.globals();
        PythonObject keysList = Python.list(globals.attr("keys").call());
        int numKeys = Python.len(keysList).toInt();
        for(int i=0;i<numKeys;i++){
            PythonObject key = keysList.get(i);
            String keyStr = key.toString();
            if (!((keyStr.startsWith("__") && keyStr.endsWith("__"))|| keyStr.startsWith("__collapsed_"))){
                String collapsedKey = _getCollapsedVarNameForContext(keyStr, contextName);
                PythonObject val = globals.attr("pop").call(key);
                globals.set(new PythonObject(collapsedKey), val);
            }
        }
    }
    private static void _expandContext(String contextName){
        String prefix = _getContextPrefix(contextName);
        PythonObject globals = Python.globals();
        PythonObject keysList = Python.list(globals.attr("keys").call());
        int numKeys = Python.len(keysList).toInt();
        for(int i=0;i<numKeys;i++){
            PythonObject key = keysList.get(i);
            String keyStr = key.toString();
            if (keyStr.startsWith(prefix)){
                String expandedKey = _expandCollapsedVarName(keyStr, contextName);
                PythonObject val = globals.attr("pop").call(key);
                globals.set(new PythonObject(expandedKey), val);
            }
        }

    }

    public static void setContext(String contextName){
        if (contextName == currentContext){
            return;
        }
        if (!hasContext(contextName)){
            addContext(contextName);
        }
        _collapseContext(currentContext);
        _expandContext(contextName);
        currentContext = contextName;

    }

    public static String getCurrentContext(){
        return currentContext;
    }

    public static void deleteContext(String contextName){
        if(contextName.equals("main")){
            throw new RuntimeException("Can not delete main context!");
        }
        if (contextName.equals(currentContext)){
            throw new RuntimeException("Can not delete current context!");
        }
        String prefix = _getContextPrefix(contextName);
        PythonObject globals = Python.globals();
        PythonObject keysList = Python.list(globals.attr("keys").call());
        int numKeys = Python.len(keysList).toInt();
        for(int i=0;i<numKeys;i++){
            PythonObject key = keysList.get(i);
            String keyStr = key.toString();
            if (keyStr.startsWith(prefix)){
                globals.attr("__delitem__").call(key);
            }
        }
        contexts.remove(contextName);
    }

    public static void deleteNonMainContexts(){
        setContext("main");
        for(String c: contexts.toArray(new String[0])){
            if (!c.equals("main")){
                deleteContext(c);
            }
        }
    }

    public String[] getContexts(){
        return contexts.toArray(new String[0]);
    }

}
