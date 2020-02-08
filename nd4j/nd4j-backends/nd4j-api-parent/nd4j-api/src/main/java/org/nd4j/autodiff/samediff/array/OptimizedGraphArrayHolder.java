package org.nd4j.autodiff.samediff.array;

import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.function.Supplier;

import java.util.*;

public class OptimizedGraphArrayHolder implements ArrayHolder {

    private final ArrayHolder underlyingHolder;
    private final Map<String, Supplier<INDArray>> functions;

    public OptimizedGraphArrayHolder(ArrayHolder underlyingHolder){
        this.underlyingHolder = underlyingHolder;
        this.functions = new HashMap<>();
    }

    public void setFunction(String name, Supplier<INDArray> fn){
        if(underlyingHolder.hasArray(name))
            underlyingHolder.removeArray(name);
        functions.put(name, fn);
    }

    @Override
    public boolean hasArray(String name) {
        return functions.containsKey(name) || underlyingHolder.hasArray(name);
    }

    @Override
    public INDArray getArray(String name) {
        if(functions.containsKey(name))
            return functions.get(name).get();
        return underlyingHolder.getArray(name);
    }

    @Override
    public void setArray(String name, INDArray array) {
        Preconditions.checkState(!functions.containsKey(name), "Cannot set array when existing array is only accessible via a function");
        underlyingHolder.setArray(name, array);
    }

    @Override
    public INDArray removeArray(String name) {
        Supplier<INDArray> s = functions.remove(name);
        if(s != null)
            return s.get();
        return underlyingHolder.removeArray(name);
    }

    @Override
    public int size() {
        return underlyingHolder.size() + functions.size();
    }

    @Override
    public void initFrom(ArrayHolder arrayHolder) {
        underlyingHolder.initFrom(arrayHolder);
    }

    @Override
    public Collection<String> arrayNames() {
        Set<String> set = new HashSet<>();
        set.addAll(underlyingHolder.arrayNames());
        set.addAll(functions.keySet());
        return set;
    }

    @Override
    public void rename(String from, String to) {
        if(functions.containsKey(from)) {
            functions.put(to, functions.remove(from));
        } else {
            underlyingHolder.rename(from, to);
        }
    }
}
