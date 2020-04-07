/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.autodiff.samediff.ops;

import org.nd4j.base.Preconditions;
import org.nd4j.shade.guava.collect.Sets;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import lombok.NonNull;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.samediff.ArgumentInterceptor;
import org.nd4j.autodiff.samediff.NameScope;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SameDiffLambda;
import org.nd4j.autodiff.samediff.SameDiffNoArgSingleLambda;
import org.nd4j.autodiff.samediff.SameDiffSingleLambda;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Merge;
import org.nd4j.linalg.api.ops.impl.shape.OneHot;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.GradientBackwardsMarker;
import org.nd4j.linalg.indexing.conditions.Condition;

import static org.nd4j.autodiff.samediff.ops.SDValidation.*;

/**
 * Core op creator methods available via SameDiff class directly
 *
 * @author Alex Black
 * @see SDMath SDMath for Math operations
 * @see SDRandom SDRandom for random number generator operations
 * @see SDNN SDNN for general neural network operations
 * @see SDCNN SDCNN for Convolutional Neural Network operations
 * @see SDRNN SDRNN for Recurrent Neural Network operations
 * @see SDLoss SDLoss for loss function operations
 */
public abstract class SDBaseOps {

    /**
     * Intended for internal/developer use
     */
    protected SDVariable gradientBackwardsMarker(SDVariable x) {
        return gradientBackwardsMarker(generateNewVarName(new GradientBackwardsMarker().opName(), 0), x);
    }

    /**
     * Intended for internal/developer use
     */
    protected SDVariable gradientBackwardsMarker(String name, SDVariable x) {
        SDVariable result = f().gradientBackwardsMarker(x);
        return updateVariableNameAndReference(result, name);
    }

    protected abstract String generateNewVarName(String baseName, int argIndex);

    protected abstract DifferentialFunctionFactory f();

    protected abstract SDVariable updateVariableNameAndReference(SDVariable varToUpdate, String newVarName);

    protected abstract SameDiff sd();

    /**
     * Matrix multiply a batch of matrices. matricesA and matricesB have to be arrays of same
     * length and each pair taken from these sets has to have dimensions (M, N) and (N, K),
     * respectively. If transposeA is true, matrices from matricesA will have shape (N, M) instead.
     * Likewise, if transposeB is true, matrices from matricesB will have shape (K, N).
     * <p>
     * <p>
     * The result of this operation will be a batch of multiplied matrices. The
     * result has the same length as both input batches and each output matrix is of shape (M, K).
     *
     * @param matricesA  First array of input matrices, all of shape (M, N) or (N, M)
     * @param matricesB  Second array of input matrices, all of shape (N, K) or (K, N)
     * @param transposeA whether first batch of matrices is transposed.
     * @param transposeB whether second batch of matrices is transposed.
     * @return Array of multiplied SDVariables of shape (M, K)
     */
    public SDVariable[] batchMmul(SDVariable[] matricesA, SDVariable[] matricesB,
                                  boolean transposeA, boolean transposeB) {
        return batchMmul(null, matricesA, matricesB, transposeA, transposeB);
    }

    /**
     * Matrix multiply a batch of matrices. matricesA and matricesB have to be arrays of same
     * length and each pair taken from these sets has to have dimensions (M, N) and (N, K),
     * respectively. If transposeA is true, matrices from matricesA will have shape (N, M) instead.
     * Likewise, if transposeB is true, matrices from matricesB will have shape (K, N).
     * <p>
     * <p>
     * The result of this operation will be a batch of multiplied matrices. The
     * result has the same length as both input batches and each output matrix is of shape (M, K).
     *
     * @param matricesA  First array of input matrices, all of shape (M, N) or (N, M)
     * @param matricesB  Second array of input matrices, all of shape (N, K) or (K, N)
     * @param transposeA whether first batch of matrices is transposed.
     * @param transposeB whether second batch of matrices is transposed.
     * @param names      names for all provided SDVariables
     * @return Array of multiplied SDVariables of shape (M, K)
     */
    public SDVariable[] batchMmul(String[] names, SDVariable[] matricesA, SDVariable[] matricesB,
                                  boolean transposeA, boolean transposeB) {
        validateSameType("batchMmul", true, matricesA);
        validateSameType("batchMmul", true, matricesB);
        SDVariable[] result = f().batchMmul(matricesA, matricesB, transposeA, transposeB);
        return updateVariableNamesAndReferences(result, names);
    }

    protected abstract SDVariable[] updateVariableNamesAndReferences(SDVariable[] variablesToUpdate, String[] newVariableNames);

    /**
     * Matrix multiply a batch of matrices. matricesA and matricesB have to be arrays of same
     * length and each pair taken from these sets has to have dimensions (M, N) and (N, K),
     * respectively. The result of this operation will be a batch of multiplied matrices. The
     * result has the same length as both input batches and each output matrix is of shape (M, K).
     *
     * @param matricesA First array of input matrices, all of shape (M, N)
     * @param matricesB Second array of input matrices, all of shape (N, K)
     * @return Array of multiplied SDVariables of shape (M, K)
     */
    public SDVariable[] batchMmul(SDVariable[] matricesA, SDVariable[] matricesB) {
        return batchMmul(null, matricesA, matricesB, false, false);
    }

    /**
     * @see #concat(String, int, SDVariable...)
     */
    public SDVariable concat(int dimension, SDVariable... inputs) {
        return concat(null, dimension, inputs);
    }

    /**
     * Concatenate a set of inputs along the specified dimension.<br>
     * Note that inputs must have identical rank and identical dimensions, other than the dimension to stack on.<br>
     * For example, if 2 inputs have shape [a, x, c] and [a, y, c] and dimension = 1, then the output has shape [a, x+y, c]
     *
     * @param name      Name of the output variable
     * @param dimension Dimension to concatenate on
     * @param inputs    Input variables
     * @return Output variable
     * @see #stack(String, int, SDVariable...)
     */
    public SDVariable concat(String name, int dimension, SDVariable... inputs) {
        validateSameType("concat", false, inputs);
        SDVariable result = f().concat(dimension, inputs);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #dynamicPartition(String[], SDVariable, SDVariable, int)
     */
    public SDVariable[] dynamicPartition(SDVariable x, SDVariable partitions, int numPartitions) {
        return dynamicPartition(null, x, partitions, numPartitions);
    }

    /**
     * Dynamically partition the input variable values into the specified number of paritions, using the indices.<br>
     * Example:<br>
     * <pre>
     * {@code input = [1,2,3,4,5]
     * numPartitions = 2
     * partitions = [1,0,0,1,0]
     * out[0] = [2,3,5]
     * out[1] = [1,4] }
     * </pre>
     *
     * @param name          Names for the output variables. Length must be equal to numPartitions
     * @param x             Input variable
     * @param partitions    1D input with values 0 to numPartitions-1
     * @param numPartitions Number of partitions, >= 1
     * @return Output variables (equal in number to numPartitions)
     */
    public SDVariable[] dynamicPartition(String[] name, SDVariable x, SDVariable partitions, int numPartitions) {
        SDVariable[] ret = f().dynamicPartition(x, partitions, numPartitions);
        return updateVariableNamesAndReferences(ret, name);
    }

    /**
     * TODO doc string
     *
     * @param df
     * @param indices
     * @return
     */
    public SDVariable gatherNd(SDVariable df, SDVariable indices) {
        return gatherNd(null, df, indices);
    }

    /**
     * TODO doc string
     *
     * @param name
     * @param df
     * @param indices
     * @return
     */
    public SDVariable gatherNd(String name, SDVariable df, SDVariable indices) {
        SDVariable ret = f().gatherNd(df, indices);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Create a new 1d array with values evenly spaced between values 'start' and 'stop'
     * For example, linspace(start=3.0, stop=4.0, number=3) will generate [3.0, 3.5, 4.0]
     *
     * @param name   Name of the new variable
     * @param from   Start value
     * @param to     Stop value
     * @param length Number of values to generate
     * @param dt     Data type of the output array
     * @return SDVariable with linearly spaced elements
     */
    public SDVariable linspace(String name, SDVariable from, SDVariable to, SDVariable length, DataType dt) {
        SDVariable ret = f().linspace(from, to, length, dt);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #stack(String, int, SDVariable...)
     */
    public SDVariable parallel_stack(SDVariable[] values) {
        return parallel_stack(null, values);
    }

    /**
     * @see #stack(String, int, SDVariable...)
     */
    public SDVariable parallel_stack(String name, SDVariable[] values) {
        validateSameType("parallel_stack", false, values);
        SDVariable ret = f().parallel_stack(values);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #repeat(String, SDVariable, int)
     */
    public SDVariable repeat(SDVariable df, int axis) {
        return repeat(null, df, axis);
    }

    /**
     * @see #repeat(String, SDVariable, int)
     */
    public SDVariable repeat(String name, SDVariable df, int axis) {
        SDVariable ret = f().repeat(df, axis);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #stack(String, int, SDVariable...)
     */
    public SDVariable stack(int axis, SDVariable... values) {
        return stack(null, axis, values);
    }

    /**
     * Stack a set of N SDVariables of rank X into one rank X+1 variable.
     * If inputs have shape [a,b,c] then output has shape:<br>
     * axis = 0: [N,a,b,c]<br>
     * axis = 1: [a,N,b,c]<br>
     * axis = 2: [a,b,N,c]<br>
     * axis = 3: [a,b,c,N]<br>
     *
     * @param name   Name of the output variable
     * @param axis   Axis to stack on
     * @param values Input variables to stack. Must have the same shape for all inputs
     * @return Output variable
     * @see #unstack(String[], SDVariable, int, int)
     */
    public SDVariable stack(String name, int axis, SDVariable... values) {
        validateSameType("stack", false, values);
        SDVariable ret = f().stack(values, axis);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @param x
     * @param y
     * @param dimensions
     * @return
     */
    public SDVariable tensorMmul(SDVariable x,
                                 SDVariable y,
                                 int[][] dimensions) {
        return tensorMmul(null, x, y, dimensions);
    }

    /**
     * @param x          Input variable x
     * @param y          Input variable y
     * @param dimensions dimensions
     * @return Output variable
     */
    public SDVariable tensorMmul(String name,
                                 SDVariable x,
                                 SDVariable y,
                                 int[][] dimensions) {
        validateNumerical("tensorMmul", x, y);
        SDVariable result = f().tensorMmul(x, y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * See {@link #whileLoop(String[], String, SDVariable[], SameDiffSingleLambda, SameDiffLambda)}
     */
    public SDVariable[] whileLoop(@NonNull SDVariable[] loopVars,
            @NonNull SameDiffSingleLambda cond, @NonNull SameDiffLambda body){
        return whileLoop(null, null, loopVars, cond, body);
    }

    /**
     * See {@link #whileLoop(String[], String, SDVariable[], SameDiffSingleLambda, SameDiffLambda)}
     */
    public SDVariable[] whileLoop(String loopName, @NonNull SDVariable[] loopVars,
            @NonNull SameDiffSingleLambda cond, @NonNull SameDiffLambda body){
        return whileLoop(null, loopName, loopVars, cond, body);
    }

    /**
     * Constructs a While loop using the tensorflow style control flow operations (Switch, Merge, Enter, Exit, and NextIteration)
     *
     * Repeatedly executes body on the loop variables and updates them with the results, until cond evaluates to false
     *
     * Note that cond and body lambdas are only called once to construct the graph.  The constructed graph is used for further iterations.
     *
     * See <a href="http://download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf">Tensorflow Control Flow Implementation</a>
     *
     * @param outputNames  Names to give the output variables.  If null, doesn't rename
     * @param loopName  The name of the loop block and frame (must be unique).  If null, uses "if"
     * @param loopVars  Loop variables' inputs
     * @param cond  A lambda evaluating to the loop condition
     * @param body  A lambda doing the loop operation and returning the new loop variable values
     * @return  The values of the loop variables once condition is false
     */
    public SDVariable[] whileLoop(String[] outputNames, final String loopName, @NonNull SDVariable[] loopVars,
            @NonNull SameDiffSingleLambda cond, @NonNull SameDiffLambda body){

        final String frameName = sd().newBlockName(loopName == null ? "while" : loopName);

        NameScope loopScope = sd().withNameScope(frameName);

        //SDVariable counter = SD.scalar(SD.generateNewVarName("counter", 0), 0);

        SDVariable[] entered = new SDVariable[loopVars.length];
        for(int i = 0 ; i < loopVars.length ; i++){
            entered[i] = f().enter(loopVars[i], frameName);
        }

        //counter = SD.f().enter(counter, frameName);

        SDVariable[] merged = new SDVariable[loopVars.length];
        Merge[] mergeOps = new Merge[loopVars.length];
        for(int i = 0 ; i < loopVars.length ; i++){
            // the second arg will later be replaced with the output of NextIteration
            // but that isn't available yet (and can't be, as it depends on this)
            mergeOps[i] = new Merge(sd(), entered[i], entered[i]);
            merged[i] = mergeOps[i].outputVariable();
        }

        //Merge counterMerge = new Merge(SD, counter, counter);
        //counter = counterMerge.outputVariable();

        NameScope condScope = sd().withNameScope("cond");
        SDVariable cond_result = cond.define(sd(), merged);
        condScope.close();


        if (cond_result.dataType() != DataType.BOOL)
            throw new IllegalStateException("Can not use " + cond_result.name() + " as the condition of an While loop, the condition must be a boolean.");


        final Set<String> alreadyEntered = Sets.newHashSet();
        SDVariable[] trueSwitches = new SDVariable[loopVars.length];
        SDVariable[] exits = new SDVariable[loopVars.length];
        for(int i = 0 ; i < loopVars.length ; i++){
            SDVariable[] s = f().switchOp(merged[i], cond_result);
            trueSwitches[i] = s[1];
            alreadyEntered.add(s[1].name());
            exits[i] = f().exit(s[0]);
        }

        //SDVariable[] cs = SD.f().switchOp(counter, cond_result);
        //SDVariable counterExit = SD.f().exit(cs[0]);
        //counter = cs[1];

        final Set<String> declared = Sets.newHashSet(sd().variableMap().keySet());
        final Map<String, SDVariable> done = new HashMap<>();

        sd().addArgumentInterceptor(new ArgumentInterceptor() {
            @Override
            public SDVariable intercept(SDVariable argument) {

                if(!declared.contains(argument.name()))
                    return argument;

                if(alreadyEntered.contains(argument.name()))
                    return argument;

                if(done.containsKey(argument.name()))
                    return done.get(argument.name());

                SDVariable e = f().enter(argument, frameName, true);
                done.put(argument.name(), e);
                return e;
            }
        });

        NameScope bodyScope = sd().withNameScope("body");
        SDVariable[] outs = body.define(sd(), trueSwitches);
        bodyScope.close();
        sd().removeArgumentInterceptor();

        //counter.add(1);

        for(int i = 0 ; i < loopVars.length ; i++){
            SDVariable n = f().nextIteration(outs[i]);
            mergeOps[i].replaceArg(1,n);
        }

        //counterMerge.replaceArg(1, counter);

        loopScope.close();
        return updateVariableNamesAndReferences(exits, outputNames);
    }

    /**
     * See {@link #ifCond(String, String, SameDiffNoArgSingleLambda, SameDiffNoArgSingleLambda, SameDiffNoArgSingleLambda)}
     */
    public SDVariable ifCond(@NonNull SameDiffNoArgSingleLambda cond,
            @NonNull SameDiffNoArgSingleLambda trueBody, @NonNull SameDiffNoArgSingleLambda falseBody){
        return ifCond(null, null, cond, trueBody, falseBody);
    }


    /**
     * See {@link #ifCond(String, String, SameDiffNoArgSingleLambda, SameDiffNoArgSingleLambda, SameDiffNoArgSingleLambda)}
     */
    public SDVariable ifCond(String ifName, @NonNull SameDiffNoArgSingleLambda cond,
            @NonNull SameDiffNoArgSingleLambda trueBody, @NonNull SameDiffNoArgSingleLambda falseBody){
        return ifCond(null, ifName, cond, trueBody, falseBody);
    }

    /**
     * Constructs a If statement using the tensorflow style control flow operations (Switch and Merge)
     *
     * If the result of cond is true, returns the result of trueBody, otherwise returns the result of falseBody
     *
     * Note that cond and body lambdas are only called once to construct the graph.  The constructed graph is used to evaluate.
     *
     * See <a href="http://download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf">Tensorflow Control Flow Implementation</a>
     *
     * @param outputName Name to give the output variable.  If null, doesn't rename
     * @param ifName  The name of the if block.  If null, uses "if"
     * @param cond  A lambda evaluating to the if condition
     * @param trueBody  A lambda to be executed if cond is true (the if block)
     * @param falseBody  A lambda to be executed if cond is false (the else block)
     * @return The value of trueBody if cond is true, or falseBody if it isn't
     */
    public SDVariable ifCond(String outputName, String ifName, @NonNull SameDiffNoArgSingleLambda cond,
            @NonNull SameDiffNoArgSingleLambda trueBody, @NonNull SameDiffNoArgSingleLambda falseBody){

        ifName = sd().newBlockName(ifName == null ? "if" : ifName);

        NameScope ifScope = sd().withNameScope(ifName);

        NameScope condScope = sd().withNameScope("cond");
        final SDVariable pred = cond.define(sd());
        condScope.close();

        if (pred.dataType() != DataType.BOOL) {
            //cleanup partially added block

            for(SDVariable v : sd().getVariablesInScope(ifScope))
                sd().getVariables().remove(v.name());

            for(SameDiffOp op : sd().getOpsInScope(ifScope)) {
                for(String in : op.getInputsToOp()){
                    sd().removeArgFromOp(in, op.getOp());
                }
                sd().getOps().remove(op.getName());
            }


            throw new IllegalStateException("Can not use " + pred.name()
                    + " as the condition of an If statement, the condition must be a boolean.");
        }

        final Map<String, SDVariable[]> switches = new HashMap<>();

        final Set<String> declared = Sets.newHashSet(sd().variableMap().keySet());

        sd().addArgumentInterceptor(new ArgumentInterceptor() {
            @Override
            public SDVariable intercept(SDVariable argument) {

                // if its declared in the if, we don't care acout it
                if(!declared.contains(argument.name()))
                    return argument;

                // if we've already added a switch, move on
                if(switches.containsKey(argument.name()))
                    return switches.get(argument.name())[1];

                SDVariable[] s = f().switchOp(argument, pred);
                switches.put(argument.name(), s);
                return s[1];
            }
        });
        NameScope trueScope = sd().withNameScope("trueBody");
        SDVariable trueOut = trueBody.define(sd());
        sd().removeArgumentInterceptor();

        if(declared.contains(trueOut.name())) {
            SDVariable[] s = f().switchOp(trueOut, pred);
            switches.put(trueOut.name(), s);
            trueOut = s[1];
        }

        trueScope.close();

        final Set<String> declared2 = Sets.newHashSet(sd().variableMap().keySet());
        sd().addArgumentInterceptor(new ArgumentInterceptor() {
            @Override
            public SDVariable intercept(SDVariable argument) {

                // if its declared in the if, we don't care acout it
                if(!declared2.contains(argument.name()))
                    return argument;

                // if we've already added a switch, move on
                if(switches.containsKey(argument.name()))
                    return switches.get(argument.name())[0];

                SDVariable[] s = f().switchOp(argument, pred);
                switches.put(argument.name(), s);
                return s[0];
            }
        });
        NameScope falseScope = sd().withNameScope("falseBody");
        SDVariable falseOut = falseBody.define(sd());
        sd().removeArgumentInterceptor();

        if(declared2.contains(falseOut.name())) {
            SDVariable[] s = f().switchOp(falseOut, pred);
            switches.put(falseOut.name(), s);
            falseOut = s[0];
        }
        falseScope.close();

        SDVariable output = f().merge(trueOut, falseOut);

        ifScope.close();

        return updateVariableNameAndReference(output, outputName);
    }

    /**
     * Boolean and array reduction operation, optionally along specified dimensions<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) (BOOL type)
     */
    public SDVariable all(SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("all", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.bool.All(sd(),x, dimensions).outputVariable();
    }

    /**
     * Boolean and array reduction operation, optionally along specified dimensions<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) (BOOL type)
     */
    public SDVariable all(String name, SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("all", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.bool.All(sd(),x, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Boolean or array reduction operation, optionally along specified dimensions<br>
     *
     * @param x  Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) (BOOL type)
     */
    public SDVariable any(SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("any", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.bool.Any(sd(),x, dimensions).outputVariable();
    }

    /**
     * Boolean or array reduction operation, optionally along specified dimensions<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x  Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) (BOOL type)
     */
    public SDVariable any(String name, SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("any", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.bool.Any(sd(),x, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Argmax array reduction operation, optionally along specified dimensions.<br>
     * Output values are the index of the maximum value of each slice along the specified dimension.<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param in Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or
     *  of rank (input rank) if keepdims = true (NUMERIC type)
     */
    public SDVariable argmax(SDVariable in, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("argmax", "in", in);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.indexaccum.IMax(sd(),in, keepDims, dimensions).outputVariable();
    }

    /**
     * Argmax array reduction operation, optionally along specified dimensions.<br>
     * Output values are the index of the maximum value of each slice along the specified dimension.<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or
     *  of rank (input rank) if keepdims = true (NUMERIC type)
     */
    public SDVariable argmax(String name, SDVariable in, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("argmax", "in", in);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.indexaccum.IMax(sd(),in, keepDims, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Argmax array reduction operation, optionally along specified dimensions.<br>
     * Output values are the index of the maximum value of each slice along the specified dimension.<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param in Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or
     *  of rank (input rank) if keepdims = true (NUMERIC type)
     */
    public SDVariable argmax(SDVariable in, int... dimensions) {
        SDValidation.validateNumerical("argmax", "in", in);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.indexaccum.IMax(sd(),in, false, dimensions).outputVariable();
    }

    /**
     * Argmax array reduction operation, optionally along specified dimensions.<br>
     * Output values are the index of the maximum value of each slice along the specified dimension.<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or
     *  of rank (input rank) if keepdims = true (NUMERIC type)
     */
    public SDVariable argmax(String name, SDVariable in, int... dimensions) {
        SDValidation.validateNumerical("argmax", "in", in);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.indexaccum.IMax(sd(),in, false, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Argmin array reduction operation, optionally along specified dimensions.<br>
     * Output values are the index of the minimum value of each slice along the specified dimension.<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * @param in Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or of rank (input rank) if keepdims = true (NUMERIC type)
     */
    public SDVariable argmin(SDVariable in, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("argmin", "in", in);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.indexaccum.IMin(sd(),in, keepDims, dimensions).outputVariable();
    }

    /**
     * Argmin array reduction operation, optionally along specified dimensions.<br>
     * Output values are the index of the minimum value of each slice along the specified dimension.<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or of rank (input rank) if keepdims = true (NUMERIC type)
     */
    public SDVariable argmin(String name, SDVariable in, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("argmin", "in", in);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.indexaccum.IMin(sd(),in, keepDims, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Argmin array reduction operation, optionally along specified dimensions.<br>
     * Output values are the index of the minimum value of each slice along the specified dimension.<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * @param in Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or of rank (input rank) if keepdims = true (NUMERIC type)
     */
    public SDVariable argmin(SDVariable in, int... dimensions) {
        SDValidation.validateNumerical("argmin", "in", in);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.indexaccum.IMin(sd(),in, false, dimensions).outputVariable();
    }

    /**
     * Argmin array reduction operation, optionally along specified dimensions.<br>
     * Output values are the index of the minimum value of each slice along the specified dimension.<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or of rank (input rank) if keepdims = true (NUMERIC type)
     */
    public SDVariable argmin(String name, SDVariable in, int... dimensions) {
        SDValidation.validateNumerical("argmin", "in", in);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.indexaccum.IMin(sd(),in, false, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Cast input tensor to input data type.<br>
     *
     * @param toCast Input variable (NUMERIC type)
     * @param toType Target data type
     * @return output Input variable being cast to passed data type (BOOL type)
     */
    public SDVariable castTo(SDVariable toCast, DataType toType) {
        SDValidation.validateNumerical("castTo", "toCast", toCast);
        return new org.nd4j.linalg.api.ops.impl.transforms.dtype.Cast(sd(),toCast, toType).outputVariable();
    }

    /**
     * Cast input tensor to input data type.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param toCast Input variable (NUMERIC type)
     * @param toType Target data type
     * @return output Input variable being cast to passed data type (BOOL type)
     */
    public SDVariable castTo(String name, SDVariable toCast, DataType toType) {
        SDValidation.validateNumerical("castTo", "toCast", toCast);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.dtype.Cast(sd(),toCast, toType).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Concatenate a set of inputs along the specified dimension.<br>
     * Note that inputs must have identical rank and identical dimensions, other than the dimension to stack on.<br>
     * For example, if 2 inputs have shape [a, x, c] and [a, y, c] and dimension = 1, then the output has shape [a, x+y, c]<br>
     *
     * Inputs must satisfy the following constraints: <br>
     * Input arrays must all be the same datatype: isSameType(inputs)<br>
     *
     * @param inputs Input variables (NUMERIC type)
     * @param dimension Dimension to concatenate on
     * @return output  (NUMERIC type)
     */
    public SDVariable concat(SDVariable[] inputs, int dimension) {
        SDValidation.validateNumerical("concat", "inputs", inputs);
        Preconditions.checkArgument(inputs.length >= 1, "inputs has incorrect size/length. Expected: inputs.length >= 1, got %s", inputs.length);
        Preconditions.checkArgument(isSameType(inputs), "Input arrays must all be the same datatype");
        return new org.nd4j.linalg.api.ops.impl.shape.Concat(sd(),inputs, dimension).outputVariable();
    }

    /**
     * Concatenate a set of inputs along the specified dimension.<br>
     * Note that inputs must have identical rank and identical dimensions, other than the dimension to stack on.<br>
     * For example, if 2 inputs have shape [a, x, c] and [a, y, c] and dimension = 1, then the output has shape [a, x+y, c]<br>
     *
     * Inputs must satisfy the following constraints: <br>
     * Input arrays must all be the same datatype: isSameType(inputs)<br>
     *
     * @param name name May be null. Name for the output variable
     * @param inputs Input variables (NUMERIC type)
     * @param dimension Dimension to concatenate on
     * @return output  (NUMERIC type)
     */
    public SDVariable concat(String name, SDVariable[] inputs, int dimension) {
        SDValidation.validateNumerical("concat", "inputs", inputs);
        Preconditions.checkArgument(inputs.length >= 1, "inputs has incorrect size/length. Expected: inputs.length >= 1, got %s", inputs.length);
        Preconditions.checkArgument(isSameType(inputs), "Input arrays must all be the same datatype");
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Concat(sd(),inputs, dimension).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Cumulative product operation.<br>
     * For input: [ a, b, c], output is:<br>
     * exclusive=false, reverse=false: [a, a*b, a*b*c]<br>
     * exclusive=true, reverse=false, [0, a, a*b]<br>
     * exclusive=false, reverse=true: [a*b*c, b*c, c]<br>
     * exclusive=true, reverse=true: [b*c, c, 0]<br>
     *
     * @param in Input variable (NUMERIC type)
     * @param exclusive If true: exclude the first value
     * @param reverse If true: reverse the direction of the accumulation
     * @param axis Scalar axis argument for dimension to perform cumululative sum operations along (Size: AtLeast(min=1))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable cumprod(SDVariable in, boolean exclusive, boolean reverse, int... axis) {
        SDValidation.validateNumerical("cumprod", "in", in);
        Preconditions.checkArgument(axis.length >= 1, "axis has incorrect size/length. Expected: axis.length >= 1, got %s", axis.length);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.CumProd(sd(),in, exclusive, reverse, axis).outputVariable();
    }

    /**
     * Cumulative product operation.<br>
     * For input: [ a, b, c], output is:<br>
     * exclusive=false, reverse=false: [a, a*b, a*b*c]<br>
     * exclusive=true, reverse=false, [0, a, a*b]<br>
     * exclusive=false, reverse=true: [a*b*c, b*c, c]<br>
     * exclusive=true, reverse=true: [b*c, c, 0]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @param exclusive If true: exclude the first value
     * @param reverse If true: reverse the direction of the accumulation
     * @param axis Scalar axis argument for dimension to perform cumululative sum operations along (Size: AtLeast(min=1))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable cumprod(String name, SDVariable in, boolean exclusive, boolean reverse,
                              int... axis) {
        SDValidation.validateNumerical("cumprod", "in", in);
        Preconditions.checkArgument(axis.length >= 1, "axis has incorrect size/length. Expected: axis.length >= 1, got %s", axis.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CumProd(sd(),in, exclusive, reverse, axis).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Cumulative product operation.<br>
     * For input: [ a, b, c], output is:<br>
     * exclusive=false, reverse=false: [a, a*b, a*b*c]<br>
     * exclusive=true, reverse=false, [0, a, a*b]<br>
     * exclusive=false, reverse=true: [a*b*c, b*c, c]<br>
     * exclusive=true, reverse=true: [b*c, c, 0]<br>
     *
     * @param in Input variable (NUMERIC type)
     * @param axis Scalar axis argument for dimension to perform cumululative sum operations along (Size: AtLeast(min=1))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable cumprod(SDVariable in, int... axis) {
        SDValidation.validateNumerical("cumprod", "in", in);
        Preconditions.checkArgument(axis.length >= 1, "axis has incorrect size/length. Expected: axis.length >= 1, got %s", axis.length);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.CumProd(sd(),in, false, false, axis).outputVariable();
    }

    /**
     * Cumulative product operation.<br>
     * For input: [ a, b, c], output is:<br>
     * exclusive=false, reverse=false: [a, a*b, a*b*c]<br>
     * exclusive=true, reverse=false, [0, a, a*b]<br>
     * exclusive=false, reverse=true: [a*b*c, b*c, c]<br>
     * exclusive=true, reverse=true: [b*c, c, 0]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @param axis Scalar axis argument for dimension to perform cumululative sum operations along (Size: AtLeast(min=1))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable cumprod(String name, SDVariable in, int... axis) {
        SDValidation.validateNumerical("cumprod", "in", in);
        Preconditions.checkArgument(axis.length >= 1, "axis has incorrect size/length. Expected: axis.length >= 1, got %s", axis.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CumProd(sd(),in, false, false, axis).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Cumulative sum operation.<br>
     * For input: [ a, b, c], output is:<br>
     * exclusive=false, reverse=false: [a, a+b, a+b+c]<br>
     * exclusive=true, reverse=false, [0, a, a+b]<br>
     * exclusive=false, reverse=true: [a+b+c, b+c, c]<br>
     * exclusive=true, reverse=true: [b+c, c, 0]<br>
     *
     * @param in Input variable (NUMERIC type)
     * @param exclusive If true: exclude the first value
     * @param reverse If true: reverse the direction of the accumulation
     * @param axis Scalar axis argument for dimension to perform cumululative sum operations along (Size: AtLeast(min=1))
     * @return output  (NUMERIC type)
     */
    public SDVariable cumsum(SDVariable in, boolean exclusive, boolean reverse, int... axis) {
        SDValidation.validateNumerical("cumsum", "in", in);
        Preconditions.checkArgument(axis.length >= 1, "axis has incorrect size/length. Expected: axis.length >= 1, got %s", axis.length);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.CumSum(sd(),in, exclusive, reverse, axis).outputVariable();
    }

    /**
     * Cumulative sum operation.<br>
     * For input: [ a, b, c], output is:<br>
     * exclusive=false, reverse=false: [a, a+b, a+b+c]<br>
     * exclusive=true, reverse=false, [0, a, a+b]<br>
     * exclusive=false, reverse=true: [a+b+c, b+c, c]<br>
     * exclusive=true, reverse=true: [b+c, c, 0]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @param exclusive If true: exclude the first value
     * @param reverse If true: reverse the direction of the accumulation
     * @param axis Scalar axis argument for dimension to perform cumululative sum operations along (Size: AtLeast(min=1))
     * @return output  (NUMERIC type)
     */
    public SDVariable cumsum(String name, SDVariable in, boolean exclusive, boolean reverse,
                             int... axis) {
        SDValidation.validateNumerical("cumsum", "in", in);
        Preconditions.checkArgument(axis.length >= 1, "axis has incorrect size/length. Expected: axis.length >= 1, got %s", axis.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CumSum(sd(),in, exclusive, reverse, axis).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Cumulative sum operation.<br>
     * For input: [ a, b, c], output is:<br>
     * exclusive=false, reverse=false: [a, a+b, a+b+c]<br>
     * exclusive=true, reverse=false, [0, a, a+b]<br>
     * exclusive=false, reverse=true: [a+b+c, b+c, c]<br>
     * exclusive=true, reverse=true: [b+c, c, 0]<br>
     *
     * @param in Input variable (NUMERIC type)
     * @param axis Scalar axis argument for dimension to perform cumululative sum operations along (Size: AtLeast(min=1))
     * @return output  (NUMERIC type)
     */
    public SDVariable cumsum(SDVariable in, int... axis) {
        SDValidation.validateNumerical("cumsum", "in", in);
        Preconditions.checkArgument(axis.length >= 1, "axis has incorrect size/length. Expected: axis.length >= 1, got %s", axis.length);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.CumSum(sd(),in, false, false, axis).outputVariable();
    }

    /**
     * Cumulative sum operation.<br>
     * For input: [ a, b, c], output is:<br>
     * exclusive=false, reverse=false: [a, a+b, a+b+c]<br>
     * exclusive=true, reverse=false, [0, a, a+b]<br>
     * exclusive=false, reverse=true: [a+b+c, b+c, c]<br>
     * exclusive=true, reverse=true: [b+c, c, 0]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @param axis Scalar axis argument for dimension to perform cumululative sum operations along (Size: AtLeast(min=1))
     * @return output  (NUMERIC type)
     */
    public SDVariable cumsum(String name, SDVariable in, int... axis) {
        SDValidation.validateNumerical("cumsum", "in", in);
        Preconditions.checkArgument(axis.length >= 1, "axis has incorrect size/length. Expected: axis.length >= 1, got %s", axis.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CumSum(sd(),in, false, false, axis).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Pairwise dot product reduction along dimension<br>
     * output = sum(i=0 ... size(dim)-1) x[i] * y[i]<br>
     *
     * @param x first input (NUMERIC type)
     * @param y second input (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output output variable (NUMERIC type)
     */
    public SDVariable dot(SDVariable x, SDVariable y, int... dimensions) {
        SDValidation.validateNumerical("dot", "x", x);
        SDValidation.validateNumerical("dot", "y", y);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce3.Dot(sd(),x, y, dimensions).outputVariable();
    }

    /**
     * Pairwise dot product reduction along dimension<br>
     * output = sum(i=0 ... size(dim)-1) x[i] * y[i]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x first input (NUMERIC type)
     * @param y second input (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output output variable (NUMERIC type)
     */
    public SDVariable dot(String name, SDVariable x, SDVariable y, int... dimensions) {
        SDValidation.validateNumerical("dot", "x", x);
        SDValidation.validateNumerical("dot", "y", y);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce3.Dot(sd(),x, y, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Dynamically partition the input variable values into the specified number of paritions, using the indices.<br>
     * Example:<br>
     * <pre><br>
     * input = [1,2,3,4,5]<br>
     * numPartitions = 2<br>
     * partitions = [1,0,0,1,0]<br>
     * out[0] = [2,3,5]<br>
     * out[1] = [1,4] }<br>
     * </pre><br>
     *
     * @param x Input variable (NUMERIC type)
     * @param partitions 1D input with values 0 to numPartitions-1 (INT type)
     * @param numPartitions Number of partitions, >= 1
     * @return output Output variables (equal in number to numPartitions) (NUMERIC type)
     */
    public SDVariable dynamicPartition(SDVariable x, SDVariable[] partitions, int numPartitions) {
        SDValidation.validateNumerical("dynamicPartition", "x", x);
        SDValidation.validateInteger("dynamicPartition", "partitions", partitions);
        Preconditions.checkArgument(partitions.length >= 1, "partitions has incorrect size/length. Expected: partitions.length >= 1, got %s", partitions.length);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.DynamicPartition(sd(),x, partitions, numPartitions).outputVariable();
    }

    /**
     * Dynamically partition the input variable values into the specified number of paritions, using the indices.<br>
     * Example:<br>
     * <pre><br>
     * input = [1,2,3,4,5]<br>
     * numPartitions = 2<br>
     * partitions = [1,0,0,1,0]<br>
     * out[0] = [2,3,5]<br>
     * out[1] = [1,4] }<br>
     * </pre><br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param partitions 1D input with values 0 to numPartitions-1 (INT type)
     * @param numPartitions Number of partitions, >= 1
     * @return output Output variables (equal in number to numPartitions) (NUMERIC type)
     */
    public SDVariable dynamicPartition(String name, SDVariable x, SDVariable[] partitions,
                                       int numPartitions) {
        SDValidation.validateNumerical("dynamicPartition", "x", x);
        SDValidation.validateInteger("dynamicPartition", "partitions", partitions);
        Preconditions.checkArgument(partitions.length >= 1, "partitions has incorrect size/length. Expected: partitions.length >= 1, got %s", partitions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.DynamicPartition(sd(),x, partitions, numPartitions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Dynamically merge the specified input arrays into a single array, using the specified indices<br>
     *
     * @param x Input variables. (NUMERIC type)
     * @param indices Indices to use when merging. Must be >= 1, same length as input variables (INT type)
     * @return output Merged output variable (NUMERIC type)
     */
    public SDVariable dynamicStitch(SDVariable[] x, SDVariable[] indices) {
        SDValidation.validateNumerical("dynamicStitch", "x", x);
        Preconditions.checkArgument(x.length >= 1, "x has incorrect size/length. Expected: x.length >= 1, got %s", x.length);
        SDValidation.validateInteger("dynamicStitch", "indices", indices);
        Preconditions.checkArgument(indices.length >= 1, "indices has incorrect size/length. Expected: indices.length >= 1, got %s", indices.length);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.DynamicStitch(sd(),x, indices).outputVariable();
    }

    /**
     * Dynamically merge the specified input arrays into a single array, using the specified indices<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variables. (NUMERIC type)
     * @param indices Indices to use when merging. Must be >= 1, same length as input variables (INT type)
     * @return output Merged output variable (NUMERIC type)
     */
    public SDVariable dynamicStitch(String name, SDVariable[] x, SDVariable[] indices) {
        SDValidation.validateNumerical("dynamicStitch", "x", x);
        Preconditions.checkArgument(x.length >= 1, "x has incorrect size/length. Expected: x.length >= 1, got %s", x.length);
        SDValidation.validateInteger("dynamicStitch", "indices", indices);
        Preconditions.checkArgument(indices.length >= 1, "indices has incorrect size/length. Expected: indices.length >= 1, got %s", indices.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.DynamicStitch(sd(),x, indices).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Equals operation: elementwise x == y<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param x Input array (NUMERIC type)
     * @param y Double value argument to use in operation
     * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable eq(SDVariable x, double y) {
        SDValidation.validateNumerical("eq", "x", x);
        return new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarEquals(sd(),x, y).outputVariable();
    }

    /**
     * Equals operation: elementwise x == y<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input array (NUMERIC type)
     * @param y Double value argument to use in operation
     * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable eq(String name, SDVariable x, double y) {
        SDValidation.validateNumerical("eq", "x", x);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarEquals(sd(),x, y).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Equal to operation: elementwise x == y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param x Input 1 (NUMERIC type)
     * @param y Input 2 (NUMERIC type)
     * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable eq(SDVariable x, SDVariable y) {
        SDValidation.validateNumerical("eq", "x", x);
        SDValidation.validateNumerical("eq", "y", y);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.EqualTo(sd(),x, y).outputVariable();
    }

    /**
     * Equal to operation: elementwise x == y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input 1 (NUMERIC type)
     * @param y Input 2 (NUMERIC type)
     * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable eq(String name, SDVariable x, SDVariable y) {
        SDValidation.validateNumerical("eq", "x", x);
        SDValidation.validateNumerical("eq", "y", y);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.EqualTo(sd(),x, y).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Reshape the input by adding a 1 at the specified location.<br>
     * For example, if input has shape [a, b], then output shape is:<br>
     * axis = 0: [1, a, b]<br>
     * axis = 1: [a, 1, b]<br>
     * axis = 2: [a, b, 1]<br>
     *
     * @param x Input variable (NDARRAY type)
     * @param axis Axis to expand
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable expandDims(SDVariable x, int axis) {
        return new org.nd4j.linalg.api.ops.impl.shape.ExpandDims(sd(),x, axis).outputVariable();
    }

    /**
     * Reshape the input by adding a 1 at the specified location.<br>
     * For example, if input has shape [a, b], then output shape is:<br>
     * axis = 0: [1, a, b]<br>
     * axis = 1: [a, 1, b]<br>
     * axis = 2: [a, b, 1]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NDARRAY type)
     * @param axis Axis to expand
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable expandDims(String name, SDVariable x, int axis) {
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.ExpandDims(sd(),x, axis).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Generate an output variable with the specified (dynamic) shape with all elements set to the specified value<br>
     *
     * @param shape Shape: must be a 1D array/variable (INT type)
     * @param dataType Datatype of the output array
     * @param value Value to set all elements to
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable fill(SDVariable shape, DataType dataType, double value) {
        SDValidation.validateInteger("fill", "shape", shape);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.Fill(sd(),shape, dataType, value).outputVariable();
    }

    /**
     * Generate an output variable with the specified (dynamic) shape with all elements set to the specified value<br>
     *
     * @param name name May be null. Name for the output variable
     * @param shape Shape: must be a 1D array/variable (INT type)
     * @param dataType Datatype of the output array
     * @param value Value to set all elements to
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable fill(String name, SDVariable shape, DataType dataType, double value) {
        SDValidation.validateInteger("fill", "shape", shape);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.Fill(sd(),shape, dataType, value).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Gather slices from the input variable where the indices are specified as fixed int[] values.<br>
     * Output shape is same as input shape, except for axis dimension, which has size equal to indices.length.<br>
     *
     * @param df Input variable (NUMERIC type)
     * @param indices Indices to get (Size: AtLeast(min=1))
     * @param axis Axis that the indices refer to
     * @return output Output variable with slices pulled from the specified axis (NUMERIC type)
     */
    public SDVariable gather(SDVariable df, int[] indices, int axis) {
        SDValidation.validateNumerical("gather", "df", df);
        Preconditions.checkArgument(indices.length >= 1, "indices has incorrect size/length. Expected: indices.length >= 1, got %s", indices.length);
        return new org.nd4j.linalg.api.ops.impl.shape.Gather(sd(),df, indices, axis).outputVariable();
    }

    /**
     * Gather slices from the input variable where the indices are specified as fixed int[] values.<br>
     * Output shape is same as input shape, except for axis dimension, which has size equal to indices.length.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param df Input variable (NUMERIC type)
     * @param indices Indices to get (Size: AtLeast(min=1))
     * @param axis Axis that the indices refer to
     * @return output Output variable with slices pulled from the specified axis (NUMERIC type)
     */
    public SDVariable gather(String name, SDVariable df, int[] indices, int axis) {
        SDValidation.validateNumerical("gather", "df", df);
        Preconditions.checkArgument(indices.length >= 1, "indices has incorrect size/length. Expected: indices.length >= 1, got %s", indices.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Gather(sd(),df, indices, axis).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Gather slices from the input variable where the indices are specified as dynamic array values.<br>
     * Output shape is same as input shape, except for axis dimension, which has size equal to indices.length.<br>
     *
     * @param df Input variable (NUMERIC type)
     * @param indices Indices to get slices for. Rank 0 or 1 input (INT type)
     * @param axis Axis that the indices refer to
     * @return output Output variable with slices pulled from the specified axis (NUMERIC type)
     */
    public SDVariable gather(SDVariable df, SDVariable indices, int axis) {
        SDValidation.validateNumerical("gather", "df", df);
        SDValidation.validateInteger("gather", "indices", indices);
        return new org.nd4j.linalg.api.ops.impl.shape.Gather(sd(),df, indices, axis).outputVariable();
    }

    /**
     * Gather slices from the input variable where the indices are specified as dynamic array values.<br>
     * Output shape is same as input shape, except for axis dimension, which has size equal to indices.length.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param df Input variable (NUMERIC type)
     * @param indices Indices to get slices for. Rank 0 or 1 input (INT type)
     * @param axis Axis that the indices refer to
     * @return output Output variable with slices pulled from the specified axis (NUMERIC type)
     */
    public SDVariable gather(String name, SDVariable df, SDVariable indices, int axis) {
        SDValidation.validateNumerical("gather", "df", df);
        SDValidation.validateInteger("gather", "indices", indices);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Gather(sd(),df, indices, axis).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Gather slices from df with shape specified by indices. <br>
     *
     * @param df  (NUMERIC type)
     * @param indices  (NUMERIC type)
     * @return output  (NUMERIC type)
     */
    public SDVariable gatherNd(SDVariable[] df, SDVariable[] indices) {
        SDValidation.validateNumerical("gatherNd", "df", df);
        Preconditions.checkArgument(df.length >= 1, "df has incorrect size/length. Expected: df.length >= 1, got %s", df.length);
        SDValidation.validateNumerical("gatherNd", "indices", indices);
        Preconditions.checkArgument(indices.length >= 1, "indices has incorrect size/length. Expected: indices.length >= 1, got %s", indices.length);
        return new org.nd4j.linalg.api.ops.impl.shape.GatherNd(sd(),df, indices).outputVariable();
    }

    /**
     * Gather slices from df with shape specified by indices. <br>
     *
     * @param name name May be null. Name for the output variable
     * @param df  (NUMERIC type)
     * @param indices  (NUMERIC type)
     * @return output  (NUMERIC type)
     */
    public SDVariable gatherNd(String name, SDVariable[] df, SDVariable[] indices) {
        SDValidation.validateNumerical("gatherNd", "df", df);
        Preconditions.checkArgument(df.length >= 1, "df has incorrect size/length. Expected: df.length >= 1, got %s", df.length);
        SDValidation.validateNumerical("gatherNd", "indices", indices);
        Preconditions.checkArgument(indices.length >= 1, "indices has incorrect size/length. Expected: indices.length >= 1, got %s", indices.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.GatherNd(sd(),df, indices).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Greater than operation: elementwise x > y<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param x Input array (NUMERIC type)
     * @param y Double value argument to use in operation
     * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable gt(SDVariable x, double y) {
        SDValidation.validateNumerical("gt", "x", x);
        return new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThan(sd(),x, y).outputVariable();
    }

    /**
     * Greater than operation: elementwise x > y<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input array (NUMERIC type)
     * @param y Double value argument to use in operation
     * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable gt(String name, SDVariable x, double y) {
        SDValidation.validateNumerical("gt", "x", x);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThan(sd(),x, y).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Greater than operation: elementwise x > y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param x Input 1 (NUMERIC type)
     * @param y Input 2 (NUMERIC type)
     * @return output Output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable gt(SDVariable x, SDVariable y) {
        SDValidation.validateNumerical("gt", "x", x);
        SDValidation.validateNumerical("gt", "y", y);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.GreaterThan(sd(),x, y).outputVariable();
    }

    /**
     * Greater than operation: elementwise x > y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input 1 (NUMERIC type)
     * @param y Input 2 (NUMERIC type)
     * @return output Output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable gt(String name, SDVariable x, SDVariable y) {
        SDValidation.validateNumerical("gt", "x", x);
        SDValidation.validateNumerical("gt", "y", y);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.GreaterThan(sd(),x, y).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Greater than or equals operation: elementwise x >= y<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param x Input array (NUMERIC type)
     * @param y Double value argument to use in operation
     * @return output Output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable gte(SDVariable x, double y) {
        SDValidation.validateNumerical("gte", "x", x);
        return new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThanOrEqual(sd(),x, y).outputVariable();
    }

    /**
     * Greater than or equals operation: elementwise x >= y<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input array (NUMERIC type)
     * @param y Double value argument to use in operation
     * @return output Output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable gte(String name, SDVariable x, double y) {
        SDValidation.validateNumerical("gte", "x", x);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThanOrEqual(sd(),x, y).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Greater than or equal to operation: elementwise x >= y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param x Input 1 (NUMERIC type)
     * @param y Input 2 (NUMERIC type)
     * @return output  (NUMERIC type)
     */
    public SDVariable gte(SDVariable x, SDVariable y) {
        SDValidation.validateNumerical("gte", "x", x);
        SDValidation.validateNumerical("gte", "y", y);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.GreaterThanOrEqual(sd(),x, y).outputVariable();
    }

    /**
     * Greater than or equal to operation: elementwise x >= y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input 1 (NUMERIC type)
     * @param y Input 2 (NUMERIC type)
     * @return output  (NUMERIC type)
     */
    public SDVariable gte(String name, SDVariable x, SDVariable y) {
        SDValidation.validateNumerical("gte", "x", x);
        SDValidation.validateNumerical("gte", "y", y);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.GreaterThanOrEqual(sd(),x, y).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Elementwise identity operation: out = x<br>
     *
     * @param input Input variable (NUMERIC type)
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable identity(SDVariable input) {
        SDValidation.validateNumerical("identity", "input", input);
        return new org.nd4j.linalg.api.ops.impl.transforms.same.Identity(sd(),input).outputVariable();
    }

    /**
     * Elementwise identity operation: out = x<br>
     *
     * @param name name May be null. Name for the output variable
     * @param input Input variable (NUMERIC type)
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable identity(String name, SDVariable input) {
        SDValidation.validateNumerical("identity", "input", input);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.same.Identity(sd(),input).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Compute the inverse permutation indices for a permutation operation<br>
     * Example: if input is [2, 0, 1] then output is [1, 2, 0]<br>
     * The idea is that x.permute(input).permute(invertPermutation(input)) == x<br>
     *
     * @param input 1D indices for permutation (INT type)
     * @return output 1D inverted permutation (INT type)
     */
    public SDVariable invertPermutation(SDVariable input) {
        SDValidation.validateInteger("invertPermutation", "input", input);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.InvertPermutation(sd(),input).outputVariable();
    }

    /**
     * Compute the inverse permutation indices for a permutation operation<br>
     * Example: if input is [2, 0, 1] then output is [1, 2, 0]<br>
     * The idea is that x.permute(input).permute(invertPermutation(input)) == x<br>
     *
     * @param name name May be null. Name for the output variable
     * @param input 1D indices for permutation (INT type)
     * @return output 1D inverted permutation (INT type)
     */
    public SDVariable invertPermutation(String name, SDVariable input) {
        SDValidation.validateInteger("invertPermutation", "input", input);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.InvertPermutation(sd(),input).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Is the director a numeric tensor? In the current version of ND4J/SameDiff, this always returns true/1<br>
     *
     * @param x Input variable (NUMERIC type)
     * @return output scalar boolean with value true or false (NDARRAY type)
     */
    public SDVariable isNumericTensor(SDVariable x) {
        SDValidation.validateNumerical("isNumericTensor", "x", x);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.IsNumericTensor(sd(),x).outputVariable();
    }

    /**
     * Is the director a numeric tensor? In the current version of ND4J/SameDiff, this always returns true/1<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @return output scalar boolean with value true or false (NDARRAY type)
     */
    public SDVariable isNumericTensor(String name, SDVariable x) {
        SDValidation.validateNumerical("isNumericTensor", "x", x);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.IsNumericTensor(sd(),x).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Create a new 1d array with values evenly spaced between values 'start' and 'stop'<br>
     * For example, linspace(start=3.0, stop=4.0, number=3) will generate [3.0, 3.5, 4.0]<br>
     *
     * @param dataType Data type of the output array
     * @param start Start value
     * @param stop Stop value
     * @param number Number of values to generate
     * @return output INDArray  with linearly spaced elements (NUMERIC type)
     */
    public SDVariable linspace(DataType dataType, double start, double stop, long number) {
        return new org.nd4j.linalg.api.ops.impl.shape.Linspace(sd(),dataType, start, stop, number).outputVariable();
    }

    /**
     * Create a new 1d array with values evenly spaced between values 'start' and 'stop'<br>
     * For example, linspace(start=3.0, stop=4.0, number=3) will generate [3.0, 3.5, 4.0]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param dataType Data type of the output array
     * @param start Start value
     * @param stop Stop value
     * @param number Number of values to generate
     * @return output INDArray  with linearly spaced elements (NUMERIC type)
     */
    public SDVariable linspace(String name, DataType dataType, double start, double stop,
                               long number) {
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Linspace(sd(),dataType, start, stop, number).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Less than operation: elementwise x < y<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param x Input array (NUMERIC type)
     * @param y Double value argument to use in operation
     * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable lt(SDVariable x, double y) {
        SDValidation.validateNumerical("lt", "x", x);
        return new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarLessThan(sd(),x, y).outputVariable();
    }

    /**
     * Less than operation: elementwise x < y<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input array (NUMERIC type)
     * @param y Double value argument to use in operation
     * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable lt(String name, SDVariable x, double y) {
        SDValidation.validateNumerical("lt", "x", x);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarLessThan(sd(),x, y).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Less than operation: elementwise x < y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param x Input 1 (NUMERIC type)
     * @param y Input 2 (NUMERIC type)
     * @return output Output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable lt(SDVariable x, SDVariable y) {
        SDValidation.validateNumerical("lt", "x", x);
        SDValidation.validateNumerical("lt", "y", y);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.LessThan(sd(),x, y).outputVariable();
    }

    /**
     * Less than operation: elementwise x < y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input 1 (NUMERIC type)
     * @param y Input 2 (NUMERIC type)
     * @return output Output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable lt(String name, SDVariable x, SDVariable y) {
        SDValidation.validateNumerical("lt", "x", x);
        SDValidation.validateNumerical("lt", "y", y);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.LessThan(sd(),x, y).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Less than or equals operation: elementwise x <= y<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param x Input array (NUMERIC type)
     * @param y Double value argument to use in operation
     * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable lte(SDVariable x, double y) {
        SDValidation.validateNumerical("lte", "x", x);
        return new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarLessThanOrEqual(sd(),x, y).outputVariable();
    }

    /**
     * Less than or equals operation: elementwise x <= y<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input array (NUMERIC type)
     * @param y Double value argument to use in operation
     * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable lte(String name, SDVariable x, double y) {
        SDValidation.validateNumerical("lte", "x", x);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarLessThanOrEqual(sd(),x, y).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Less than or equal to operation: elementwise x <= y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param x Input 1 (NUMERIC type)
     * @param y Input 2 (NUMERIC type)
     * @return output Output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable lte(SDVariable x, SDVariable y) {
        SDValidation.validateNumerical("lte", "x", x);
        SDValidation.validateNumerical("lte", "y", y);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.LessThanOrEqual(sd(),x, y).outputVariable();
    }

    /**
     * Less than or equal to operation: elementwise x <= y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input 1 (NUMERIC type)
     * @param y Input 2 (NUMERIC type)
     * @return output Output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable lte(String name, SDVariable x, SDVariable y) {
        SDValidation.validateNumerical("lte", "x", x);
        SDValidation.validateNumerical("lte", "y", y);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.LessThanOrEqual(sd(),x, y).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Returns a boolean mask of equal shape to the input, where the condition is satisfied - value 1 where satisfied, 0 otherwise<br>
     *
     * @param in Input (NUMERIC type)
     * @param condition Condition
     * @return output Boolean mask (NUMERIC type)
     */
    public SDVariable matchCondition(SDVariable in, Condition condition) {
        SDValidation.validateNumerical("matchCondition", "in", in);
        return new org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform(sd(),in, condition).outputVariable();
    }

    /**
     * Returns a boolean mask of equal shape to the input, where the condition is satisfied - value 1 where satisfied, 0 otherwise<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input (NUMERIC type)
     * @param condition Condition
     * @return output Boolean mask (NUMERIC type)
     */
    public SDVariable matchCondition(String name, SDVariable in, Condition condition) {
        SDValidation.validateNumerical("matchCondition", "in", in);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform(sd(),in, condition).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Returns a count of the number of elements that satisfy the condition<br>
     *
     * @param in Input (NUMERIC type)
     * @param condition Condition
     * @return output Number of elements that the condition is satisfied for (NUMERIC type)
     */
    public SDVariable matchConditionCount(SDVariable in, Condition condition) {
        SDValidation.validateNumerical("matchConditionCount", "in", in);
        return new org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition(sd(),in, condition).outputVariable();
    }

    /**
     * Returns a count of the number of elements that satisfy the condition<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input (NUMERIC type)
     * @param condition Condition
     * @return output Number of elements that the condition is satisfied for (NUMERIC type)
     */
    public SDVariable matchConditionCount(String name, SDVariable in, Condition condition) {
        SDValidation.validateNumerical("matchConditionCount", "in", in);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition(sd(),in, condition).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Returns a count of the number of elements that satisfy the condition (for each slice along the specified dimensions)<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param in Input variable (NUMERIC type)
     * @param condition Condition
     * @param keepDim If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output Number of elements that the condition is satisfied for (NUMERIC type)
     */
    public SDVariable matchConditionCount(SDVariable in, Condition condition, boolean keepDim,
                                          int... dimensions) {
        SDValidation.validateNumerical("matchConditionCount", "in", in);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition(sd(),in, condition, keepDim, dimensions).outputVariable();
    }

    /**
     * Returns a count of the number of elements that satisfy the condition (for each slice along the specified dimensions)<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @param condition Condition
     * @param keepDim If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output Number of elements that the condition is satisfied for (NUMERIC type)
     */
    public SDVariable matchConditionCount(String name, SDVariable in, Condition condition,
                                          boolean keepDim, int... dimensions) {
        SDValidation.validateNumerical("matchConditionCount", "in", in);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition(sd(),in, condition, keepDim, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Returns a count of the number of elements that satisfy the condition (for each slice along the specified dimensions)<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param in Input variable (NUMERIC type)
     * @param condition Condition
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output Number of elements that the condition is satisfied for (NUMERIC type)
     */
    public SDVariable matchConditionCount(SDVariable in, Condition condition, int... dimensions) {
        SDValidation.validateNumerical("matchConditionCount", "in", in);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition(sd(),in, condition, false, dimensions).outputVariable();
    }

    /**
     * Returns a count of the number of elements that satisfy the condition (for each slice along the specified dimensions)<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @param condition Condition
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output Number of elements that the condition is satisfied for (NUMERIC type)
     */
    public SDVariable matchConditionCount(String name, SDVariable in, Condition condition,
                                          int... dimensions) {
        SDValidation.validateNumerical("matchConditionCount", "in", in);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition(sd(),in, condition, false, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Max array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable max(SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("max", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.same.Max(sd(),x, keepDims, dimensions).outputVariable();
    }

    /**
     * Max array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable max(String name, SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("max", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.same.Max(sd(),x, keepDims, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Max array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable max(SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("max", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.same.Max(sd(),x, false, dimensions).outputVariable();
    }

    /**
     * Max array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable max(String name, SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("max", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.same.Max(sd(),x, false, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Element-wise maximum operation: out[i] = max(first[i], second[i])<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * @param first First input array (NUMERIC type)
     * @param second Second input array (NUMERIC type)
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable max(SDVariable first, SDVariable second) {
        SDValidation.validateNumerical("max", "first", first);
        SDValidation.validateNumerical("max", "second", second);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.Max(sd(),first, second).outputVariable();
    }

    /**
     * Element-wise maximum operation: out[i] = max(first[i], second[i])<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param first First input array (NUMERIC type)
     * @param second Second input array (NUMERIC type)
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable max(String name, SDVariable first, SDVariable second) {
        SDValidation.validateNumerical("max", "first", first);
        SDValidation.validateNumerical("max", "second", second);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.Max(sd(),first, second).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Mean (average) array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable mean(SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("mean", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.floating.Mean(sd(),x, keepDims, dimensions).outputVariable();
    }

    /**
     * Mean (average) array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable mean(String name, SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("mean", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.floating.Mean(sd(),x, keepDims, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Mean (average) array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable mean(SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("mean", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.floating.Mean(sd(),x, false, dimensions).outputVariable();
    }

    /**
     * Mean (average) array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable mean(String name, SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("mean", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.floating.Mean(sd(),x, false, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Minimum array reduction operation, optionally along specified dimensions. out = min(in)<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable min(SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("min", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.same.Min(sd(),x, keepDims, dimensions).outputVariable();
    }

    /**
     * Minimum array reduction operation, optionally along specified dimensions. out = min(in)<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable min(String name, SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("min", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.same.Min(sd(),x, keepDims, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Minimum array reduction operation, optionally along specified dimensions. out = min(in)<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable min(SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("min", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.same.Min(sd(),x, false, dimensions).outputVariable();
    }

    /**
     * Minimum array reduction operation, optionally along specified dimensions. out = min(in)<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable min(String name, SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("min", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.same.Min(sd(),x, false, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Element-wise minimum operation: out[i] = min(first[i], second[i])<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * @param first First input array (NUMERIC type)
     * @param second Second input array (NUMERIC type)
     * @return output Second input array (NUMERIC type)
     */
    public SDVariable min(SDVariable first, SDVariable second) {
        SDValidation.validateNumerical("min", "first", first);
        SDValidation.validateNumerical("min", "second", second);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.Min(sd(),first, second).outputVariable();
    }

    /**
     * Element-wise minimum operation: out[i] = min(first[i], second[i])<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param first First input array (NUMERIC type)
     * @param second Second input array (NUMERIC type)
     * @return output Second input array (NUMERIC type)
     */
    public SDVariable min(String name, SDVariable first, SDVariable second) {
        SDValidation.validateNumerical("min", "first", first);
        SDValidation.validateNumerical("min", "second", second);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.Min(sd(),first, second).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Matrix multiplication: out = mmul(x,y)<br>
     * Supports specifying transpose argument to perform operation such as mmul(a^T, b), etc.<br>
     *
     * @param x First input variable (NUMERIC type)
     * @param y Second input variable (NUMERIC type)
     * @param transposeX Transpose x (first argument)
     * @param transposeY Transpose y (second argument)
     * @param transposeZ Transpose result array
     * @return output  (NUMERIC type)
     */
    public SDVariable mmul(SDVariable x, SDVariable y, boolean transposeX, boolean transposeY,
                           boolean transposeZ) {
        SDValidation.validateNumerical("mmul", "x", x);
        SDValidation.validateNumerical("mmul", "y", y);
        return new org.nd4j.linalg.api.ops.impl.reduce.Mmul(sd(),x, y, transposeX, transposeY, transposeZ).outputVariable();
    }

    /**
     * Matrix multiplication: out = mmul(x,y)<br>
     * Supports specifying transpose argument to perform operation such as mmul(a^T, b), etc.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x First input variable (NUMERIC type)
     * @param y Second input variable (NUMERIC type)
     * @param transposeX Transpose x (first argument)
     * @param transposeY Transpose y (second argument)
     * @param transposeZ Transpose result array
     * @return output  (NUMERIC type)
     */
    public SDVariable mmul(String name, SDVariable x, SDVariable y, boolean transposeX,
                           boolean transposeY, boolean transposeZ) {
        SDValidation.validateNumerical("mmul", "x", x);
        SDValidation.validateNumerical("mmul", "y", y);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.Mmul(sd(),x, y, transposeX, transposeY, transposeZ).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Matrix multiplication: out = mmul(x,y)<br>
     * Supports specifying transpose argument to perform operation such as mmul(a^T, b), etc.<br>
     *
     * @param x First input variable (NUMERIC type)
     * @param y Second input variable (NUMERIC type)
     * @return output  (NUMERIC type)
     */
    public SDVariable mmul(SDVariable x, SDVariable y) {
        SDValidation.validateNumerical("mmul", "x", x);
        SDValidation.validateNumerical("mmul", "y", y);
        return new org.nd4j.linalg.api.ops.impl.reduce.Mmul(sd(),x, y, false, false, false).outputVariable();
    }

    /**
     * Matrix multiplication: out = mmul(x,y)<br>
     * Supports specifying transpose argument to perform operation such as mmul(a^T, b), etc.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x First input variable (NUMERIC type)
     * @param y Second input variable (NUMERIC type)
     * @return output  (NUMERIC type)
     */
    public SDVariable mmul(String name, SDVariable x, SDVariable y) {
        SDValidation.validateNumerical("mmul", "x", x);
        SDValidation.validateNumerical("mmul", "y", y);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.Mmul(sd(),x, y, false, false, false).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Not equals operation: elementwise x != y<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param x Input array (NUMERIC type)
     * @param y Double value argument to use in operation
     * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable neq(SDVariable x, double y) {
        SDValidation.validateNumerical("neq", "x", x);
        return new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarNotEquals(sd(),x, y).outputVariable();
    }

    /**
     * Not equals operation: elementwise x != y<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input array (NUMERIC type)
     * @param y Double value argument to use in operation
     * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable neq(String name, SDVariable x, double y) {
        SDValidation.validateNumerical("neq", "x", x);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarNotEquals(sd(),x, y).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Not equal to operation: elementwise x != y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param x Input 1 (NUMERIC type)
     * @param y Input 2 (NUMERIC type)
     * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable neq(SDVariable x, SDVariable y) {
        SDValidation.validateNumerical("neq", "x", x);
        SDValidation.validateNumerical("neq", "y", y);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.NotEqualTo(sd(),x, y).outputVariable();
    }

    /**
     * Not equal to operation: elementwise x != y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     *
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     *
     * Return boolean array with values true where satisfied, or false otherwise.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input 1 (NUMERIC type)
     * @param y Input 2 (NUMERIC type)
     * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
     */
    public SDVariable neq(String name, SDVariable x, SDVariable y) {
        SDValidation.validateNumerical("neq", "x", x);
        SDValidation.validateNumerical("neq", "y", y);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.NotEqualTo(sd(),x, y).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Norm1 (L1 norm) reduction operation: The output contains the L1 norm for each tensor/subset along the specified dimensions: <br>
     * out = sum_i abs(x[i])<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions dimensions to reduce over (Size: AtLeast(min=0))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable norm1(SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("norm1", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.floating.Norm1(sd(),x, keepDims, dimensions).outputVariable();
    }

    /**
     * Norm1 (L1 norm) reduction operation: The output contains the L1 norm for each tensor/subset along the specified dimensions: <br>
     * out = sum_i abs(x[i])<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions dimensions to reduce over (Size: AtLeast(min=0))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable norm1(String name, SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("norm1", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.floating.Norm1(sd(),x, keepDims, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Norm1 (L1 norm) reduction operation: The output contains the L1 norm for each tensor/subset along the specified dimensions: <br>
     * out = sum_i abs(x[i])<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param dimensions dimensions to reduce over (Size: AtLeast(min=0))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable norm1(SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("norm1", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.floating.Norm1(sd(),x, false, dimensions).outputVariable();
    }

    /**
     * Norm1 (L1 norm) reduction operation: The output contains the L1 norm for each tensor/subset along the specified dimensions: <br>
     * out = sum_i abs(x[i])<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param dimensions dimensions to reduce over (Size: AtLeast(min=0))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable norm1(String name, SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("norm1", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.floating.Norm1(sd(),x, false, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Norm2 (L2 norm) reduction operation: The output contains the L2 norm for each tensor/subset along the specified dimensions:<br>
     * out = sqrt(sum_i x[i]^2)<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions dimensions dimensions to reduce over (Size: AtLeast(min=0))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable norm2(SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("norm2", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.floating.Norm2(sd(),x, keepDims, dimensions).outputVariable();
    }

    /**
     * Norm2 (L2 norm) reduction operation: The output contains the L2 norm for each tensor/subset along the specified dimensions:<br>
     * out = sqrt(sum_i x[i]^2)<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions dimensions dimensions to reduce over (Size: AtLeast(min=0))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable norm2(String name, SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("norm2", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.floating.Norm2(sd(),x, keepDims, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Norm2 (L2 norm) reduction operation: The output contains the L2 norm for each tensor/subset along the specified dimensions:<br>
     * out = sqrt(sum_i x[i]^2)<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param dimensions dimensions dimensions to reduce over (Size: AtLeast(min=0))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable norm2(SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("norm2", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.floating.Norm2(sd(),x, false, dimensions).outputVariable();
    }

    /**
     * Norm2 (L2 norm) reduction operation: The output contains the L2 norm for each tensor/subset along the specified dimensions:<br>
     * out = sqrt(sum_i x[i]^2)<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param dimensions dimensions dimensions to reduce over (Size: AtLeast(min=0))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable norm2(String name, SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("norm2", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.floating.Norm2(sd(),x, false, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Max norm (infinity norm) reduction operation: The output contains the max norm for each tensor/subset along the<br>
     * specified dimensions:<br>
     * out = max(abs(x[i]))<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions dimensions to reduce over (Size: AtLeast(min=0))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable normmax(SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("normmax", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.floating.NormMax(sd(),x, keepDims, dimensions).outputVariable();
    }

    /**
     * Max norm (infinity norm) reduction operation: The output contains the max norm for each tensor/subset along the<br>
     * specified dimensions:<br>
     * out = max(abs(x[i]))<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions dimensions to reduce over (Size: AtLeast(min=0))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable normmax(String name, SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("normmax", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.floating.NormMax(sd(),x, keepDims, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Max norm (infinity norm) reduction operation: The output contains the max norm for each tensor/subset along the<br>
     * specified dimensions:<br>
     * out = max(abs(x[i]))<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param dimensions dimensions to reduce over (Size: AtLeast(min=0))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable normmax(SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("normmax", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.floating.NormMax(sd(),x, false, dimensions).outputVariable();
    }

    /**
     * Max norm (infinity norm) reduction operation: The output contains the max norm for each tensor/subset along the<br>
     * specified dimensions:<br>
     * out = max(abs(x[i]))<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param dimensions dimensions to reduce over (Size: AtLeast(min=0))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable normmax(String name, SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("normmax", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.floating.NormMax(sd(),x, false, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Convert the array to a one-hot array with walues and  for each entry<br>
     * If input has shape [ a, ..., n] then output has shape [ a, ..., n, depth],<br>
     * with {out[i, ..., j, in[i,...,j]]  with other values being set to<br>
     *
     * @param indices Indices - value 0 to depth-1 (NUMERIC type)
     * @param depth Number of classes
     * @param axis
     * @param on
     * @param off
     * @param dataType Output data type
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable oneHot(SDVariable indices, int depth, int axis, double on, double off,
                             DataType dataType) {
        SDValidation.validateNumerical("oneHot", "indices", indices);
        return new org.nd4j.linalg.api.ops.impl.shape.OneHot(sd(),indices, depth, axis, on, off, dataType).outputVariable();
    }

    /**
     * Convert the array to a one-hot array with walues and  for each entry<br>
     * If input has shape [ a, ..., n] then output has shape [ a, ..., n, depth],<br>
     * with {out[i, ..., j, in[i,...,j]]  with other values being set to<br>
     *
     * @param name name May be null. Name for the output variable
     * @param indices Indices - value 0 to depth-1 (NUMERIC type)
     * @param depth Number of classes
     * @param axis
     * @param on
     * @param off
     * @param dataType Output data type
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable oneHot(String name, SDVariable indices, int depth, int axis, double on,
                             double off, DataType dataType) {
        SDValidation.validateNumerical("oneHot", "indices", indices);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.OneHot(sd(),indices, depth, axis, on, off, dataType).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Convert the array to a one-hot array with walues and  for each entry<br>
     * If input has shape [ a, ..., n] then output has shape [ a, ..., n, depth],<br>
     * with {out[i, ..., j, in[i,...,j]]  with other values being set to<br>
     *
     * @param indices Indices - value 0 to depth-1 (NUMERIC type)
     * @param depth Number of classes
     * @param axis
     * @param on
     * @param off
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable oneHot(SDVariable indices, int depth, int axis, double on, double off) {
        SDValidation.validateNumerical("oneHot", "indices", indices);
        return new org.nd4j.linalg.api.ops.impl.shape.OneHot(sd(),indices, depth, axis, on, off, DataType.FLOAT).outputVariable();
    }

    /**
     * Convert the array to a one-hot array with walues and  for each entry<br>
     * If input has shape [ a, ..., n] then output has shape [ a, ..., n, depth],<br>
     * with {out[i, ..., j, in[i,...,j]]  with other values being set to<br>
     *
     * @param name name May be null. Name for the output variable
     * @param indices Indices - value 0 to depth-1 (NUMERIC type)
     * @param depth Number of classes
     * @param axis
     * @param on
     * @param off
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable oneHot(String name, SDVariable indices, int depth, int axis, double on,
                             double off) {
        SDValidation.validateNumerical("oneHot", "indices", indices);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.OneHot(sd(),indices, depth, axis, on, off, DataType.FLOAT).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Convert the array to a one-hot array with walues 0 and 1 for each entry<br>
     * If input has shape [ a, ..., n] then output has shape [ a, ..., n, depth],<br>
     * with out[i, ..., j, in[i,...,j]] = 1 with other values being set to 0<br>
     * see oneHot(SDVariable, int, int, double, double)<br>
     *
     * @param indices Indices - value 0 to depth-1 (NUMERIC type)
     * @param depth Number of classes
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable oneHot(SDVariable indices, int depth) {
        SDValidation.validateNumerical("oneHot", "indices", indices);
        return new org.nd4j.linalg.api.ops.impl.shape.OneHot(sd(),indices, depth).outputVariable();
    }

    /**
     * Convert the array to a one-hot array with walues 0 and 1 for each entry<br>
     * If input has shape [ a, ..., n] then output has shape [ a, ..., n, depth],<br>
     * with out[i, ..., j, in[i,...,j]] = 1 with other values being set to 0<br>
     * see oneHot(SDVariable, int, int, double, double)<br>
     *
     * @param name name May be null. Name for the output variable
     * @param indices Indices - value 0 to depth-1 (NUMERIC type)
     * @param depth Number of classes
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable oneHot(String name, SDVariable indices, int depth) {
        SDValidation.validateNumerical("oneHot", "indices", indices);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.OneHot(sd(),indices, depth).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Return a variable of all 1s, with the same shape as the input variable. Note that this is dynamic:<br>
     * if the input shape changes in later execution, the returned variable's shape will also be updated<br>
     *
     * @param input Input INDArray  (NUMERIC type)
     * @return output A new INDArray  with the same (dynamic) shape as the input (NUMERIC type)
     */
    public SDVariable onesLike(SDVariable input) {
        SDValidation.validateNumerical("onesLike", "input", input);
        return new org.nd4j.linalg.api.ops.impl.shape.OnesLike(sd(),input).outputVariable();
    }

    /**
     * Return a variable of all 1s, with the same shape as the input variable. Note that this is dynamic:<br>
     * if the input shape changes in later execution, the returned variable's shape will also be updated<br>
     *
     * @param name name May be null. Name for the output variable
     * @param input Input INDArray  (NUMERIC type)
     * @return output A new INDArray  with the same (dynamic) shape as the input (NUMERIC type)
     */
    public SDVariable onesLike(String name, SDVariable input) {
        SDValidation.validateNumerical("onesLike", "input", input);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.OnesLike(sd(),input).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * As per onesLike(String, SDVariable) but the output datatype may be specified<br>
     *
     * @param input  (NUMERIC type)
     * @param dataType
     * @return output  (NUMERIC type)
     */
    public SDVariable onesLike(SDVariable input, DataType dataType) {
        SDValidation.validateNumerical("onesLike", "input", input);
        return new org.nd4j.linalg.api.ops.impl.shape.OnesLike(sd(),input, dataType).outputVariable();
    }

    /**
     * As per onesLike(String, SDVariable) but the output datatype may be specified<br>
     *
     * @param name name May be null. Name for the output variable
     * @param input  (NUMERIC type)
     * @param dataType
     * @return output  (NUMERIC type)
     */
    public SDVariable onesLike(String name, SDVariable input, DataType dataType) {
        SDValidation.validateNumerical("onesLike", "input", input);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.OnesLike(sd(),input, dataType).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Array permutation operation: permute the dimensions according to the specified permutation indices.<br>
     * Example: if input has shape [a,b,c] and dimensions = [2,0,1] the output has shape [c,a,b]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param dimensions  (Size: AtLeast(min=0))
     * @return output Output variable (permuted input) (NUMERIC type)
     */
    public SDVariable permute(SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("permute", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.shape.Permute(sd(),x, dimensions).outputVariable();
    }

    // TODO: generate me
    public SDVariable permute(SDVariable x, SDVariable dimensions) {
        SDValidation.validateNumerical("permute", "x", x);
        SDValidation.validateNumerical("permute", "dimensions", dimensions);
        return new org.nd4j.linalg.api.ops.impl.shape.Permute(sd(),x, dimensions).outputVariable();
    }

    /**
     * Array permutation operation: permute the dimensions according to the specified permutation indices.<br>
     * Example: if input has shape [a,b,c] and dimensions = [2,0,1] the output has shape [c,a,b]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param dimensions  (Size: AtLeast(min=0))
     * @return output Output variable (permuted input) (NUMERIC type)
     */
    public SDVariable permute(String name, SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("permute", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Permute(sd(),x, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Product array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output  (NUMERIC type)
     */
    public SDVariable prod(SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("prod", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.same.Prod(sd(),x, keepDims, dimensions).outputVariable();
    }

    /**
     * Product array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output  (NUMERIC type)
     */
    public SDVariable prod(String name, SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("prod", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.same.Prod(sd(),x, keepDims, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Product array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output  (NUMERIC type)
     */
    public SDVariable prod(SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("prod", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.same.Prod(sd(),x, false, dimensions).outputVariable();
    }

    /**
     * Product array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output  (NUMERIC type)
     */
    public SDVariable prod(String name, SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("prod", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.same.Prod(sd(),x, false, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Create a new variable with a 1d array, where the values start at from and increment by step<br>
     * up to (but not including) limit.<br>
     * For example, range(1.0, 3.0, 0.5) will return [1.0, 1.5, 2.0, 2.5]<br>
     *
     * @param from Initial/smallest value
     * @param to Largest value (exclusive)
     * @param step Step size
     * @param dataType
     * @return output INDArray  with the specified values (NUMERIC type)
     */
    public SDVariable range(double from, double to, double step, DataType dataType) {
        return new org.nd4j.linalg.api.ops.random.impl.Range(sd(),from, to, step, dataType).outputVariable();
    }

    /**
     * Create a new variable with a 1d array, where the values start at from and increment by step<br>
     * up to (but not including) limit.<br>
     * For example, range(1.0, 3.0, 0.5) will return [1.0, 1.5, 2.0, 2.5]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param from Initial/smallest value
     * @param to Largest value (exclusive)
     * @param step Step size
     * @param dataType
     * @return output INDArray  with the specified values (NUMERIC type)
     */
    public SDVariable range(String name, double from, double to, double step, DataType dataType) {
        SDVariable out =  new org.nd4j.linalg.api.ops.random.impl.Range(sd(),from, to, step, dataType).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Returns the rank (number of dimensions, i.e., length(shape)) of the specified INDArray  as a 0D scalar variable<br>
     *
     * @param in Input variable (NUMERIC type)
     * @return output (scalar) output variable with value equal to the rank of the input variable (NUMERIC type)
     */
    public SDVariable rank(SDVariable in) {
        SDValidation.validateNumerical("rank", "in", in);
        return new org.nd4j.linalg.api.ops.impl.shape.Rank(sd(),in).outputVariable();
    }

    /**
     * Returns the rank (number of dimensions, i.e., length(shape)) of the specified INDArray  as a 0D scalar variable<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @return output (scalar) output variable with value equal to the rank of the input variable (NUMERIC type)
     */
    public SDVariable rank(String name, SDVariable in) {
        SDValidation.validateNumerical("rank", "in", in);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Rank(sd(),in).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Element-wise replace where condition:<br>
     * out[i] = from[i] if condition(update[i]) is satisfied, or<br>
     * out[i] = update[i] if condition(update[i]) is NOT satisfied<br>
     *
     * @param update Source array (NUMERIC type)
     * @param from Replacement values array (used conditionally). Must be same shape as 'update' array (NUMERIC type)
     * @param condition Condition to check on update array elements
     * @return output New array with values replaced where condition is satisfied (NUMERIC type)
     */
    public SDVariable replaceWhere(SDVariable update, SDVariable from, Condition condition) {
        SDValidation.validateNumerical("replaceWhere", "update", update);
        SDValidation.validateNumerical("replaceWhere", "from", from);
        return new org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndReplace(sd(),update, from, condition).outputVariable();
    }

    /**
     * Element-wise replace where condition:<br>
     * out[i] = from[i] if condition(update[i]) is satisfied, or<br>
     * out[i] = update[i] if condition(update[i]) is NOT satisfied<br>
     *
     * @param name name May be null. Name for the output variable
     * @param update Source array (NUMERIC type)
     * @param from Replacement values array (used conditionally). Must be same shape as 'update' array (NUMERIC type)
     * @param condition Condition to check on update array elements
     * @return output New array with values replaced where condition is satisfied (NUMERIC type)
     */
    public SDVariable replaceWhere(String name, SDVariable update, SDVariable from,
                                   Condition condition) {
        SDValidation.validateNumerical("replaceWhere", "update", update);
        SDValidation.validateNumerical("replaceWhere", "from", from);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndReplace(sd(),update, from, condition).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Reshape the input variable to the specified (fixed) shape. The output variable will have the same values as the<br>
     * input, but with the specified shape.<br>
     * Note that prod(shape) must match length(input) == prod(input.shape)<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param shape New shape for variable (NUMERIC type)
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable reshape(SDVariable x, SDVariable shape) {
        SDValidation.validateNumerical("reshape", "x", x);
        SDValidation.validateNumerical("reshape", "shape", shape);
        return new org.nd4j.linalg.api.ops.impl.shape.Reshape(sd(),x, shape).outputVariable();
    }

    // TODO: generate me
    public SDVariable reshape(SDVariable x, long... shape) {
        SDValidation.validateNumerical("reshape", "x", x);
        //SDValidation.validateNumerical("reshape", "shape", shape);
        return new org.nd4j.linalg.api.ops.impl.shape.Reshape(sd(),x, shape).outputVariable();
    }

    /**
     * Reshape the input variable to the specified (fixed) shape. The output variable will have the same values as the<br>
     * input, but with the specified shape.<br>
     * Note that prod(shape) must match length(input) == prod(input.shape)<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param shape New shape for variable (NUMERIC type)
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable reshape(String name, SDVariable x, SDVariable shape) {
        SDValidation.validateNumerical("reshape", "x", x);
        SDValidation.validateNumerical("reshape", "shape", shape);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Reshape(sd(),x, shape).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Reverse the values of an array for the specified dimensions<br>
     * If input is:<br>
     * [ 1, 2, 3]<br>
     * [ 4, 5, 6]<br>
     * then<br>
     * reverse(in, 0):<br>
     * [3, 2, 1]<br>
     * [6, 5, 4]<br>
     * reverse(in, 1):<br>
     * [4, 5, 6]<br>
     * [1, 2 3]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param dimensions Input variable (Size: AtLeast(min=0))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable reverse(SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("reverse", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.Reverse(sd(),x, dimensions).outputVariable();
    }

    /**
     * Reverse the values of an array for the specified dimensions<br>
     * If input is:<br>
     * [ 1, 2, 3]<br>
     * [ 4, 5, 6]<br>
     * then<br>
     * reverse(in, 0):<br>
     * [3, 2, 1]<br>
     * [6, 5, 4]<br>
     * reverse(in, 1):<br>
     * [4, 5, 6]<br>
     * [1, 2 3]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param dimensions Input variable (Size: AtLeast(min=0))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable reverse(String name, SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("reverse", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.Reverse(sd(),x, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Reverse sequence op: for each slice along dimension seqDimension, the first seqLength values are reversed<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param seq_lengths Length of the sequences (INT type)
     * @param seqDim Sequence dimension
     * @param batchDim Batch dimension
     * @return output Reversed sequences (NUMERIC type)
     */
    public SDVariable reverseSequence(SDVariable x, SDVariable seq_lengths, int seqDim,
                                      int batchDim) {
        SDValidation.validateNumerical("reverseSequence", "x", x);
        SDValidation.validateInteger("reverseSequence", "seq_lengths", seq_lengths);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.ReverseSequence(sd(),x, seq_lengths, seqDim, batchDim).outputVariable();
    }

    /**
     * Reverse sequence op: for each slice along dimension seqDimension, the first seqLength values are reversed<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param seq_lengths Length of the sequences (INT type)
     * @param seqDim Sequence dimension
     * @param batchDim Batch dimension
     * @return output Reversed sequences (NUMERIC type)
     */
    public SDVariable reverseSequence(String name, SDVariable x, SDVariable seq_lengths, int seqDim,
                                      int batchDim) {
        SDValidation.validateNumerical("reverseSequence", "x", x);
        SDValidation.validateInteger("reverseSequence", "seq_lengths", seq_lengths);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.ReverseSequence(sd(),x, seq_lengths, seqDim, batchDim).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Reverse sequence op: for each slice along dimension seqDimension, the first seqLength values are reversed<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param seq_lengths Length of the sequences (INT type)
     * @return output Reversed sequences (NUMERIC type)
     */
    public SDVariable reverseSequence(SDVariable x, SDVariable seq_lengths) {
        SDValidation.validateNumerical("reverseSequence", "x", x);
        SDValidation.validateInteger("reverseSequence", "seq_lengths", seq_lengths);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.ReverseSequence(sd(),x, seq_lengths, -1, 0).outputVariable();
    }

    /**
     * Reverse sequence op: for each slice along dimension seqDimension, the first seqLength values are reversed<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param seq_lengths Length of the sequences (INT type)
     * @return output Reversed sequences (NUMERIC type)
     */
    public SDVariable reverseSequence(String name, SDVariable x, SDVariable seq_lengths) {
        SDValidation.validateNumerical("reverseSequence", "x", x);
        SDValidation.validateInteger("reverseSequence", "seq_lengths", seq_lengths);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.ReverseSequence(sd(),x, seq_lengths, -1, 0).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Element-wise scalar floor division operation: out = floorDiv(in, value).<br>
     *
     * @param in Input variable (NUMERIC type)
     * @param value Scalar value to compare
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable scalarFloorDiv(SDVariable in, double value) {
        SDValidation.validateNumerical("scalarFloorDiv", "in", in);
        return new org.nd4j.linalg.api.ops.impl.scalar.ScalarDivision(sd(),in, value).outputVariable();
    }

    /**
     * Element-wise scalar floor division operation: out = floorDiv(in, value).<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @param value Scalar value to compare
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable scalarFloorDiv(String name, SDVariable in, double value) {
        SDValidation.validateNumerical("scalarFloorDiv", "in", in);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.ScalarDivision(sd(),in, value).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Element-wise scalar floor modulus operation: out = floorMod(in, value).<br>
     * i.e., returns the remainder after division by 'value'<br>
     *
     * @param in Input variable (NUMERIC type)
     * @param value Scalar value to compare
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable scalarFloorMod(SDVariable in, double value) {
        SDValidation.validateNumerical("scalarFloorMod", "in", in);
        return new org.nd4j.linalg.api.ops.impl.scalar.ScalarFMod(sd(),in, value).outputVariable();
    }

    /**
     * Element-wise scalar floor modulus operation: out = floorMod(in, value).<br>
     * i.e., returns the remainder after division by 'value'<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @param value Scalar value to compare
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable scalarFloorMod(String name, SDVariable in, double value) {
        SDValidation.validateNumerical("scalarFloorMod", "in", in);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.ScalarFMod(sd(),in, value).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Element-wise scalar maximum operation: out = max(in, value)<br>
     *
     * @param in Input variable (NUMERIC type)
     * @param value Scalar value to compare
     * @return output Scalar value to compare (NUMERIC type)
     */
    public SDVariable scalarMax(SDVariable in, double value) {
        SDValidation.validateNumerical("scalarMax", "in", in);
        return new org.nd4j.linalg.api.ops.impl.scalar.ScalarMax(sd(),in, value).outputVariable();
    }

    /**
     * Element-wise scalar maximum operation: out = max(in, value)<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @param value Scalar value to compare
     * @return output Scalar value to compare (NUMERIC type)
     */
    public SDVariable scalarMax(String name, SDVariable in, double value) {
        SDValidation.validateNumerical("scalarMax", "in", in);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.ScalarMax(sd(),in, value).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Element-wise scalar minimum operation: out = min(in, value)<br>
     *
     * @param in Input variable (NUMERIC type)
     * @param value Scalar value to compare
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable scalarMin(SDVariable in, double value) {
        SDValidation.validateNumerical("scalarMin", "in", in);
        return new org.nd4j.linalg.api.ops.impl.scalar.ScalarMin(sd(),in, value).outputVariable();
    }

    /**
     * Element-wise scalar minimum operation: out = min(in, value)<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @param value Scalar value to compare
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable scalarMin(String name, SDVariable in, double value) {
        SDValidation.validateNumerical("scalarMin", "in", in);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.ScalarMin(sd(),in, value).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Return a variable with equal shape to the input, but all elements set to value 'set'<br>
     *
     * @param in Input variable (NUMERIC type)
     * @param set Value to set
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable scalarSet(SDVariable in, double set) {
        SDValidation.validateNumerical("scalarSet", "in", in);
        return new org.nd4j.linalg.api.ops.impl.scalar.ScalarSet(sd(),in, set).outputVariable();
    }

    /**
     * Return a variable with equal shape to the input, but all elements set to value 'set'<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @param set Value to set
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable scalarSet(String name, SDVariable in, double set) {
        SDValidation.validateNumerical("scalarSet", "in", in);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.ScalarSet(sd(),in, set).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Scatter addition operation.<br>
     *
     * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
     *
     * @param ref Initial/source variable (NUMERIC type)
     * @param indices Indices array (NUMERIC type)
     * @param updates Updates to add to the initial/source array (NUMERIC type)
     * @return output The updated variable (NUMERIC type)
     */
    public SDVariable scatterAdd(SDVariable ref, SDVariable indices, SDVariable updates) {
        SDValidation.validateNumerical("scatterAdd", "ref", ref);
        SDValidation.validateNumerical("scatterAdd", "indices", indices);
        SDValidation.validateNumerical("scatterAdd", "updates", updates);
        return new org.nd4j.linalg.api.ops.impl.scatter.ScatterAdd(sd(),ref, indices, updates).outputVariable();
    }

    /**
     * Scatter addition operation.<br>
     *
     * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
     *
     * @param name name May be null. Name for the output variable
     * @param ref Initial/source variable (NUMERIC type)
     * @param indices Indices array (NUMERIC type)
     * @param updates Updates to add to the initial/source array (NUMERIC type)
     * @return output The updated variable (NUMERIC type)
     */
    public SDVariable scatterAdd(String name, SDVariable ref, SDVariable indices,
                                 SDVariable updates) {
        SDValidation.validateNumerical("scatterAdd", "ref", ref);
        SDValidation.validateNumerical("scatterAdd", "indices", indices);
        SDValidation.validateNumerical("scatterAdd", "updates", updates);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scatter.ScatterAdd(sd(),ref, indices, updates).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Scatter division operation.<br>
     *
     * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
     *
     * @param ref Initial/source variable (NUMERIC type)
     * @param indices Indices array (NUMERIC type)
     * @param updates Updates to add to the initial/source array (NUMERIC type)
     * @return output The updated variable (NUMERIC type)
     */
    public SDVariable scatterDiv(SDVariable ref, SDVariable indices, SDVariable updates) {
        SDValidation.validateNumerical("scatterDiv", "ref", ref);
        SDValidation.validateNumerical("scatterDiv", "indices", indices);
        SDValidation.validateNumerical("scatterDiv", "updates", updates);
        return new org.nd4j.linalg.api.ops.impl.scatter.ScatterDiv(sd(),ref, indices, updates).outputVariable();
    }

    /**
     * Scatter division operation.<br>
     *
     * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
     *
     * @param name name May be null. Name for the output variable
     * @param ref Initial/source variable (NUMERIC type)
     * @param indices Indices array (NUMERIC type)
     * @param updates Updates to add to the initial/source array (NUMERIC type)
     * @return output The updated variable (NUMERIC type)
     */
    public SDVariable scatterDiv(String name, SDVariable ref, SDVariable indices,
                                 SDVariable updates) {
        SDValidation.validateNumerical("scatterDiv", "ref", ref);
        SDValidation.validateNumerical("scatterDiv", "indices", indices);
        SDValidation.validateNumerical("scatterDiv", "updates", updates);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scatter.ScatterDiv(sd(),ref, indices, updates).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Scatter max operation.<br>
     *
     * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
     *
     * @param ref Initial/source variable (NUMERIC type)
     * @param indices Indices array (NUMERIC type)
     * @param updates Updates to add to the initial/source array (NUMERIC type)
     * @return output The updated variable (NUMERIC type)
     */
    public SDVariable scatterMax(SDVariable ref, SDVariable indices, SDVariable updates) {
        SDValidation.validateNumerical("scatterMax", "ref", ref);
        SDValidation.validateNumerical("scatterMax", "indices", indices);
        SDValidation.validateNumerical("scatterMax", "updates", updates);
        return new org.nd4j.linalg.api.ops.impl.scatter.ScatterMax(sd(),ref, indices, updates).outputVariable();
    }

    /**
     * Scatter max operation.<br>
     *
     * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
     *
     * @param name name May be null. Name for the output variable
     * @param ref Initial/source variable (NUMERIC type)
     * @param indices Indices array (NUMERIC type)
     * @param updates Updates to add to the initial/source array (NUMERIC type)
     * @return output The updated variable (NUMERIC type)
     */
    public SDVariable scatterMax(String name, SDVariable ref, SDVariable indices,
                                 SDVariable updates) {
        SDValidation.validateNumerical("scatterMax", "ref", ref);
        SDValidation.validateNumerical("scatterMax", "indices", indices);
        SDValidation.validateNumerical("scatterMax", "updates", updates);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scatter.ScatterMax(sd(),ref, indices, updates).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Scatter min operation.<br>
     *
     * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
     *
     * @param ref Initial/source variable (NUMERIC type)
     * @param indices Indices array (NUMERIC type)
     * @param updates Updates to add to the initial/source array (NUMERIC type)
     * @return output The updated variable (NUMERIC type)
     */
    public SDVariable scatterMin(SDVariable ref, SDVariable indices, SDVariable updates) {
        SDValidation.validateNumerical("scatterMin", "ref", ref);
        SDValidation.validateNumerical("scatterMin", "indices", indices);
        SDValidation.validateNumerical("scatterMin", "updates", updates);
        return new org.nd4j.linalg.api.ops.impl.scatter.ScatterMin(sd(),ref, indices, updates).outputVariable();
    }

    /**
     * Scatter min operation.<br>
     *
     * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
     *
     * @param name name May be null. Name for the output variable
     * @param ref Initial/source variable (NUMERIC type)
     * @param indices Indices array (NUMERIC type)
     * @param updates Updates to add to the initial/source array (NUMERIC type)
     * @return output The updated variable (NUMERIC type)
     */
    public SDVariable scatterMin(String name, SDVariable ref, SDVariable indices,
                                 SDVariable updates) {
        SDValidation.validateNumerical("scatterMin", "ref", ref);
        SDValidation.validateNumerical("scatterMin", "indices", indices);
        SDValidation.validateNumerical("scatterMin", "updates", updates);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scatter.ScatterMin(sd(),ref, indices, updates).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Scatter multiplication operation.<br>
     *
     * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
     *
     * @param ref Initial/source variable (NUMERIC type)
     * @param indices Indices array (NUMERIC type)
     * @param updates Updates to add to the initial/source array (NUMERIC type)
     * @return output The updated variable (NUMERIC type)
     */
    public SDVariable scatterMul(SDVariable ref, SDVariable indices, SDVariable updates) {
        SDValidation.validateNumerical("scatterMul", "ref", ref);
        SDValidation.validateNumerical("scatterMul", "indices", indices);
        SDValidation.validateNumerical("scatterMul", "updates", updates);
        return new org.nd4j.linalg.api.ops.impl.scatter.ScatterMul(sd(),ref, indices, updates).outputVariable();
    }

    /**
     * Scatter multiplication operation.<br>
     *
     * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
     *
     * @param name name May be null. Name for the output variable
     * @param ref Initial/source variable (NUMERIC type)
     * @param indices Indices array (NUMERIC type)
     * @param updates Updates to add to the initial/source array (NUMERIC type)
     * @return output The updated variable (NUMERIC type)
     */
    public SDVariable scatterMul(String name, SDVariable ref, SDVariable indices,
                                 SDVariable updates) {
        SDValidation.validateNumerical("scatterMul", "ref", ref);
        SDValidation.validateNumerical("scatterMul", "indices", indices);
        SDValidation.validateNumerical("scatterMul", "updates", updates);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scatter.ScatterMul(sd(),ref, indices, updates).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Scatter subtraction operation.<br>
     *
     * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
     *
     * @param ref Initial/source variable (NUMERIC type)
     * @param indices Indices array (NUMERIC type)
     * @param updates Updates to add to the initial/source array (NUMERIC type)
     * @return output The updated variable (NUMERIC type)
     */
    public SDVariable scatterSub(SDVariable ref, SDVariable indices, SDVariable updates) {
        SDValidation.validateNumerical("scatterSub", "ref", ref);
        SDValidation.validateNumerical("scatterSub", "indices", indices);
        SDValidation.validateNumerical("scatterSub", "updates", updates);
        return new org.nd4j.linalg.api.ops.impl.scatter.ScatterSub(sd(),ref, indices, updates).outputVariable();
    }

    /**
     * Scatter subtraction operation.<br>
     *
     * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
     *
     * @param name name May be null. Name for the output variable
     * @param ref Initial/source variable (NUMERIC type)
     * @param indices Indices array (NUMERIC type)
     * @param updates Updates to add to the initial/source array (NUMERIC type)
     * @return output The updated variable (NUMERIC type)
     */
    public SDVariable scatterSub(String name, SDVariable ref, SDVariable indices,
                                 SDVariable updates) {
        SDValidation.validateNumerical("scatterSub", "ref", ref);
        SDValidation.validateNumerical("scatterSub", "indices", indices);
        SDValidation.validateNumerical("scatterSub", "updates", updates);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scatter.ScatterSub(sd(),ref, indices, updates).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Scatter update operation.<br>
     *
     * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
     *
     * @param ref Initial/source variable (NUMERIC type)
     * @param indices Indices array (NUMERIC type)
     * @param updates Updates to add to the initial/source array (NUMERIC type)
     * @return output The updated variable (NUMERIC type)
     */
    public SDVariable scatterUpdate(SDVariable ref, SDVariable indices, SDVariable updates) {
        SDValidation.validateNumerical("scatterUpdate", "ref", ref);
        SDValidation.validateNumerical("scatterUpdate", "indices", indices);
        SDValidation.validateNumerical("scatterUpdate", "updates", updates);
        return new org.nd4j.linalg.api.ops.impl.scatter.ScatterUpdate(sd(),ref, indices, updates).outputVariable();
    }

    /**
     * Scatter update operation.<br>
     *
     * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
     *
     * @param name name May be null. Name for the output variable
     * @param ref Initial/source variable (NUMERIC type)
     * @param indices Indices array (NUMERIC type)
     * @param updates Updates to add to the initial/source array (NUMERIC type)
     * @return output The updated variable (NUMERIC type)
     */
    public SDVariable scatterUpdate(String name, SDVariable ref, SDVariable indices,
                                    SDVariable updates) {
        SDValidation.validateNumerical("scatterUpdate", "ref", ref);
        SDValidation.validateNumerical("scatterUpdate", "indices", indices);
        SDValidation.validateNumerical("scatterUpdate", "updates", updates);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.scatter.ScatterUpdate(sd(),ref, indices, updates).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Segment max operation.<br>
     *
     * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
     * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
     * then output = [6, 9, 8] = [op(3,6), op(1,4,9), op(2,8)]<br>
     * Note that the segment IDs must be sorted from smallest to largest segment.<br>
     * See {unsortedSegment (String, SDVariable, SDVariable, int) ops<br>
     * for the same op without this sorted requirement<br>
     *
     * @param data Data to perform segment max on (NDARRAY type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @return output Segment output (NUMERIC type)
     */
    public SDVariable segmentMax(SDVariable data, SDVariable segmentIds) {
        SDValidation.validateNumerical("segmentMax", "segmentIds", segmentIds);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentMax(sd(),data, segmentIds).outputVariable();
    }

    /**
     * Segment max operation.<br>
     *
     * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
     * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
     * then output = [6, 9, 8] = [op(3,6), op(1,4,9), op(2,8)]<br>
     * Note that the segment IDs must be sorted from smallest to largest segment.<br>
     * See {unsortedSegment (String, SDVariable, SDVariable, int) ops<br>
     * for the same op without this sorted requirement<br>
     *
     * @param name name May be null. Name for the output variable
     * @param data Data to perform segment max on (NDARRAY type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @return output Segment output (NUMERIC type)
     */
    public SDVariable segmentMax(String name, SDVariable data, SDVariable segmentIds) {
        SDValidation.validateNumerical("segmentMax", "segmentIds", segmentIds);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentMax(sd(),data, segmentIds).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Segment mean operation.<br>
     *
     * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
     * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
     * then output = [6, 9, 8] = [op(3,6), op(1,4,9), op(2,8)]<br>
     * Note that the segment IDs must be sorted from smallest to largest segment.<br>
     * See {unsortedSegment (String, SDVariable, SDVariable, int) ops<br>
     * for the same op without this sorted requirement<br>
     *
     * @param data Data to perform segment max on (NDARRAY type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @return output Segment output (NUMERIC type)
     */
    public SDVariable segmentMean(SDVariable data, SDVariable segmentIds) {
        SDValidation.validateNumerical("segmentMean", "segmentIds", segmentIds);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentMean(sd(),data, segmentIds).outputVariable();
    }

    /**
     * Segment mean operation.<br>
     *
     * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
     * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
     * then output = [6, 9, 8] = [op(3,6), op(1,4,9), op(2,8)]<br>
     * Note that the segment IDs must be sorted from smallest to largest segment.<br>
     * See {unsortedSegment (String, SDVariable, SDVariable, int) ops<br>
     * for the same op without this sorted requirement<br>
     *
     * @param name name May be null. Name for the output variable
     * @param data Data to perform segment max on (NDARRAY type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @return output Segment output (NUMERIC type)
     */
    public SDVariable segmentMean(String name, SDVariable data, SDVariable segmentIds) {
        SDValidation.validateNumerical("segmentMean", "segmentIds", segmentIds);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentMean(sd(),data, segmentIds).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Segment min operation.<br>
     *
     * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
     * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
     * then output = [6, 9, 8] = [op(3,6), op(1,4,9), op(2,8)]<br>
     * Note that the segment IDs must be sorted from smallest to largest segment.<br>
     * See {unsortedSegment (String, SDVariable, SDVariable, int) ops<br>
     * for the same op without this sorted requirement<br>
     *
     * @param data Data to perform segment max on (NDARRAY type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @return output Segment output (NUMERIC type)
     */
    public SDVariable segmentMin(SDVariable data, SDVariable segmentIds) {
        SDValidation.validateNumerical("segmentMin", "segmentIds", segmentIds);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentMin(sd(),data, segmentIds).outputVariable();
    }

    /**
     * Segment min operation.<br>
     *
     * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
     * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
     * then output = [6, 9, 8] = [op(3,6), op(1,4,9), op(2,8)]<br>
     * Note that the segment IDs must be sorted from smallest to largest segment.<br>
     * See {unsortedSegment (String, SDVariable, SDVariable, int) ops<br>
     * for the same op without this sorted requirement<br>
     *
     * @param name name May be null. Name for the output variable
     * @param data Data to perform segment max on (NDARRAY type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @return output Segment output (NUMERIC type)
     */
    public SDVariable segmentMin(String name, SDVariable data, SDVariable segmentIds) {
        SDValidation.validateNumerical("segmentMin", "segmentIds", segmentIds);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentMin(sd(),data, segmentIds).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Segment product operation.<br>
     *
     * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
     * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
     * then output = [6, 9, 8] = [op(3,6), op(1,4,9), op(2,8)]<br>
     * Note that the segment IDs must be sorted from smallest to largest segment.<br>
     * See {unsortedSegment (String, SDVariable, SDVariable, int) ops<br>
     * for the same op without this sorted requirement<br>
     *
     * @param data Data to perform segment max on (NDARRAY type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @return output Segment output (NUMERIC type)
     */
    public SDVariable segmentProd(SDVariable data, SDVariable segmentIds) {
        SDValidation.validateNumerical("segmentProd", "segmentIds", segmentIds);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentProd(sd(),data, segmentIds).outputVariable();
    }

    /**
     * Segment product operation.<br>
     *
     * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
     * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
     * then output = [6, 9, 8] = [op(3,6), op(1,4,9), op(2,8)]<br>
     * Note that the segment IDs must be sorted from smallest to largest segment.<br>
     * See {unsortedSegment (String, SDVariable, SDVariable, int) ops<br>
     * for the same op without this sorted requirement<br>
     *
     * @param name name May be null. Name for the output variable
     * @param data Data to perform segment max on (NDARRAY type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @return output Segment output (NUMERIC type)
     */
    public SDVariable segmentProd(String name, SDVariable data, SDVariable segmentIds) {
        SDValidation.validateNumerical("segmentProd", "segmentIds", segmentIds);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentProd(sd(),data, segmentIds).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Segment sum operation.<br>
     *
     * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
     * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
     * then output = [6, 9, 8] = [op(3,6), op(1,4,9), op(2,8)]<br>
     * Note that the segment IDs must be sorted from smallest to largest segment.<br>
     * See {unsortedSegment (String, SDVariable, SDVariable, int) ops<br>
     * for the same op without this sorted requirement<br>
     *
     * @param data Data to perform segment max on (NDARRAY type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @return output Segment output (NUMERIC type)
     */
    public SDVariable segmentSum(SDVariable data, SDVariable segmentIds) {
        SDValidation.validateNumerical("segmentSum", "segmentIds", segmentIds);
        return new org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentSum(sd(),data, segmentIds).outputVariable();
    }

    /**
     * Segment sum operation.<br>
     *
     * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
     * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
     * then output = [6, 9, 8] = [op(3,6), op(1,4,9), op(2,8)]<br>
     * Note that the segment IDs must be sorted from smallest to largest segment.<br>
     * See {unsortedSegment (String, SDVariable, SDVariable, int) ops<br>
     * for the same op without this sorted requirement<br>
     *
     * @param name name May be null. Name for the output variable
     * @param data Data to perform segment max on (NDARRAY type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @return output Segment output (NUMERIC type)
     */
    public SDVariable segmentSum(String name, SDVariable data, SDVariable segmentIds) {
        SDValidation.validateNumerical("segmentSum", "segmentIds", segmentIds);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentSum(sd(),data, segmentIds).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Generate a sequence mask (with values 0 or 1) based on the specified lengths <br>
     * Specifically, out[i, ..., k, j] = (j < lengths[i, ..., k] ? 1.0 : 0.0)<br>
     *
     * @param lengths Lengths of the sequences (NUMERIC type)
     * @param maxLen Maximum sequence length
     * @param dataType
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable sequenceMask(SDVariable lengths, int maxLen, DataType dataType) {
        SDValidation.validateNumerical("sequenceMask", "lengths", lengths);
        return new org.nd4j.linalg.api.ops.impl.shape.SequenceMask(sd(),lengths, maxLen, dataType).outputVariable();
    }

    /**
     * Generate a sequence mask (with values 0 or 1) based on the specified lengths <br>
     * Specifically, out[i, ..., k, j] = (j < lengths[i, ..., k] ? 1.0 : 0.0)<br>
     *
     * @param name name May be null. Name for the output variable
     * @param lengths Lengths of the sequences (NUMERIC type)
     * @param maxLen Maximum sequence length
     * @param dataType
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable sequenceMask(String name, SDVariable lengths, int maxLen, DataType dataType) {
        SDValidation.validateNumerical("sequenceMask", "lengths", lengths);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.SequenceMask(sd(),lengths, maxLen, dataType).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * see sequenceMask(String, SDVariable, SDVariable, DataType)<br>
     *
     * @param lengths  (NUMERIC type)
     * @param dataType
     * @return output  (NUMERIC type)
     */
    public SDVariable sequenceMask(SDVariable lengths, DataType dataType) {
        SDValidation.validateNumerical("sequenceMask", "lengths", lengths);
        return new org.nd4j.linalg.api.ops.impl.shape.SequenceMask(sd(),lengths, dataType).outputVariable();
    }

    /**
     * see sequenceMask(String, SDVariable, SDVariable, DataType)<br>
     *
     * @param name name May be null. Name for the output variable
     * @param lengths  (NUMERIC type)
     * @param dataType
     * @return output  (NUMERIC type)
     */
    public SDVariable sequenceMask(String name, SDVariable lengths, DataType dataType) {
        SDValidation.validateNumerical("sequenceMask", "lengths", lengths);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.SequenceMask(sd(),lengths, dataType).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Returns the shape of the specified INDArray  as a 1D INDArray <br>
     *
     * @param input Input variable (NUMERIC type)
     * @return output 1D output variable with contents equal to the shape of the input (NUMERIC type)
     */
    public SDVariable shape(SDVariable input) {
        SDValidation.validateNumerical("shape", "input", input);
        return new org.nd4j.linalg.api.ops.impl.shape.Shape(sd(),input).outputVariable();
    }

    /**
     * Returns the shape of the specified INDArray  as a 1D INDArray <br>
     *
     * @param name name May be null. Name for the output variable
     * @param input Input variable (NUMERIC type)
     * @return output 1D output variable with contents equal to the shape of the input (NUMERIC type)
     */
    public SDVariable shape(String name, SDVariable input) {
        SDValidation.validateNumerical("shape", "input", input);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Shape(sd(),input).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Returns the size (number of elements, i.e., prod(shape)) of the specified INDArray  as a 0D scalar variable<br>
     *
     * @param in Input variable (NUMERIC type)
     * @return output 0D (scalar) output variable with value equal to the number of elements in the specified array (NUMERIC type)
     */
    public SDVariable size(SDVariable in) {
        SDValidation.validateNumerical("size", "in", in);
        return new org.nd4j.linalg.api.ops.impl.shape.Size(sd(),in).outputVariable();
    }

    /**
     * Returns the size (number of elements, i.e., prod(shape)) of the specified INDArray  as a 0D scalar variable<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @return output 0D (scalar) output variable with value equal to the number of elements in the specified array (NUMERIC type)
     */
    public SDVariable size(String name, SDVariable in) {
        SDValidation.validateNumerical("size", "in", in);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Size(sd(),in).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Returns a rank 0 (scalar) variable for the size of the specified dimension.<br>
     * For example, if X has shape [10,20,30] then sizeAt(X,1)=20. Similarly, sizeAt(X,-1)=30<br>
     *
     * @param in Input variable (NUMERIC type)
     * @param dimension Dimension to get size of
     * @return output Scalar INDArray  for size at specified variable (NUMERIC type)
     */
    public SDVariable sizeAt(SDVariable in, int dimension) {
        SDValidation.validateNumerical("sizeAt", "in", in);
        return new org.nd4j.linalg.api.ops.impl.shape.SizeAt(sd(),in, dimension).outputVariable();
    }

    /**
     * Returns a rank 0 (scalar) variable for the size of the specified dimension.<br>
     * For example, if X has shape [10,20,30] then sizeAt(X,1)=20. Similarly, sizeAt(X,-1)=30<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Input variable (NUMERIC type)
     * @param dimension Dimension to get size of
     * @return output Scalar INDArray  for size at specified variable (NUMERIC type)
     */
    public SDVariable sizeAt(String name, SDVariable in, int dimension) {
        SDValidation.validateNumerical("sizeAt", "in", in);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.SizeAt(sd(),in, dimension).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Get a subset of the specified input, by specifying the first element and the size of the array.<br>
     * For example, if input is:<br>
     * [a, b, c]<br>
     * [d, e, f]<br>
     * then slice(input, begin=[0,1], size=[2,1] will return:<br>
     * [b]<br>
     * [e]<br>
     * Note that for each dimension i, begin[i] + size[i] <= input.size(i)<br>
     *
     * @param input input Variable to get subset of (NUMERIC type)
     * @param begin Beginning index. Must be same length as rank of input array (Size: AtLeast(min=1))
     * @param size Size of the output array. Must be same length as rank of input array (Size: AtLeast(min=1))
     * @return output Subset of the input (NUMERIC type)
     */
    public SDVariable slice(SDVariable input, int[] begin, int... size) {
        SDValidation.validateNumerical("slice", "input", input);
        Preconditions.checkArgument(begin.length >= 1, "begin has incorrect size/length. Expected: begin.length >= 1, got %s", begin.length);
        Preconditions.checkArgument(size.length >= 1, "size has incorrect size/length. Expected: size.length >= 1, got %s", size.length);
        return new org.nd4j.linalg.api.ops.impl.shape.Slice(sd(),input, begin, size).outputVariable();
    }

    /**
     * Get a subset of the specified input, by specifying the first element and the size of the array.<br>
     * For example, if input is:<br>
     * [a, b, c]<br>
     * [d, e, f]<br>
     * then slice(input, begin=[0,1], size=[2,1] will return:<br>
     * [b]<br>
     * [e]<br>
     * Note that for each dimension i, begin[i] + size[i] <= input.size(i)<br>
     *
     * @param name name May be null. Name for the output variable
     * @param input input Variable to get subset of (NUMERIC type)
     * @param begin Beginning index. Must be same length as rank of input array (Size: AtLeast(min=1))
     * @param size Size of the output array. Must be same length as rank of input array (Size: AtLeast(min=1))
     * @return output Subset of the input (NUMERIC type)
     */
    public SDVariable slice(String name, SDVariable input, int[] begin, int... size) {
        SDValidation.validateNumerical("slice", "input", input);
        Preconditions.checkArgument(begin.length >= 1, "begin has incorrect size/length. Expected: begin.length >= 1, got %s", begin.length);
        Preconditions.checkArgument(size.length >= 1, "size has incorrect size/length. Expected: size.length >= 1, got %s", size.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Slice(sd(),input, begin, size).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Squared L2 norm: see norm2(String, SDVariable, boolean, int...)<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x  (NUMERIC type)
     * @param keepDims
     * @param dimensions  (Size: AtLeast(min=0))
     * @return output  (NUMERIC type)
     */
    public SDVariable squaredNorm(SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("squaredNorm", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.floating.SquaredNorm(sd(),x, keepDims, dimensions).outputVariable();
    }

    /**
     * Squared L2 norm: see norm2(String, SDVariable, boolean, int...)<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x  (NUMERIC type)
     * @param keepDims
     * @param dimensions  (Size: AtLeast(min=0))
     * @return output  (NUMERIC type)
     */
    public SDVariable squaredNorm(String name, SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("squaredNorm", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.floating.SquaredNorm(sd(),x, keepDims, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Squared L2 norm: see norm2(String, SDVariable, boolean, int...)<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x  (NUMERIC type)
     * @param dimensions  (Size: AtLeast(min=0))
     * @return output  (NUMERIC type)
     */
    public SDVariable squaredNorm(SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("squaredNorm", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.floating.SquaredNorm(sd(),x, false, dimensions).outputVariable();
    }

    /**
     * Squared L2 norm: see norm2(String, SDVariable, boolean, int...)<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x  (NUMERIC type)
     * @param dimensions  (Size: AtLeast(min=0))
     * @return output  (NUMERIC type)
     */
    public SDVariable squaredNorm(String name, SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("squaredNorm", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.floating.SquaredNorm(sd(),x, false, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Remove a single dimension of size 1.<br>
     * For example, if input has shape [a,b,1,c] then squeeze(input, 2) returns an array of shape [a,b,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param axis Size 1 dimension to remove
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable squeeze(SDVariable x, int axis) {
        SDValidation.validateNumerical("squeeze", "x", x);
        return new org.nd4j.linalg.api.ops.impl.shape.Squeeze(sd(),x, axis).outputVariable();
    }

    /**
     * Remove a single dimension of size 1.<br>
     * For example, if input has shape [a,b,1,c] then squeeze(input, 2) returns an array of shape [a,b,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param axis Size 1 dimension to remove
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable squeeze(String name, SDVariable x, int axis) {
        SDValidation.validateNumerical("squeeze", "x", x);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Squeeze(sd(),x, axis).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Stack a set of N INDArray of rank X into one rank X+1 variable.<br>
     * If inputs have shape [a,b,c] then output has shape:<br>
     * axis = 0: [N,a,b,c]<br>
     * axis = 1: [a,N,b,c]<br>
     * axis = 2: [a,b,N,c]<br>
     * axis = 3: [a,b,c,N]<br>
     * see unstack(String[], SDVariable, int, int)<br>
     *
     * @param values Input variables to stack. Must have the same shape for all inputs (NDARRAY type)
     * @param axis Axis to stack on
     * @return output Output variable (NDARRAY type)
     */
    public SDVariable stack(SDVariable values, int axis) {
        return new org.nd4j.linalg.api.ops.impl.shape.Stack(sd(),values, axis).outputVariable();
    }

    /**
     * Stack a set of N INDArray of rank X into one rank X+1 variable.<br>
     * If inputs have shape [a,b,c] then output has shape:<br>
     * axis = 0: [N,a,b,c]<br>
     * axis = 1: [a,N,b,c]<br>
     * axis = 2: [a,b,N,c]<br>
     * axis = 3: [a,b,c,N]<br>
     * see unstack(String[], SDVariable, int, int)<br>
     *
     * @param name name May be null. Name for the output variable
     * @param values Input variables to stack. Must have the same shape for all inputs (NDARRAY type)
     * @param axis Axis to stack on
     * @return output Output variable (NDARRAY type)
     */
    public SDVariable stack(String name, SDVariable values, int axis) {
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Stack(sd(),values, axis).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Stardard deviation array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param biasCorrected If true: divide by (N-1) (i.e., sample stdev). If false: divide by N (population stdev)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable standardDeviation(SDVariable x, boolean biasCorrected, boolean keepDims,
                                        int... dimensions) {
        SDValidation.validateNumerical("standardDeviation", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.summarystats.StandardDeviation(sd(),x, biasCorrected, keepDims, dimensions).outputVariable();
    }

    /**
     * Stardard deviation array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param biasCorrected If true: divide by (N-1) (i.e., sample stdev). If false: divide by N (population stdev)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable standardDeviation(String name, SDVariable x, boolean biasCorrected,
                                        boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("standardDeviation", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.summarystats.StandardDeviation(sd(),x, biasCorrected, keepDims, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Stardard deviation array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param biasCorrected If true: divide by (N-1) (i.e., sample stdev). If false: divide by N (population stdev)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable standardDeviation(SDVariable x, boolean biasCorrected, int... dimensions) {
        SDValidation.validateNumerical("standardDeviation", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.summarystats.StandardDeviation(sd(),x, biasCorrected, false, dimensions).outputVariable();
    }

    /**
     * Stardard deviation array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param biasCorrected If true: divide by (N-1) (i.e., sample stdev). If false: divide by N (population stdev)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable standardDeviation(String name, SDVariable x, boolean biasCorrected,
                                        int... dimensions) {
        SDValidation.validateNumerical("standardDeviation", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.summarystats.StandardDeviation(sd(),x, biasCorrected, false, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Get a subset of the specified input, by specifying the first element, last element, and the strides.<br>
     * For example, if input is:<br>
     * [a, b, c]<br>
     * [d, e, f]<br>
     * [g, h, i]<br>
     * then stridedSlice(input, begin=[0,1], end=[2,2], strides=[2,1], all masks = 0) will return:<br>
     * [b, c]<br>
     * [h, i]<br>
     *
     * @param in Variable to get subset of (NUMERIC type)
     * @param begin Beginning index (Size: AtLeast(min=1))
     * @param end End index (Size: AtLeast(min=1))
     * @param strides Stride ("step size") for each dimension. For example, stride of 2 means take every second element. (Size: AtLeast(min=1))
     * @param beginMask Bit mask: If the ith bit is set to 1, then the value in the begin long[] is ignored, and a value of 0 is used instead for the beginning index for that dimension
     * @param endMask Bit mask: If the ith bit is set to 1, then the value in the end long[] is ignored, and a value of size(i)-1 is used instead for the end index for that dimension
     * @param ellipsisMask Bit mask: only one non-zero value is allowed here. If a non-zero value is set, then other dimensions are inserted as required at the specified position
     * @param newAxisMask Bit mask: if the ith bit is set to 1, then the begin/end/stride values are ignored, and a size 1 dimension is inserted at this point
     * @param shrinkAxisMask Bit mask: if the ith bit is set to 1, then the begin/end/stride values are ignored, and a size 1 dimension is removed at this point. Note that begin/end/stride values must result in a size 1 output for these dimensions
     * @return output A subset of the input array (NUMERIC type)
     */
    public SDVariable stridedSlice(SDVariable in, int[] begin, int[] end, int[] strides,
                                   int beginMask, int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        SDValidation.validateNumerical("stridedSlice", "in", in);
        Preconditions.checkArgument(begin.length >= 1, "begin has incorrect size/length. Expected: begin.length >= 1, got %s", begin.length);
        Preconditions.checkArgument(end.length >= 1, "end has incorrect size/length. Expected: end.length >= 1, got %s", end.length);
        Preconditions.checkArgument(strides.length >= 1, "strides has incorrect size/length. Expected: strides.length >= 1, got %s", strides.length);
        return new org.nd4j.linalg.api.ops.impl.shape.StridedSlice(sd(),in, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask).outputVariable();
    }

    /**
     * Get a subset of the specified input, by specifying the first element, last element, and the strides.<br>
     * For example, if input is:<br>
     * [a, b, c]<br>
     * [d, e, f]<br>
     * [g, h, i]<br>
     * then stridedSlice(input, begin=[0,1], end=[2,2], strides=[2,1], all masks = 0) will return:<br>
     * [b, c]<br>
     * [h, i]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Variable to get subset of (NUMERIC type)
     * @param begin Beginning index (Size: AtLeast(min=1))
     * @param end End index (Size: AtLeast(min=1))
     * @param strides Stride ("step size") for each dimension. For example, stride of 2 means take every second element. (Size: AtLeast(min=1))
     * @param beginMask Bit mask: If the ith bit is set to 1, then the value in the begin long[] is ignored, and a value of 0 is used instead for the beginning index for that dimension
     * @param endMask Bit mask: If the ith bit is set to 1, then the value in the end long[] is ignored, and a value of size(i)-1 is used instead for the end index for that dimension
     * @param ellipsisMask Bit mask: only one non-zero value is allowed here. If a non-zero value is set, then other dimensions are inserted as required at the specified position
     * @param newAxisMask Bit mask: if the ith bit is set to 1, then the begin/end/stride values are ignored, and a size 1 dimension is inserted at this point
     * @param shrinkAxisMask Bit mask: if the ith bit is set to 1, then the begin/end/stride values are ignored, and a size 1 dimension is removed at this point. Note that begin/end/stride values must result in a size 1 output for these dimensions
     * @return output A subset of the input array (NUMERIC type)
     */
    public SDVariable stridedSlice(String name, SDVariable in, int[] begin, int[] end, int[] strides,
                                   int beginMask, int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        SDValidation.validateNumerical("stridedSlice", "in", in);
        Preconditions.checkArgument(begin.length >= 1, "begin has incorrect size/length. Expected: begin.length >= 1, got %s", begin.length);
        Preconditions.checkArgument(end.length >= 1, "end has incorrect size/length. Expected: end.length >= 1, got %s", end.length);
        Preconditions.checkArgument(strides.length >= 1, "strides has incorrect size/length. Expected: strides.length >= 1, got %s", strides.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.StridedSlice(sd(),in, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Get a subset of the specified input, by specifying the first element, last element, and the strides.<br>
     * For example, if input is:<br>
     * [a, b, c]<br>
     * [d, e, f]<br>
     * [g, h, i]<br>
     * then stridedSlice(input, begin=[0,1], end=[2,2], strides=[2,1], all masks = 0) will return:<br>
     * [b, c]<br>
     * [h, i]<br>
     *
     * @param in Variable to get subset of (NUMERIC type)
     * @param begin Beginning index (Size: AtLeast(min=1))
     * @param end End index (Size: AtLeast(min=1))
     * @param strides Stride ("step size") for each dimension. For example, stride of 2 means take every second element. (Size: AtLeast(min=1))
     * @return output A subset of the input array (NUMERIC type)
     */
    public SDVariable stridedSlice(SDVariable in, int[] begin, int[] end, int... strides) {
        SDValidation.validateNumerical("stridedSlice", "in", in);
        Preconditions.checkArgument(begin.length >= 1, "begin has incorrect size/length. Expected: begin.length >= 1, got %s", begin.length);
        Preconditions.checkArgument(end.length >= 1, "end has incorrect size/length. Expected: end.length >= 1, got %s", end.length);
        Preconditions.checkArgument(strides.length >= 1, "strides has incorrect size/length. Expected: strides.length >= 1, got %s", strides.length);
        return new org.nd4j.linalg.api.ops.impl.shape.StridedSlice(sd(),in, begin, end, strides, 0, 0, 0, 0, 0).outputVariable();
    }

    /**
     * Get a subset of the specified input, by specifying the first element, last element, and the strides.<br>
     * For example, if input is:<br>
     * [a, b, c]<br>
     * [d, e, f]<br>
     * [g, h, i]<br>
     * then stridedSlice(input, begin=[0,1], end=[2,2], strides=[2,1], all masks = 0) will return:<br>
     * [b, c]<br>
     * [h, i]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param in Variable to get subset of (NUMERIC type)
     * @param begin Beginning index (Size: AtLeast(min=1))
     * @param end End index (Size: AtLeast(min=1))
     * @param strides Stride ("step size") for each dimension. For example, stride of 2 means take every second element. (Size: AtLeast(min=1))
     * @return output A subset of the input array (NUMERIC type)
     */
    public SDVariable stridedSlice(String name, SDVariable in, int[] begin, int[] end,
                                   int... strides) {
        SDValidation.validateNumerical("stridedSlice", "in", in);
        Preconditions.checkArgument(begin.length >= 1, "begin has incorrect size/length. Expected: begin.length >= 1, got %s", begin.length);
        Preconditions.checkArgument(end.length >= 1, "end has incorrect size/length. Expected: end.length >= 1, got %s", end.length);
        Preconditions.checkArgument(strides.length >= 1, "strides has incorrect size/length. Expected: strides.length >= 1, got %s", strides.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.StridedSlice(sd(),in, begin, end, strides, 0, 0, 0, 0, 0).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Sum array reduction operation, optionally along specified dimensions.<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or of rank (input rank) if keepdims = true (NUMERIC type)
     */
    public SDVariable sum(SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("sum", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.same.Sum(sd(),x, keepDims, dimensions).outputVariable();
    }

    /**
     * Sum array reduction operation, optionally along specified dimensions.<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param keepDims If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or of rank (input rank) if keepdims = true (NUMERIC type)
     */
    public SDVariable sum(String name, SDVariable x, boolean keepDims, int... dimensions) {
        SDValidation.validateNumerical("sum", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.same.Sum(sd(),x, keepDims, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Sum array reduction operation, optionally along specified dimensions.<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or of rank (input rank) if keepdims = true (NUMERIC type)
     */
    public SDVariable sum(SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("sum", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.same.Sum(sd(),x, false, dimensions).outputVariable();
    }

    /**
     * Sum array reduction operation, optionally along specified dimensions.<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or of rank (input rank) if keepdims = true (NUMERIC type)
     */
    public SDVariable sum(String name, SDVariable x, int... dimensions) {
        SDValidation.validateNumerical("sum", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.same.Sum(sd(),x, false, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * //TODO: Ops must be documented.<br>
     *
     * @param x Input variable x (NUMERIC type)
     * @param y Input variable y (NUMERIC type)
     * @param dimensionsX dimensions for first input array (x) (Size: AtLeast(min=1))
     * @param dimensionsY dimensions for second input array (y) (Size: AtLeast(min=1))
     * @param transposeX Transpose x (first argument)
     * @param transposeY Transpose y (second argument)
     * @param transposeZ Transpose result array
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable tensorMmul(SDVariable x, SDVariable y, int[] dimensionsX, int[] dimensionsY,
                                 boolean transposeX, boolean transposeY, boolean transposeZ) {
        SDValidation.validateNumerical("tensorMmul", "x", x);
        SDValidation.validateNumerical("tensorMmul", "y", y);
        Preconditions.checkArgument(dimensionsX.length >= 1, "dimensionsX has incorrect size/length. Expected: dimensionsX.length >= 1, got %s", dimensionsX.length);
        Preconditions.checkArgument(dimensionsY.length >= 1, "dimensionsY has incorrect size/length. Expected: dimensionsY.length >= 1, got %s", dimensionsY.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.TensorMmul(sd(),x, y, dimensionsX, dimensionsY, transposeX, transposeY, transposeZ).outputVariable();
    }

    /**
     * //TODO: Ops must be documented.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable x (NUMERIC type)
     * @param y Input variable y (NUMERIC type)
     * @param dimensionsX dimensions for first input array (x) (Size: AtLeast(min=1))
     * @param dimensionsY dimensions for second input array (y) (Size: AtLeast(min=1))
     * @param transposeX Transpose x (first argument)
     * @param transposeY Transpose y (second argument)
     * @param transposeZ Transpose result array
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable tensorMmul(String name, SDVariable x, SDVariable y, int[] dimensionsX,
                                 int[] dimensionsY, boolean transposeX, boolean transposeY, boolean transposeZ) {
        SDValidation.validateNumerical("tensorMmul", "x", x);
        SDValidation.validateNumerical("tensorMmul", "y", y);
        Preconditions.checkArgument(dimensionsX.length >= 1, "dimensionsX has incorrect size/length. Expected: dimensionsX.length >= 1, got %s", dimensionsX.length);
        Preconditions.checkArgument(dimensionsY.length >= 1, "dimensionsY has incorrect size/length. Expected: dimensionsY.length >= 1, got %s", dimensionsY.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.TensorMmul(sd(),x, y, dimensionsX, dimensionsY, transposeX, transposeY, transposeZ).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * //TODO: Ops must be documented.<br>
     *
     * @param x Input variable x (NUMERIC type)
     * @param y Input variable y (NUMERIC type)
     * @param dimensionsX dimensions for first input array (x) (Size: AtLeast(min=1))
     * @param dimensionsY dimensions for second input array (y) (Size: AtLeast(min=1))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable tensorMmul(SDVariable x, SDVariable y, int[] dimensionsX, int... dimensionsY) {
        SDValidation.validateNumerical("tensorMmul", "x", x);
        SDValidation.validateNumerical("tensorMmul", "y", y);
        Preconditions.checkArgument(dimensionsX.length >= 1, "dimensionsX has incorrect size/length. Expected: dimensionsX.length >= 1, got %s", dimensionsX.length);
        Preconditions.checkArgument(dimensionsY.length >= 1, "dimensionsY has incorrect size/length. Expected: dimensionsY.length >= 1, got %s", dimensionsY.length);
        return new org.nd4j.linalg.api.ops.impl.reduce.TensorMmul(sd(),x, y, dimensionsX, dimensionsY, false, false, false).outputVariable();
    }

    /**
     * //TODO: Ops must be documented.<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable x (NUMERIC type)
     * @param y Input variable y (NUMERIC type)
     * @param dimensionsX dimensions for first input array (x) (Size: AtLeast(min=1))
     * @param dimensionsY dimensions for second input array (y) (Size: AtLeast(min=1))
     * @return output Output variable (NUMERIC type)
     */
    public SDVariable tensorMmul(String name, SDVariable x, SDVariable y, int[] dimensionsX,
                                 int... dimensionsY) {
        SDValidation.validateNumerical("tensorMmul", "x", x);
        SDValidation.validateNumerical("tensorMmul", "y", y);
        Preconditions.checkArgument(dimensionsX.length >= 1, "dimensionsX has incorrect size/length. Expected: dimensionsX.length >= 1, got %s", dimensionsX.length);
        Preconditions.checkArgument(dimensionsY.length >= 1, "dimensionsY has incorrect size/length. Expected: dimensionsY.length >= 1, got %s", dimensionsY.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.TensorMmul(sd(),x, y, dimensionsX, dimensionsY, false, false, false).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Repeat (tile) the input tensor the specified number of times.<br>
     * For example, if input is<br>
     * [1, 2]<br>
     * [3, 4]<br>
     * and repeat is [2, 3]<br>
     * then output is<br>
     * [1, 2, 1, 2, 1, 2]<br>
     * [3, 4, 3, 4, 3, 4]<br>
     * [1, 2, 1, 2, 1, 2]<br>
     * [3, 4, 3, 4, 3, 4]<br>
     *
     * @param x Input variable (NDARRAY type)
     * @param repeat Number of times to repeat in each axis. Must have length equal to the rank of the input array (INT type)
     * @return output Output variable (NDARRAY type)
     */
    public SDVariable tile(SDVariable x, SDVariable repeat) {
        SDValidation.validateInteger("tile", "repeat", repeat);
        return new org.nd4j.linalg.api.ops.impl.shape.Tile(sd(),x, repeat).outputVariable();
    }

    /**
     * Repeat (tile) the input tensor the specified number of times.<br>
     * For example, if input is<br>
     * [1, 2]<br>
     * [3, 4]<br>
     * and repeat is [2, 3]<br>
     * then output is<br>
     * [1, 2, 1, 2, 1, 2]<br>
     * [3, 4, 3, 4, 3, 4]<br>
     * [1, 2, 1, 2, 1, 2]<br>
     * [3, 4, 3, 4, 3, 4]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NDARRAY type)
     * @param repeat Number of times to repeat in each axis. Must have length equal to the rank of the input array (INT type)
     * @return output Output variable (NDARRAY type)
     */
    public SDVariable tile(String name, SDVariable x, SDVariable repeat) {
        SDValidation.validateInteger("tile", "repeat", repeat);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Tile(sd(),x, repeat).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * see tile(String, SDVariable, int...)<br>
     *
     * @param x  (NDARRAY type)
     * @param repeat  (Size: AtLeast(min=1))
     * @return output  (NDARRAY type)
     */
    public SDVariable tile(SDVariable x, int... repeat) {
        Preconditions.checkArgument(repeat.length >= 1, "repeat has incorrect size/length. Expected: repeat.length >= 1, got %s", repeat.length);
        return new org.nd4j.linalg.api.ops.impl.shape.Tile(sd(),x, repeat).outputVariable();
    }

    /**
     * see tile(String, SDVariable, int...)<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x  (NDARRAY type)
     * @param repeat  (Size: AtLeast(min=1))
     * @return output  (NDARRAY type)
     */
    public SDVariable tile(String name, SDVariable x, int... repeat) {
        Preconditions.checkArgument(repeat.length >= 1, "repeat has incorrect size/length. Expected: repeat.length >= 1, got %s", repeat.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Tile(sd(),x, repeat).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Matrix transpose operation: If input has shape [a,b] output has shape [b,a]<br>
     *
     * @param x Input variable (NDARRAY type)
     * @return output transposed input (NDARRAY type)
     */
    public SDVariable transpose(SDVariable x) {
        return new org.nd4j.linalg.api.ops.impl.shape.Transpose(sd(),x).outputVariable();
    }

    /**
     * Matrix transpose operation: If input has shape [a,b] output has shape [b,a]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NDARRAY type)
     * @return output transposed input (NDARRAY type)
     */
    public SDVariable transpose(String name, SDVariable x) {
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Transpose(sd(),x).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Unsorted segment max operation. As per segmentMax(String, SDVariable, SDVariable) but without<br>
     * the requirement for the indices to be sorted.<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [6, 9, 8] = [max(3,6), max(1,4,9), max(2,8)]<br>
     *
     * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @param numSegments Number of segments
     * @return output Unsorted segment output (NUMERIC type)
     */
    public SDVariable unsortedSegmentMax(SDVariable data, SDVariable segmentIds, int numSegments) {
        SDValidation.validateNumerical("unsortedSegmentMax", "data", data);
        SDValidation.validateNumerical("unsortedSegmentMax", "segmentIds", segmentIds);
        return new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMax(sd(),data, segmentIds, numSegments).outputVariable();
    }

    /**
     * Unsorted segment max operation. As per segmentMax(String, SDVariable, SDVariable) but without<br>
     * the requirement for the indices to be sorted.<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [6, 9, 8] = [max(3,6), max(1,4,9), max(2,8)]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @param numSegments Number of segments
     * @return output Unsorted segment output (NUMERIC type)
     */
    public SDVariable unsortedSegmentMax(String name, SDVariable data, SDVariable segmentIds,
                                         int numSegments) {
        SDValidation.validateNumerical("unsortedSegmentMax", "data", data);
        SDValidation.validateNumerical("unsortedSegmentMax", "segmentIds", segmentIds);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMax(sd(),data, segmentIds, numSegments).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Unsorted segment mean operation. As per segmentMean(String, SDVariable, SDVariable) but without<br>
     * the requirement for the indices to be sorted.<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [4.5, 4.666, 5] = [mean(3,6), mean(1,4,9), mean(2,8)]<br>
     *
     * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @param numSegments Number of segments
     * @return output Unsorted segment output (NUMERIC type)
     */
    public SDVariable unsortedSegmentMean(SDVariable data, SDVariable segmentIds, int numSegments) {
        SDValidation.validateNumerical("unsortedSegmentMean", "data", data);
        SDValidation.validateNumerical("unsortedSegmentMean", "segmentIds", segmentIds);
        return new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMean(sd(),data, segmentIds, numSegments).outputVariable();
    }

    /**
     * Unsorted segment mean operation. As per segmentMean(String, SDVariable, SDVariable) but without<br>
     * the requirement for the indices to be sorted.<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [4.5, 4.666, 5] = [mean(3,6), mean(1,4,9), mean(2,8)]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @param numSegments Number of segments
     * @return output Unsorted segment output (NUMERIC type)
     */
    public SDVariable unsortedSegmentMean(String name, SDVariable data, SDVariable segmentIds,
                                          int numSegments) {
        SDValidation.validateNumerical("unsortedSegmentMean", "data", data);
        SDValidation.validateNumerical("unsortedSegmentMean", "segmentIds", segmentIds);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMean(sd(),data, segmentIds, numSegments).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Unsorted segment min operation. As per segmentMin(String, SDVariable, SDVariable) but without<br>
     * the requirement for the indices to be sorted.<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [3, 1, 2] = [min(3,6), min(1,4,9), min(2,8)]<br>
     *
     * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @param numSegments Number of segments
     * @return output Unsorted segment output (NUMERIC type)
     */
    public SDVariable unsortedSegmentMin(SDVariable data, SDVariable segmentIds, int numSegments) {
        SDValidation.validateNumerical("unsortedSegmentMin", "data", data);
        SDValidation.validateNumerical("unsortedSegmentMin", "segmentIds", segmentIds);
        return new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMin(sd(),data, segmentIds, numSegments).outputVariable();
    }

    /**
     * Unsorted segment min operation. As per segmentMin(String, SDVariable, SDVariable) but without<br>
     * the requirement for the indices to be sorted.<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [3, 1, 2] = [min(3,6), min(1,4,9), min(2,8)]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @param numSegments Number of segments
     * @return output Unsorted segment output (NUMERIC type)
     */
    public SDVariable unsortedSegmentMin(String name, SDVariable data, SDVariable segmentIds,
                                         int numSegments) {
        SDValidation.validateNumerical("unsortedSegmentMin", "data", data);
        SDValidation.validateNumerical("unsortedSegmentMin", "segmentIds", segmentIds);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMin(sd(),data, segmentIds, numSegments).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Unsorted segment product operation. As per segmentProd(String, SDVariable, SDVariable) but without<br>
     * the requirement for the indices to be sorted.<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [4.5, 4.666, 5] = [mean(3,6), mean(1,4,9), mean(2,8)]<br>
     *
     * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @param numSegments Number of segments
     * @return output Unsorted segment output (NUMERIC type)
     */
    public SDVariable unsortedSegmentProd(SDVariable data, SDVariable segmentIds, int numSegments) {
        SDValidation.validateNumerical("unsortedSegmentProd", "data", data);
        SDValidation.validateNumerical("unsortedSegmentProd", "segmentIds", segmentIds);
        return new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentProd(sd(),data, segmentIds, numSegments).outputVariable();
    }

    /**
     * Unsorted segment product operation. As per segmentProd(String, SDVariable, SDVariable) but without<br>
     * the requirement for the indices to be sorted.<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [4.5, 4.666, 5] = [mean(3,6), mean(1,4,9), mean(2,8)]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @param numSegments Number of segments
     * @return output Unsorted segment output (NUMERIC type)
     */
    public SDVariable unsortedSegmentProd(String name, SDVariable data, SDVariable segmentIds,
                                          int numSegments) {
        SDValidation.validateNumerical("unsortedSegmentProd", "data", data);
        SDValidation.validateNumerical("unsortedSegmentProd", "segmentIds", segmentIds);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentProd(sd(),data, segmentIds, numSegments).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Unsorted segment sqrtN operation. Simply returns the sqrt of the count of the number of values in each segment<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [1.414, 1.732, 1.414] = [sqrt(2), sqrtN(3), sqrtN(2)]<br>
     *
     * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @param numSegments Number of segments
     * @return output Unsorted segment output (NUMERIC type)
     */
    public SDVariable unsortedSegmentSqrtN(SDVariable data, SDVariable segmentIds, int numSegments) {
        SDValidation.validateNumerical("unsortedSegmentSqrtN", "data", data);
        SDValidation.validateNumerical("unsortedSegmentSqrtN", "segmentIds", segmentIds);
        return new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentSqrtN(sd(),data, segmentIds, numSegments).outputVariable();
    }

    /**
     * Unsorted segment sqrtN operation. Simply returns the sqrt of the count of the number of values in each segment<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [1.414, 1.732, 1.414] = [sqrt(2), sqrtN(3), sqrtN(2)]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @param numSegments Number of segments
     * @return output Unsorted segment output (NUMERIC type)
     */
    public SDVariable unsortedSegmentSqrtN(String name, SDVariable data, SDVariable segmentIds,
                                           int numSegments) {
        SDValidation.validateNumerical("unsortedSegmentSqrtN", "data", data);
        SDValidation.validateNumerical("unsortedSegmentSqrtN", "segmentIds", segmentIds);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentSqrtN(sd(),data, segmentIds, numSegments).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Unsorted segment sum operation. As per segmentSum(String, SDVariable, SDVariable) but without<br>
     * the requirement for the indices to be sorted.<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [9, 14, 10] = [sum(3,6), sum(1,4,9), sum(2,8)]<br>
     *
     * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @param numSegments Number of segments
     * @return output Unsorted segment output (NUMERIC type)
     */
    public SDVariable unsortedSegmentSum(SDVariable data, SDVariable segmentIds, int numSegments) {
        SDValidation.validateNumerical("unsortedSegmentSum", "data", data);
        SDValidation.validateNumerical("unsortedSegmentSum", "segmentIds", segmentIds);
        return new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentSum(sd(),data, segmentIds, numSegments).outputVariable();
    }

    /**
     * Unsorted segment sum operation. As per segmentSum(String, SDVariable, SDVariable) but without<br>
     * the requirement for the indices to be sorted.<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [9, 14, 10] = [sum(3,6), sum(1,4,9), sum(2,8)]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
     * @param segmentIds Variable for the segment IDs (NUMERIC type)
     * @param numSegments Number of segments
     * @return output Unsorted segment output (NUMERIC type)
     */
    public SDVariable unsortedSegmentSum(String name, SDVariable data, SDVariable segmentIds,
                                         int numSegments) {
        SDValidation.validateNumerical("unsortedSegmentSum", "data", data);
        SDValidation.validateNumerical("unsortedSegmentSum", "segmentIds", segmentIds);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentSum(sd(),data, segmentIds, numSegments).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Variance array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param biasCorrected If true: divide by (N-1) (i.e., sample variable). If false: divide by N (population variance)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable variance(SDVariable x, boolean biasCorrected, boolean keepDims,
                               int... dimensions) {
        SDValidation.validateNumerical("variance", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.summarystats.Variance(sd(),x, biasCorrected, keepDims, dimensions).outputVariable();
    }

    /**
     * Variance array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param biasCorrected If true: divide by (N-1) (i.e., sample variable). If false: divide by N (population variance)
     * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable variance(String name, SDVariable x, boolean biasCorrected, boolean keepDims,
                               int... dimensions) {
        SDValidation.validateNumerical("variance", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.summarystats.Variance(sd(),x, biasCorrected, keepDims, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Variance array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param x Input variable (NUMERIC type)
     * @param biasCorrected If true: divide by (N-1) (i.e., sample variable). If false: divide by N (population variance)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable variance(SDVariable x, boolean biasCorrected, int... dimensions) {
        SDValidation.validateNumerical("variance", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        return new org.nd4j.linalg.api.ops.impl.summarystats.Variance(sd(),x, biasCorrected, false, dimensions).outputVariable();
    }

    /**
     * Variance array reduction operation, optionally along specified dimensions<br>
     *
     * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]<br>
     *
     * @param name name May be null. Name for the output variable
     * @param x Input variable (NUMERIC type)
     * @param biasCorrected If true: divide by (N-1) (i.e., sample variable). If false: divide by N (population variance)
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
     * @return output reduced array of rank (input rank - num dimensions) (NUMERIC type)
     */
    public SDVariable variance(String name, SDVariable x, boolean biasCorrected, int... dimensions) {
        SDValidation.validateNumerical("variance", "x", x);
        Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.summarystats.Variance(sd(),x, biasCorrected, false, dimensions).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }

    /**
     * Return a variable of all 0s, with the same shape as the input variable. Note that this is dynamic:<br>
     * if the input shape changes in later execution, the returned variable's shape will also be updated<br>
     *
     * @param input Input  (NUMERIC type)
     * @return output A new Variable with the same (dynamic) shape as the input (NUMERIC type)
     */
    public SDVariable zerosLike(SDVariable input) {
        SDValidation.validateNumerical("zerosLike", "input", input);
        return new org.nd4j.linalg.api.ops.impl.shape.ZerosLike(sd(),input).outputVariable();
    }

    /**
     * Return a variable of all 0s, with the same shape as the input variable. Note that this is dynamic:<br>
     * if the input shape changes in later execution, the returned variable's shape will also be updated<br>
     *
     * @param name name May be null. Name for the output variable
     * @param input Input  (NUMERIC type)
     * @return output A new Variable with the same (dynamic) shape as the input (NUMERIC type)
     */
    public SDVariable zerosLike(String name, SDVariable input) {
        SDValidation.validateNumerical("zerosLike", "input", input);
        SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.ZerosLike(sd(),input).outputVariable();
        return sd().updateVariableNameAndReference(out, name);
    }
}
