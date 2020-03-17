package org.deeplearning4j.rl4j.support;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.learning.async.IAsyncGlobal;
import org.deeplearning4j.rl4j.network.NeuralNet;

import java.util.concurrent.atomic.AtomicInteger;

public class MockAsyncGlobal<NN extends NeuralNet> implements IAsyncGlobal<NN> {

    @Getter
    private final NN current;

    public boolean hasBeenStarted = false;
    public boolean hasBeenTerminated = false;

    @Getter
    int workerUpdateCount = 0;

    @Getter
    int stepCount = 0;

    @Setter
    private int maxSteps;

    public MockAsyncGlobal() {
        this(null);
    }

    public MockAsyncGlobal(NN current) {
        maxSteps = Integer.MAX_VALUE;
        this.current = current;
    }

    @Override
    public boolean isTrainingComplete() {
        return stepCount >= maxSteps;
    }

    @Override
    public NN getTarget() {
        return current;
    }

    @Override
    public void applyGradient(Gradient[] gradient, int batchSize) {
        workerUpdateCount++;
        stepCount+=batchSize;
    }

}
