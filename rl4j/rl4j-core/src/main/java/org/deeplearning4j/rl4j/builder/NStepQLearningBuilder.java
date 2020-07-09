/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.deeplearning4j.rl4j.builder;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.SuperBuilder;
import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.rl4j.agent.IAgentLearner;
import org.deeplearning4j.rl4j.agent.learning.algorithm.IUpdateAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.algorithm.NStepQLearning;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.agent.learning.update.updater.GradientsNeuralNetUpdater;
import org.deeplearning4j.rl4j.agent.learning.update.updater.INeuralNetUpdater;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.IActionSchema;
import org.deeplearning4j.rl4j.experience.ExperienceHandler;
import org.deeplearning4j.rl4j.experience.StateActionExperienceHandler;
import org.deeplearning4j.rl4j.experience.StateActionPair;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.policy.INeuralNetPolicy;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.nd4j.linalg.api.rng.Random;

/**
 * A {@link IAgentLearner} builder that will setup a {@link NStepQLearning n-step Q-Learning} algorithm with these:
 * <li>a epsilon-greedy policy</li>
 * <li>a n-step state-action-reward experience handler</li>
 * <li>a neural net updater that expects gradient update data</li>
 * <li>a n-step Q-Learning gradient conputation algorithm</li>
 */
public class NStepQLearningBuilder extends BaseAgentLearnerBuilder<Integer, StateActionPair<Integer>, Gradients>{


    private final Configuration configuration;
    private final Random rnd;

    public NStepQLearningBuilder(Configuration configuration,
                                 ITrainableNeuralNet neuralNet,
                                 Builder<Environment<Integer>> environmentBuilder,
                                 Builder<TransformProcess> transformProcessBuilder,
                                 Random rnd) {
        super(configuration, neuralNet, environmentBuilder, transformProcessBuilder);
        this.configuration = configuration;
        this.rnd = rnd;
    }

    @Override
    protected IPolicy<Integer> buildPolicy() {
        INeuralNetPolicy<Integer> greedyPolicy = new DQNPolicy<Integer>(networks.getThreadCurrentNetwork());
        IActionSchema<Integer> actionSchema = getEnvironment().getSchema().getActionSchema();
        return new EpsGreedy(greedyPolicy, actionSchema, configuration.getPolicyConfiguration(), rnd);
    }

    @Override
    protected ExperienceHandler<Integer, StateActionPair<Integer>> buildExperienceHandler() {
        return new StateActionExperienceHandler<Integer>(configuration.getExperienceHandlerConfiguration());
    }

    @Override
    protected IUpdateAlgorithm<Gradients, StateActionPair<Integer>> buildUpdateAlgorithm() {
        IActionSchema<Integer> actionSchema = getEnvironment().getSchema().getActionSchema();
        return new NStepQLearning(networks.getThreadCurrentNetwork(), networks.getTargetNetwork(), actionSchema.getActionSpaceSize(), configuration.getNstepQLearningConfiguration());
    }

    @Override
    protected INeuralNetUpdater<Gradients> buildNeuralNetUpdater() {
        return new GradientsNeuralNetUpdater(networks.getThreadCurrentNetwork(), networks.getTargetNetwork(), configuration.getNeuralNetUpdaterConfiguration());
    }

    @EqualsAndHashCode(callSuper = true)
    @SuperBuilder
    @Data
    public static class Configuration extends BaseAgentLearnerBuilder.Configuration<Integer> {
        EpsGreedy.Configuration policyConfiguration;
        GradientsNeuralNetUpdater.Configuration neuralNetUpdaterConfiguration;
        NStepQLearning.Configuration nstepQLearningConfiguration;
        StateActionExperienceHandler.Configuration experienceHandlerConfiguration;
    }
}
