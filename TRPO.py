#!/usr/bin/env python

'''
Based on:
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''
import gym
import gym_gazebo
import time
from distutils.dir_util import copy_tree
import os
import json
import random
import numpy as np
from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
import memory
from baselines.trpo_mpi.nosharing_cnn_policy import CnnPolicy
from baselines.trpo_mpi import trpo_mpi
import baselines.common.tf_util as U


def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        print(file)
        os.unlink(file)

if __name__ == '__main__':

    #REMEMBER!: turtlebot_nn_setup.bash must be executed.
    env = gym.make('GazeboCircuit2TurtlebotLidarNn-v0')
    outdir = '/tmp/gazebo_gym_experiments/'

    sess = U.single_threaded_session()
    sess.__enter__()

    # maybe distance data
    observation = env.reset()
    observation = np.array(observation)
    observation = observation.reshape(10, 10, 1)

    box = gym.spaces.Box(low=10, high=10, shape=(10, 10, 1))
    #box = observation
    discrete = gym.spaces.Discrete(21) ##21?

    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return CnnPolicy(name=name, ob_space=box, ac_space=discrete)

    continue_execution = False
    #fill this if continue_execution=True

    weights_path = '/tmp/turtle_c2_dqn_ep200.h5'
    monitor_path = '/tmp/turtle_c2_dqn_ep200'
    params_json  = '/tmp/turtle_c2_dqn_ep200.json'

    if not continue_execution:
        #Each time we take a sample and update our weights it is called a mini-batch.
        #Each time we run through the entire dataset, it's called an epoch.
        #PARAMETER LIST

        num_timesteps = 10e6
        seed = 0

        trpo_mpi.learn(env, policy_fn, box, discrete, timesteps_per_batch=512, max_kl=0.001, cg_iters=10, cg_damping=1e-3, max_timesteps=int(num_timesteps * 1.1), gamma=0.98, lam=1.0, vf_iters=3, vf_stepsize=1e-4, entcoeff=0.00)
        env = gym.wrappers.Monitor(env, directory=outdir, force=True, write_upon_reset=True)
    else:
        #Load weights, monitor info and parameter info.
        #ADD TRY CATCH fro this else
        with open(params_json) as outfile:
            d = json.load(outfile)
            epochs = d.get('epochs')
            steps = d.get('steps')
            updateTargetNetwork = d.get('updateTargetNetwork')
            explorationRate = d.get('explorationRate')
            minibatch_size = d.get('minibatch_size')
            learnStart = d.get('learnStart')
            learningRate = d.get('learningRate')
            discountFactor = d.get('discountFactor')
            memorySize = d.get('memorySize')
            network_inputs = d.get('network_inputs')
            network_outputs = d.get('network_outputs')
            network_layers = d.get('network_structure')
            current_epoch = d.get('current_epoch')

        deepQ = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_layers)
        deepQ.loadWeights(weights_path)

        clear_monitor_files(outdir)
        copy_tree(monitor_path,outdir)
        env = gym.wrappers.Monitor(env, directory=outdir, force=True, write_upon_reset=True)

    # last100Scores = [0] * 100
    # last100ScoresIndex = 0
    # last100Filled = False
    # stepCounter = 0
    # highest_reward = 0
    #
    # start_time = time.time()
    #
    # #start iterating from 'current epoch'.
    #
    # for epoch in range(current_epoch+1, epochs+1, 1):
    #     observation = env.reset()
    #     cumulated_reward = 0
    #
    #     # number of timesteps
    #     for t in range(steps):
    #         # env.render()
    #         qValues = deepQ.getQValues(observation)
    #
    #         action = deepQ.selectAction(qValues, explorationRate)
    #
    #         newObservation, reward, done, info = env.step(action)
    #
    #         cumulated_reward += reward
    #         if highest_reward < cumulated_reward:
    #             highest_reward = cumulated_reward
    #
    #         deepQ.addMemory(observation, action, reward, newObservation, done)
    #
    #         if stepCounter >= learnStart:
    #             if stepCounter <= updateTargetNetwork:
    #                 deepQ.learnOnMiniBatch(minibatch_size, False)
    #             else :
    #                 deepQ.learnOnMiniBatch(minibatch_size, True)
    #
    #         observation = newObservation
    #
    #         if (t >= 1000):
    #             print ("reached the end! :D")
    #             done = True
    #
    #         env._flush(force=True)
    #         if done:
    #             last100Scores[last100ScoresIndex] = t
    #             last100ScoresIndex += 1
    #             if last100ScoresIndex >= 100:
    #                 last100Filled = True
    #                 last100ScoresIndex = 0
    #             if not last100Filled:
    #                 print ("EP "+str(epoch)+" - {} timesteps".format(t+1)+"   Exploration="+str(round(explorationRate, 2)))
    #             else :
    #                 m, s = divmod(int(time.time() - start_time), 60)
    #                 h, m = divmod(m, 60)
    #                 print ("EP "+str(epoch)+" - {} timesteps".format(t+1)+" - last100 Steps : "+str((sum(last100Scores)/len(last100Scores)))+" - Cumulated R: "+str(cumulated_reward)+"   Eps="+str(round(explorationRate, 2))+"     Time: %d:%02d:%02d" % (h, m, s))
    #                 if (epoch)%100==0:
    #                     #save model weights and monitoring data every 100 epochs.
    #                     deepQ.saveModel('/tmp/turtle_c2_dqn_ep'+str(epoch)+'.h5')
    #                     env._flush(force=True)
    #                     copy_tree(outdir,'/tmp/turtle_c2_dqn_ep'+str(epoch))
    #                     #save simulation parameters.
    #                     parameter_keys = ['epochs','steps','updateTargetNetwork','explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_inputs','network_outputs','network_structure','current_epoch']
    #                     parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch]
    #                     parameter_dictionary = dict(zip(parameter_keys, parameter_values))
    #                     with open('/tmp/turtle_c2_dqn_ep'+str(epoch)+'.json', 'w') as outfile:
    #                         json.dump(parameter_dictionary, outfile)
    #             break
    #
    #         stepCounter += 1
    #         if stepCounter % updateTargetNetwork == 0:
    #             deepQ.updateTargetNetwork()
    #             print ("updating target network")
    #
    #     explorationRate *= 0.995 #epsilon decay
    #     # explorationRate -= (2.0/epochs)
    #     explorationRate = max (0.05, explorationRate)
    #
    # #env.monitor.close()
    # env.close()
