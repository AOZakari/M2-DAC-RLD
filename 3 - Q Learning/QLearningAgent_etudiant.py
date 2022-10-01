import matplotlib
from numpy.core.fromnumeric import argmax
matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from datetime import datetime
import os
from utils import *
from collections import defaultdict

class QLearning(object):


    def __init__(self, env, opt):
        self.opt=opt
        self.action_space = env.action_space
        self.env=env
        self.discount=opt.gamma
        self.alpha=opt.learningRate
        self.gamma=opt.gamma
        self.explo=opt.explo
        self.exploDecay=opt.decay
        self.exploMode=opt.exploMode #0: epsilon greedy, 1: ucb
        self.sarsa=opt.sarsa
        self.modelSamples=opt.nbModelSamples
        self.test=False
        self.qstates = {}  # dictionnaire d'états rencontrés
        self.values = []   # contient, pour chaque numéro d'état, les qvaleurs des self.action_space.n actions possibles
        self.qstates_times = []
        self.lucb = opt.lucb
        self.dyna_r = []
        self.dyna_p = []
        self.dynaAlpha = opt.dynaLearningRate

    def save(self,file):
       pass


    # enregistre cette observation dans la liste des états rencontrés si pas déjà présente
    # retourne l'identifiant associé à cet état
    def storeState(self,obs):

        observation = obs.dumps()
        s = str(observation)
        ss = self.qstates.get(s, -1)

        # Si l'etat jamais rencontré
        if ss < 0:
            ss = len(self.values)
            self.qstates[s] = ss
            self.values.append(np.ones(self.action_space.n) * 1.0) # Optimism faced to uncertainty (on commence avec des valeurs à 1 pour favoriser l'exploration)
            self.qstates_times.append(np.ones(self.action_space.n) * 1.0)
            self.dyna_p.append([dict() for i in range(self.action_space.n)])
            self.dyna_r.append([dict() for i in range(self.action_space.n)])
            for state in range(len(self.values)):
                for action in range(self.action_space.n):
                    self.dyna_r[state][action][ss] = 0.0
                    self.dyna_p[state][action][ss] = 0.0
        return ss



    def act(self, obs):

        s = obs
        
        if (self.exploMode == 0): # Epsilon-greedy avec decay
            
            if (np.random.rand() < self.explo):
                a = self.action_space.sample()
            else:
                a = np.argmax(self.values[s])
        

        elif (self.exploMode == 1): # UCB-1
            a = np.argmax([self.values[s][action] + self.lucb*np.sqrt(2*np.log(np.sum(self.qstates_times[s])/self.qstates_times[s][action])) for action in range(self.action_space.n)])

        self.qstates_times[s][a] += 1
        return a  

    def store(self, ob, action, new_ob, reward, done, it):

        if self.test:
            return
        self.last_source=ob
        self.last_action=action
        self.last_dest=new_ob
        self.last_reward=reward
        if it == self.opt.maxLengthTrain:   # si on a atteint la taille limite, ce n'est pas un vrai done de l'environnement
            done = False
        self.last_done=done



    def learn(self, done):

        if done:
            self.explo *=  self.exploDecay
        else:
            if self.sarsa:
                a = self.act(self.last_dest)
            else:
                a = np.argmax([value for value in self.values[self.last_dest]])
            old_q = self.values[self.last_source][self.last_action]
            self.values[self.last_source][self.last_action] += self.alpha * (self.last_reward + self.gamma * self.values[self.last_dest][a] - old_q)
        
        if (self.modelSamples):

            if self.last_dest not in self.dyna_r[self.last_source][self.last_action]:
                self.dyna_r[self.last_source][self.last_action][self.last_dest] = 0.0
                self.dyna_p[self.last_source][self.last_action][self.last_dest] = np.random.rand()

            self.dyna_r[self.last_source][self.last_action][self.last_dest] += self.dynaAlpha * (self.last_reward - self.dyna_r[self.last_source][self.last_action][self.last_dest])
            self.dyna_p[self.last_source][self.last_action][self.last_dest] += self.dynaAlpha * (1 - self.dyna_p[self.last_source][self.last_action][self.last_dest])
            
            for s in range(len(self.dyna_p)):
                
                if self.last_dest not in self.dyna_r[s][self.last_action]:
                    self.dyna_r[s][self.last_action][self.last_dest] = 0.0
                    self.dyna_p[s][self.last_action][self.last_dest] = np.random.rand()
                
                self.dyna_p[s][self.last_action][self.last_dest] -= self.dynaAlpha *  self.dyna_p[s][self.last_action][self.last_dest]
            
            samples = set()

            for k in range(self.modelSamples):

                s = np.random.randint(0, len(self.values))
                a = np.random.randint(0, self.action_space.n)
                samples.add((s, a))

            for s, a in samples:

                old_q_k = self.values[s][a]

                self.values[s][a] += self.alpha*(np.sum([(self.dyna_r[s][a][state] * self.gamma*np.max([value for value in self.values[state]])) * self.dyna_p[s][a][state] for state in self.dyna_r[s][a]]) - old_q_k )

if __name__ == '__main__':
    env,config,outdir,logger=init('./configs/config_qlearning_gridworld.yaml',"QLearning")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = QLearning(env, config)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    nb = 0
    for i in range(episode_count):
        checkConfUpdate(outdir, config)  # permet de changer la config en cours de run

        rsum = 0
        agent.nbEvents = 0
        ob = env.reset()

        if (i > 0 and i % int(config["freqVerbose"]) == 0):
            verbose = True
        else:
            verbose = False

        if i % freqTest == 0 and i >= freqTest:  ##### Si agent.test alors retirer l'exploration
            print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()
        new_ob = agent.storeState(ob)
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action = agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.storeState(new_ob)

            j+=1

            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                #print("forced done!")

            agent.store(ob, action, new_ob, reward, done, j)
            agent.learn(done)
            rsum += reward
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                break



    env.close()