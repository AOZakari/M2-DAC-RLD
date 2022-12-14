import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
import highway_env
from matplotlib import pyplot as plt
import yaml
from datetime import datetime
import copy
from memory import Memory


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, env, opt):
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.nbEvents=0


    def act(self, obs):
        print(self.featureExtractor.getFeatures(obs).shape)
        a=self.action_space.sample()
        return a

    # sauvegarde du modèle
    def save(self,outputDir):
        pass

    # chargement du modèle.
    def load(self,inputDir):
        pass

    # apprentissage de l'agent. Dans cette version rien à faire
    def learn(self):
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            pass
        pass

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            tr = (ob, action, reward, new_ob, done)
            self.lastTransition=tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0


class DQNAgent(object):
    """The world's not so simplest agent!"""

    def __init__(self, env, opt):
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.nbEvents=0
        self.nbFeats = self.featureExtractor.getFeatures(env.reset()).shape[1]
        self.Q = NN(self.nbFeats, self.action_space.n,[200])
        self.Q_c = copy.deepcopy(self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=opt.learningRate)
        self.memSize = opt.memSize
        self.mem = Memory(opt.memSize, False)
        self.explore = opt.explore
        self.decay = opt.decay
        self.criterion = torch.nn.SmoothL1Loss()
        self.batchSize = opt.batchSize
        self.discount = opt.discount
        self.resetQ = opt.resetQ
        self.currNbOpt = 0
        self.nbOpt = 0

    def act(self, obs):

        obs = torch.tensor(obs).squeeze()
        if(np.random.random()<self.explore):
            a = self.action_space.sample()

        else:
            with torch.no_grad():
                pred = self.Q(obs)
            a = torch.argmax(pred).item()

        if self.explore>0.05:
            self.explore *= self.decay
        return a

    # sauvegarde du modèle
    def save(self,outputDir):
        pass

    # chargement du modèle.
    def load(self,inputDir):
        pass

    def get_y(self,transition):
        if transition[4]: #Si done
            return transition[2] # on retourne le reward
        
        #Sinon
        return transition[2] + self.discount * torch.max(self.Q_c(torch.tensor(transition[3]).detach().squeeze()))
    # apprentissage de l'agent. Dans cette version rien à faire

    def learn(self):
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            pass

        
        if self.memSize>0:
            _, _, batch = self.mem.sample(self.batchSize)
            x = torch.tensor([tr[0] for tr in batch]).squeeze()
            y = torch.tensor([self.get_y(tr) for tr in batch])
            pred = self.Q(x)[range(self.batchSize), torch.tensor([tr[1] for tr in batch])]
            loss = self.criterion(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
        else:
            x = self.lastTransition[0]
            y = self.get_y(self.lastTransition)
            pred = self.Q(x)
            self.optimizer.zero_grad()
            loss = self.criterion(pred, y)
        self.optimizer.step()
        
        if self.currNbOpt > self.resetQ:
            self.Q_c = copy.deepcopy(self.Q)
            self.currNbOpt = 0


        logger.direct_write("lossTrain", loss, self.nbOpt)
        self.currNbOpt += 1
        self.nbOpt += 1 
        


    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            tr = (ob, action, reward, new_ob, done)
            self.lastTransition=tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
            self.mem.store(tr)

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self, done):
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0


if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_random_lunar.yaml', "DQNAgent")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = DQNAgent(env,config)


    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    for i in range(episode_count):
        checkConfUpdate(outdir, config)

        rsum = 0
        agent.nbEvents = 0
        ob = env.reset()

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()

        new_ob = agent.featureExtractor.getFeatures(ob)
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action= agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)

            j+=1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            agent.store(ob, action, new_ob, reward, done,j)
            rsum += reward

            if agent.timeToLearn(done):
                agent.learn()
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break

    env.close()
