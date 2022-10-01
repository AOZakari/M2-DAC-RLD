import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy


class PolicyAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space, states, mdp, alpha = 0.99, eps= 10e-4):
        print("=============== INIT OF POLICY AGENT =====================")
        self.action_space = action_space
        self.states = range(len(states))
        self.non_final_states = [t[0] for t in list(mdp.items())]
        #self.final_states = self.states - self.non_final_states
        self.policy = {state: self.action_space.sample() for state in self.states}
        self.mdp = mdp
        policy_old = {state: self.action_space.sample() for state in self.states}
        while (policy_old == self.policy):
            policy_old = {state: self.action_space.sample() for state in self.states} # On s'assure qu'on n'a pas la même initialization

        # Init de V pour les états terminaux

        self.V = np.zeros(len(states))
        for key in mdp:
            for action, transitions in mdp[key].items():
                for transition in transitions:
                    if transition[3] == True:
                        self.V[transition[1]] = transition[2]

        print(self.V)

        V_old = np.random.rand(len(states))
        while (np.linalg.norm(self.V-V_old)<=eps):
            V_old = np.random.rand(len(states)) # On s'assure qu'on n'a pas la même initialization

        training = 0
        while (policy_old != self.policy):
            policy_old = self.policy
            while (np.linalg.norm(self.V-V_old)>eps):
                V_old = self.V
                for s in self.non_final_states:
                    transitions = self.mdp[s][self.policy[s]]
                    self.V[s] = np.sum([t[0]*(t[2] + alpha*V_old[t[1]]) for t in transitions])
            for s in self.non_final_states:
                self.policy[s] = np.argmax([np.sum([t[0]*(t[2] + alpha*self.V[t[1]]) for t in self.mdp[s][a]]) for a in range(self.action_space.n)])
            print("EPOCH: ", training)
            training += 1

        print("=============== POLICY AGENT READY! =====================")
        print(self.V)


    def act(self, observation):
        return self.policy[observation]


if __name__ == '__main__':


    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan4.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    
    env.seed(0)  # Initialise le seed du pseudo-random
    env.reset()
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human") #visualisation sur la console
    states, mdp = env.getMDP()  # recupere le mdp et la liste d'etats
    print("Nombre d'etats : ",len(states))
    state, transitions = list(mdp.items())[0]
    print(state)  # un etat du mdp
    print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
    agent = PolicyAgent(env.action_space, states, mdp, 0.999999)


    episode_count = 1000
    reward = 0
    done = False
    rsum = 0

    for i in range(episode_count):
        obs = env.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        while True:
            action = agent.act(env.getStateFromObs(obs))
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()