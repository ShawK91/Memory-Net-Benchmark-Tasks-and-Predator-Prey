import numpy as np, os, math
#import MultiNEAT as NEAT
import mod_mem_net as mod, sys
from random import randint, shuffle

print 'Running T-MAZE'

#TODO EVOLUTION META IN SSNE

save_foldername = 'Res_T-maze'
class tracker(): #Tracker
    def __init__(self, parameters, foldername = save_foldername):
        self.foldername = foldername
        self.best_rew = []; self.avg_best_rew = 0; self.tr_avg_best_rew = []
        self.best_opt = []; self.avg_best_opt = 0; self.tr_avg_best_opt = []
        self.hof_rew = []; self.hof_avg_rew = 0; self.hof_tr_avg_rew = []
        self.hof_opt = []; self.hof_avg_opt = 0; self.hof_tr_avg_opt = []
        if not os.path.exists(foldername):
            os.makedirs(foldername)


    def add_best_reward(self, fitness, generation):
        self.best_rew.append(fitness)
        if len(self.best_rew) > 100:
            self.best_rew.pop(0)
        self.avg_best_rew = sum(self.best_rew)/len(self.best_rew)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/best_reward_tmaze.csv'
            self.tr_avg_best_rew.append(np.array([generation, self.avg_best_rew]))
            np.savetxt(filename, np.array(self.tr_avg_best_rew), fmt='%.3f', delimiter=',')

    def add_best_optimality(self, fitness, generation):
        self.best_opt.append(fitness)
        if len(self.best_opt) > 100:
            self.best_opt.pop(0)
        self.avg_best_opt = sum(self.best_opt)/len(self.best_opt)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/best_optimality_tmaze.csv'
            self.tr_avg_best_opt.append(np.array([generation, self.avg_best_opt]))
            np.savetxt(filename, np.array(self.tr_avg_best_opt), fmt='%.3f', delimiter=',')

    def add_hof_reward(self, fitness, generation):
        self.hof_rew.append(fitness)
        if len(self.hof_rew) > 100:
            self.hof_rew.pop(0)
        self.hof_avg_rew= sum(self.hof_rew)/len(self.hof_rew)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/hof_reward_tmaze.csv'
            self.hof_tr_avg_rew.append(np.array([generation, self.hof_avg_rew]))
            np.savetxt(filename, np.array(self.hof_tr_avg_rew), fmt='%.3f', delimiter=',')

    def add_hof_optimality(self, fitness, generation):
        self.hof_opt.append(fitness)
        if len(self.hof_opt) > 100:
            self.hof_opt.pop(0)
        self.hof_avg_opt = sum(self.hof_opt)/len(self.hof_opt)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/hof_optimality_tmaze.csv'
            self.hof_tr_avg_opt.append(np.array([generation, self.hof_avg_opt]))
            np.savetxt(filename, np.array(self.hof_tr_avg_opt), fmt='%.3f', delimiter=',')



    def save_csv(self, generation, filename):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

class SSNE_param:
    def __init__(self, is_memoried):
        self.num_input = 6
        self.num_hnodes = 20
        self.num_output = 2
        if is_memoried: self.type_id = 'memoried'
        else: self.type_id = 'normal'

        self.elite_fraction = 0.1
        self.crossover_prob = 0.1
        self.mutation_prob = 0.9
        self.weight_magnitude_limit = 1000000000000
        if is_memoried:
            self.total_num_weights = 3 * (
                self.num_hnodes * (self.num_input + 1) + self.num_hnodes * (self.num_output + 1)) + 2 * self.num_hnodes * (
                self.num_hnodes + 1) + self.num_output * (self.num_hnodes + 1) + self.num_hnodes
        else:
            #Normalize network flexibility by changing hidden nodes
            naive_total_num_weights = self.num_hnodes*(self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
            #self.total_num_weights = self.num_hnodes * (self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
            #continue
            mem_weights = 3 * (
                 self.num_hnodes * (self.num_input + 1) + self.num_hnodes * (self.num_output + 1)) + 2 * self.num_hnodes * (
                 self.num_hnodes + 1) + self.num_output * (self.num_hnodes + 1) + self.num_hnodes
            normalization_factor = int(mem_weights/naive_total_num_weights)

            #Set parameters for comparable flexibility with memoried net
            self.num_hnodes *= normalization_factor + 1
            self.total_num_weights = self.num_hnodes * (self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
        print 'Num parameters: ', self.total_num_weights

class Parameters:
    def __init__(self):
            self.population_size = 100
            self.is_memoried = 1
            self.num_local_inits = 4
            self.num_trials = 10

            #DEAP/SSNE stuff
            self.use_ssne = 1
            if self.use_ssne:
                self.ssne_param = SSNE_param( self.is_memoried)
            self.total_gens = 10000

            #Reward scheme
            #1 Coarse and Order matters
            #2 Coarse and Order doesn't matter - evaluate remember performance
            #3 Binary and Order matters
            #4 Binary and Order doesn't matter - evaluate remember performance
            #5 Test - final performance only matters
            #6 Combine #2 and #3
            self.reward_scheme = 6

            self.tolerance = 1
            self.test_tolerance = 1
parameters = Parameters() #Create the Parameters class
tracker = tracker(parameters) #Initiate tracker

class T_maze:
    def __init__(self, parameters):
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param
        self.total_steps = 20
        self.high_reward_local = None

        #initialize reward_local_dist
        self.reward_local_dist = []
        for local in range(4):
            for i in range(self.parameters.num_local_inits/4): self.reward_local_dist.append(local+1)


        #Agent stuff
        self.agent = mod.SSNE(self.parameters, self.ssne_param)
        self.start = [24, 17]  # Start coordinates
        self.agent_pos = self.start[:]
        self.state = np.zeros(6)

    def is_collided(self, x, y):
        if x < 0 or y < 0 or x > 25 or y > 35: return False #Main boundary

        if x < 10 and y > 5 and y < 30: return False #Region 1 unfeasibility

        if x > 15:
            if y > 5 and y < 15 or y > 20 and y < 30: return False #Region 2 and 3 unfeasibility

        return True

    def get_dist(self, x, y, x2, y2):
        return math.sqrt((x-x2) * (x-x2) + (y - y2) * (y - y2))

    def move(self, action):
        self.agent_pos[0] += 4*(action[0][0] - 0.5)
        self.agent_pos[1] += 4*(action[1][0] - 0.5)
        return self.is_collided(self.agent_pos[0], self.agent_pos[1]) #Collision

    def set_state(self, x, y):
        agent_intent = 0
        self.state[4] = 0 #Junction state initialize to false
        self.state[5] = 0 #Reward identity initialized to false

        if y > 5 and y < 30: #Not the side hallways
            self.state[0] = x - 10

            if y > 15 and y < 20: #Start passageway
                self.state[2] = 25 - x
                if x > 15: #Not first junction
                    self.state[1] = 20 - y
                    self.state[3] = y - 15
                else: #First junction
                    self.state[1] = 35 - y
                    self.state[3] = y
                    self.state[4] = 1 #Junction state


            else: #Main central passageway minus the central junction
                self.state[1] = 35 - y
                self.state[2] = 15 - x
                self.state[3] = y

        if y < 5: #Left hallway

            if x > 10 and x < 15: #Junction
                self.state[0] = x
                self.state[1] = 35 - y
                self.state[2] = 25 - x
                self.state[3] = y
                self.state[4] = 1 #Junction state

            else: #Not junction
                self.state[0] = x
                self.state[1] = 5 - y
                self.state[2] = 25 - x
                self.state[3] = y

                # Reward
                if self.get_dist(x, y, 5 , 2) < 4:
                    if self.high_reward_local == 1: self.state[5] = 1
                    else: self.state[5] = 0.1
                    agent_intent = 1
                    print 'GOAL'

                if self.get_dist(x, y, 20, 2) < 4:
                    if self.high_reward_local == 2: self.state[5] = 1
                    else: self.state[5] = 0.1
                    agent_intent = 2
                    print 'GOAL'

        if y > 30:  # Right hallway

            if x > 10 and x < 15:  # Junction
                self.state[0] = x
                self.state[1] = 35 - y
                self.state[2] = 25 - x
                self.state[3] = y
                self.state[4] = 1  # Junction state

            else:  # Not junction
                self.state[0] = x
                self.state[1] = 35 - y
                self.state[2] = 25 - x
                self.state[3] = y - 30

                # Reward
                if self.get_dist(x, y, 5, 32) < 4:
                    if self.high_reward_local == 3: self.state[5] = 1
                    else: self.state[5] = 0.1
                    agent_intent = 3
                    print 'GOAL'

                if self.get_dist(x, y, 20, 32) < 4:
                    if self.high_reward_local == 1: self.state[5] = 1
                    agent_intent = 4
                    print 'GOAL'

        return agent_intent

    def run_simulation(self, index):
        sim_reward = 0.0
        sim_optimality = 0.0

        for local_inits in range(self.parameters.num_local_inits): #Reward inits uniform distribution
            agent_reward_trail = []  # Tracks agent's reward goals to measure optimality
            real_reward_trail = [] #tracks the real high reward locals
            self.high_reward_local = self.reward_local_dist[local_inits]

            for trial in range(self.parameters.num_trials):
                self.agent.pop[index].reset_bank()
                #Switch high reward location midway
                if trial == int(self.parameters.num_trials/2):
                    while True:
                        new_loc = randint(1,4)
                        if new_loc != self.high_reward_local:
                            self.high_reward_local = new_loc
                            break
                self.agent_pos = self.start[:] #Reset agent start location

                #Start a trial
                agent_intent = 0
                for steps in range(self.total_steps): #Start trial
                    agent_intent = self.set_state(self.agent_pos[0], self.agent_pos[1]) #Set state

                    if self.state[5] != 0: #Reward collected
                        #Update sim_reward
                        sim_reward += self.state[5]
                        break

                    action = self.agent.pop[index].feedforward(self.state)
                    if self.move(action): #Collsion
                        sim_reward -= 0
                        break

                # Update optimality metrics
                agent_reward_trail.append(agent_intent)
                real_reward_trail.append(self.high_reward_local)
                if agent_intent == self.high_reward_local: #Vacuously optimal
                    sim_optimality += 1
                    continue
                elif agent_intent == 0: #vacuously suboptimal
                    continue
                else:
                    exploration = [] #Tracks exploration activity
                    for i,j in zip(agent_reward_trail[-4:], real_reward_trail[-4:]):
                        if i == j: exploration = []
                        else: exploration.append(i)

                    if len(exploration) == len(set(exploration)): #if exploration is unique
                        sim_optimality += 1

        return sim_reward/(self.parameters.num_local_inits * self.parameters.num_trials), sim_optimality/(self.parameters.num_local_inits * self.parameters.num_trials)

    def evolve(self, gen):
        best_epoch_reward = -10000000000; best_epoch_optimality = -10000000000
        shuffle(self.reward_local_dist) #Randomize reward local distribution

        for i in range(self.parameters.population_size): #Test all genomes/individuals
            reward, optimality = self.run_simulation(i)
            self.agent.fitness_evals[i] = optimality + reward
            if reward > best_epoch_reward: best_epoch_reward = reward
            if optimality > best_epoch_optimality: best_epoch_optimality = optimality

        #HOF test net
        hof_index = self.agent.fitness_evals.index(max(self.agent.fitness_evals))
        hof_reward, hof_optimality = self.run_simulation(hof_index)
        #print self.agent.fitness_evals

        #Save population and HOF
        if (gen + 1) % 1 == 0:
            mod.pickle_object(self.agent.pop, save_foldername + '/t-maze_pop')
            mod.pickle_object(self.agent.pop[hof_index], save_foldername + '/t-maze_hof')
            np.savetxt(save_foldername + '/gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')

        self.agent.epoch()
        return best_epoch_reward, best_epoch_optimality, hof_reward, hof_optimality


if __name__ == "__main__":

    task = T_maze(parameters)
    for gen in range(parameters.total_gens):
        best_epoch_reward, best_epoch_optimality, hof_reward, hof_optimality = task.evolve(gen)

        tracker.add_best_reward(best_epoch_reward, gen)  # Add best epoch reward
        tracker.add_best_optimality(best_epoch_optimality, gen)  # Add best epoch optimality
        tracker.add_hof_reward(best_epoch_reward, gen)  # Add HOF epoch reward
        tracker.add_hof_optimality(best_epoch_optimality, gen)  # Add HOF epoch optimality
        print 'Gen:', gen, ' Epoch_rew:', "%0.3f" % best_epoch_reward, ' Epoch_opt:', "%0.2f" % best_epoch_optimality, ' HOF_Rew:', "%0.2f" % hof_reward, ' HOF_opt:', "%0.2f" % hof_optimality #,  'Cml_rew:', "%0.3f" % tracker.avg_best_rew, 'Cml_opt:'


















