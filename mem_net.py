import numpy as np, time
#import MultiNEAT as NEAT
import mod_mem_net as mod, sys
from random import randint#, choice
#from copy import deepcopy

class Deap_param:
    def __init__(self, angle_res, is_memoried):
        self.num_input = (360 * 2 / angle_res)
        self.num_hnodes = 5
        self.num_output = 5

        self.elite_fraction = 0.05
        self.crossover_prob = 0.2
        self.mutation_prob = 0.9
        if is_memoried:
            self.total_num_weights = 3 * (
                self.num_hnodes * (self.num_input + 1) + self.num_hnodes * (self.num_output + 1)) + 2 * self.num_hnodes * (
                self.num_hnodes + 1) + self.num_output * (self.num_hnodes + 1) + self.num_hnodes
        else:
            # Normalize network flexibility by changing hidden nodes
            #naive_total_num_weights = self.num_hnodes*(self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
	        self.total_num_weights = self.num_hnodes * (self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
            #continue
        #     mem_weights = 3 * (
        #         self.num_hnodes * (self.num_input + 1) + self.num_hnodes * (self.num_output + 1)) + 2 * self.num_hnodes * (
        #         self.num_hnodes + 1) + self.num_output * (self.num_hnodes + 1) + self.num_hnodes
        #     normalization_factor = int(mem_weights/naive_total_num_weights)
        #
        #     #Set parameters for comparable flexibility with memoried net
        #     self.num_hnodes *= normalization_factor + 1
        #     self.total_num_weights = self.num_hnodes * (self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
        # print 'Num parameters: ', self.total_num_weights

class Parameters:
    def __init__(self):
            self.population_size = 100
            self.grid_row = 15
            self.grid_col = 15
            self.total_steps = 25 # Total roaming steps without goal before termination
            self.num_predator = 5
            self.num_prey = 10
            self.predator_random = 1
            self.prey_random = 1
            self.total_generations = 10000
            self.angle_res = 45

            #DEAP stuff
            self.is_memoried_predator = 0
            self.is_memoried_prey = 0
            self.use_deap = 1
            if self.use_deap:
                self.deap_param_predator = Deap_param(self.angle_res, self.is_memoried_predator)
                self.deap_param_prey = Deap_param(self.angle_res, self.is_memoried_prey)


            #Tertiary Variables (Closed for cleanliness) Open to modify
            if True:
                # GIVEN

                self.D_reward = 1  # D reward scheme
                self.prey_global = 0 #Prey's reward scheme
                self.wheel_action = 0

                #TERTIARY
                self.domain_setup = 0
                self.aamas_domain = 0
                self.use_neat = 0  # Use NEAT VS. Keras based Evolution module
                self.obs_dist = 1  # Observe distance (Radius of prey that predators have to be in for successful observation)
                self.coupling = 1  # Number of predators required to simultaneously observe a prey
                self.use_py_neat = 0  # Use Python implementation of NEAT
                self.sensor_avg = True  # Average distance as state input vs (min distance by default)
                self.state_representation = 1  # 1 --> Angled brackets, 2--> List of predator/preys


                #EV0-NET
                self.use_hall_of_fame = 0
                self.hof_weight = 0.99
                self.leniency = 1  # Fitness calculation based on leniency vs averaging

                if self.use_neat: #Neat
                    if self.use_py_neat: #Python NEAT
                        self.py_neat_config = Py_neat_params()

                    else: #Multi-NEAT parameters
                        import MultiNEAT as NEAT
                        self.params = NEAT.Parameters()
                        self.params.PopulationSize = self.population_size
                        self.params.fs_neat = 1
                        self.params.evo_hidden = 10
                        self.params.MinSpecies = 5
                        self.params.MaxSpecies = 15
                        self.params.EliteFraction = 0.05
                        self.params.RecurrentProb = 0.2
                        self.params.RecurrentLoopProb = 0.2

                        self.params.MaxWeight = 8
                        self.params.MutateAddNeuronProb = 0.01
                        self.params.MutateAddLinkProb = 0.05
                        self.params.MutateRemLinkProb = 0.01
                        self.params.MutateRemSimpleNeuronProb = 0.005
                        self.params.MutateNeuronActivationTypeProb = 0.005

                        self.params.ActivationFunction_SignedSigmoid_Prob = 0.01
                        self.params.ActivationFunction_UnsignedSigmoid_Prob = 0.5
                        self.params.ActivationFunction_Tanh_Prob = 0.1
                        self.params.ActivationFunction_SignedStep_Prob = 0.1
                else: #Use keras
                    self.keras_evonet_hnodes = 25  # Keras based Evo-net's hidden nodes

class Py_neat_params:
    def __init__(self):

        #[Types]
        self.stagnation_type = 'DefaultStagnation'
        self.reproduction_type = 'DefaultReproduction'

        #Phenotype
        self.input_nodes = 21
        self.hidden_nodes = 0
        self.output_nodes = 19
        self.initial_connection = 'fs_neat' #['unconnected', 'fs_neat', 'fully_connected', 'partial']
        self.max_weight = 0.1
        self.min_weight = -0.1
        self.feedforward = 0
        self.activation_functions = 'sigmoid'
        self.weight_stdev = 0.1

        #genetic
        self.pop_size = 500
        self.max_fitness_threshold = 1000000
        self.prob_add_conn = 0.1
        self.prob_add_node = 0.05
        self.prob_delete_conn = 0.01
        self.prob_delete_node = 0.01
        self.prob_mutate_bias = 0.05
        self.bias_mutation_power = 1.093
        self.prob_mutate_response = 0.1
        self.response_mutation_power = 0.1
        self.prob_mutate_weight = 0.2
        self.prob_replace_weight = 0.05
        self.weight_mutation_power = 1
        self.prob_mutate_activation = 0.08
        self.prob_toggle_link = 0.05
        self.reset_on_extinction = 1

        #genotype compatibility
        self.compatibility_threshold = 3.0
        self.excess_coefficient = 1.4
        self.disjoint_coefficient = 1.3
        self.weight_coefficient = 0.7

        self.species_fitness_func = 'mean'
        self.max_stagnation = 15

        self.elitism = 1
        self.survival_threshold = 0.2

parameters = Parameters() #Create the Parameters class
tracker = mod.statistics(parameters) #Initiate tracker
gridworld = mod.Gridworld (parameters)  # Create gridworld


def test_policies(save_name = 'trajectory.csv'):
    gridworld.load_test_policies() #Load test policies from best_team folder Assumes perfect sync
    fake_team = np.zeros(parameters.num_predators_scout + parameters.num_predators_service_bot)  # Fake team for test_phase
    best_reward = -10000
    for i in range(parameters.population_size):
        reward, global_reward, trajectory_log  = run_simulation(parameters, gridworld, fake_team, is_test=True)
        if global_reward > best_reward:
            comment = [parameters.num_predators_scout, parameters.num_predators_service_bot, parameters.num_prey] + [0] * (len(trajectory_log[0])-3)
            trajectory_log = [comment] + trajectory_log
            trajectory_log = np.array(trajectory_log)
            np.savetxt(save_name, trajectory_log, delimiter=',',fmt='%10.5f')
            print reward, global_reward
            best_reward = global_reward

def best_performance_trajectory(parameters, gridworld, teams, save_name='best_performance_traj.csv'):
    trajectory_log = []
    gridworld.reset(teams)  # Reset board
    # mod.dispGrid(gridworld)
    for steps in range(parameters.total_steps):  # One training episode till goal is not reached
        ig_traj_log = []
        for id, predator in enumerate(gridworld.predator_list_scout):  # get all the action choices from the predators
            if steps == 0: predator.perceived_state = gridworld.get_state(predator, is_Scout=True)  # Update all predator's perceived state
            predator.take_action(teams[id])  # Make the predator take action using the Evo-net with given id from the population
            ig_traj_log.append(predator.position[0])
            ig_traj_log.append(predator.position[1])

        for id, predator in enumerate(gridworld.predator_list_service_bot):  # get all the action choices from the predators
            if steps == 0: predator.perceived_state = gridworld.get_state(predator, is_Scout=False)  # Update all predator's perceived state
            predator.take_action(teams[id + parameters.num_predators_scout])  # Make the predator take action using the Evo-net with given id from the population
            ig_traj_log.append(predator.position[0])
            ig_traj_log.append(predator.position[1])



        for prey in gridworld.prey_list:
            ig_traj_log.append(prey.position[0])
            ig_traj_log.append(prey.position[1])

        trajectory_log.append(np.array(ig_traj_log))
        gridworld.move()  # Move gridworld

        # mod.dispGrid(gridworld)
        # raw_input('E')

        gridworld.update_prey_observations()  # Figure out the prey observations and store all credit information
        if parameters.is_prey_move: gridworld.prey_move()

        for id, predator in enumerate(gridworld.predator_list_scout): predator.referesh(teams[id], gridworld)  # Update state and learn if applicable
        for id, predator in enumerate(gridworld.predator_list_service_bot): predator.referesh(teams[id], gridworld)  # Update state and learn if applicable

        if gridworld.check_goal_complete(): break  # If all prey's observed

    # Log final position
    ig_traj_log = []
    for predator in gridworld.predator_list_scout:
        ig_traj_log.append(predator.position[0])
        ig_traj_log.append(predator.position[1])

    for predator in gridworld.predator_list_service_bot:
        ig_traj_log.append(predator.position[0])
        ig_traj_log.append(predator.position[1])

    for prey in gridworld.prey_list:
        ig_traj_log.append(prey.position[0])
        ig_traj_log.append(prey.position[1])
    trajectory_log.append(np.array(ig_traj_log))
    rewards, global_reward = gridworld.get_reward(teams)

    #Save trajectory to file
    comment = [parameters.num_predators_scout, parameters.num_predators_service_bot, parameters.num_prey] + [0] * (
    len(trajectory_log[0]) - 3)
    trajectory_log = [comment] + trajectory_log
    trajectory_log = np.array(trajectory_log)
    np.savetxt(save_name, trajectory_log, delimiter=',', fmt='%10.5f')

num_evals = 5
def evolve(gridworld, parameters, generation, best_hof_score):
    avg_global = 0.0
    gridworld.new_epoch_reset() #Reset initial random positions for the epoch

    # Get new genome list and fitness evaluations trackers
    for i in range(parameters.num_predator): gridworld.predator_list[i].evo_net.referesh_genome_list()
    for i in range(parameters.num_prey): gridworld.prey_list[i].evo_net.referesh_genome_list()

    #MAIN LOOP
    for genome_ind in range(parameters.population_size): #For evaluation
        #SIMULATION AND TRACK REWARD
        rewards, global_reward = run_simulation(parameters, gridworld, genome_ind) #Returns rewards for each member of the team

        avg_global += global_reward

        #ENCODE FITNESS BACK TO predator
        for id, predator in enumerate(gridworld.predator_list):
            ig_reward = rewards[id]
            predator.evo_net.fitness_evals[genome_ind].append(ig_reward) #Assign those rewards
        # ENCODE FITNESS BACK TO prey
        for id, prey in enumerate(gridworld.prey_list):
            if parameters.prey_global:
                prey.evo_net.fitness_evals[genome_ind].append(-global_reward) #Assign those rewards
            else: #Local
                if prey.is_caught: prey.evo_net.fitness_evals[genome_ind].append(0)
                else: prey.evo_net.fitness_evals[genome_ind].append(1)


    #if generation % 25 == 0: gridworld.save_pop() #Save population periodically

    for predator in gridworld.predator_list:
        predator.evo_net.update_fitness()# Assign fitness to genomes #
        #predator.evo_net.epoch() #Run Epoch update in the population
    for prey in gridworld.prey_list:
        prey.evo_net.update_fitness()# Assign fitness to genomes #
        prey.evo_net.epoch() #Run Epoch update in the population


    return avg_global/parameters.population_size

def run_simulation(parameters, gridworld, genome_ind, is_test = False): #Run simulation given a team and return fitness for each individuals in that team

    #if is_test: trajectory_log = []
    gridworld.reset(genome_ind)  # Reset board and build net
    #mod.dispGrid(gridworld)

    for steps in range(parameters.total_steps):  # One training episode till goal is not reached
        if is_test: ig_traj_log = []

        for id, predator in enumerate(gridworld.predator_list):  #get all the action choices from the predators
            predator.perceived_state = gridworld.get_state(predator) #Update all predator's perceived state
            if is_test:
                predator.take_action_test()
                ig_traj_log.append(predator.position[0])
                ig_traj_log.append(predator.position[1])
            else:
                predator.take_action() #Make the predator take action using the Evo-net with given id from the population

        for id, prey in enumerate(gridworld.prey_list):  #get all the action choices from the predators
            prey.perceived_state = gridworld.get_state(prey) #Update all predator's perceived state
            if is_test:
                prey.take_action_test()
                ig_traj_log.append(prey.position[0])
                ig_traj_log.append(prey.position[1])
            else:
                prey.take_action() #Make the predator take action using the Evo-net with given id from the population

        gridworld.move() #Move gridworld

        # mod.dispGrid(gridworld)
        # raw_input('E')
        gridworld.update_prey_observations() #Figure out the prey observations and store all credit information




    #Log final position
    if is_test:
        ig_traj_log = []
        for predator in gridworld.predator_list_scout:
            ig_traj_log.append(predator.position[0])
            ig_traj_log.append(predator.position[1])

        for predator in gridworld.predator_list_service_bot:
            ig_traj_log.append(predator.position[0])
            ig_traj_log.append(predator.position[1])

        for prey in gridworld.prey_list:
            ig_traj_log.append(prey.position[0])
            ig_traj_log.append(prey.position[1])
        trajectory_log.append(np.array(ig_traj_log))

    rewards, global_reward = gridworld.get_reward()

    if is_test:
        #trajectory_log = np.array(trajectory_log)
        return rewards, global_reward, trajectory_log
    return rewards, global_reward

if __name__ == "__main__":

    mod.dispGrid(gridworld)
    best_hof_score = 0

    for gen in range (parameters.total_generations): #Main Loop
        curtime = time.time()
        avg_global = evolve(gridworld, parameters, gen, best_hof_score) #CCEA
        tracker.add_fitness(avg_global, gen) #Add best global performance to tracker
        elapsed = time.time() - curtime

        if parameters.use_neat and not parameters.use_py_neat :
            print 'Gen:', gen, ' D' if parameters.D_reward else ' G',  ' Best g_reward', int(avg_global * 100), ' Avg:', int(100 * tracker.avg_fitness), '  BEST HOF SCORE: ', best_hof_score,  '  Fuel:', 'ON' if parameters.is_fuel else 'Off' , '  Time_offset type:', 'Hard' if parameters.is_hard_time_offset else 'Soft', 'Time_offset: ', parameters.time_offset #, 'Delta MPC:', int(tracker.avg_mpc), '+-', int(tracker.mpc_std), 'Elapsed Time: ', elapsed #' Delta generations Survival: '      #for i in range(num_predators): print all_pop[i].delta_age / params.PopulationSize,

        else:
            print 'Gen:', gen, 'Memoried' if parameters.is_memoried_predator or parameters.is_memoried_prey  else 'Normal', ' Best global', int(avg_global * 100), ' Avg:', int(100 * tracker.avg_fitness)#, 'Best hof_score: ', best_hof_score













