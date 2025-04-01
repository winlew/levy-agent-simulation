import numpy as np
import torch
from agent import *

class EvolutionaryAlgorithm:
    """
    Evolutionary Algorithm class that trains the agent population by evolving them.

    """

    def __init__(self, params):
        """
        Args:
            params (Params): class that stores parameters of the simulation
                population_size (int): number of agents in the population
                elite_fraction (float): fraction of the best performing agents that will be copied to the next generation
                mutation_fraction (float): fraction of the population that will be mutated
                mutation_rate (float): probability of a weight being mutated
                mutation_strength (float): strength of the mutation
                tolerance (float): tolerance value to set weights
        """
        self.population_size = params.population_size
        self.elite_fraction = params.elite_fraction
        self.mutation_fraction = params.mutation_fraction
        self.mutation_rate = params.mutation_rate
        self.mutation_strength = params.mutation_strength
        self.tolerance = params.tolerance

    def evolve(self, population):
        """
        Evolve the population of agents by:
        - copying the best performing agents to the next generation
        - mutating a fraction of the population
        - recombining the remaining agents
        Only the elite of the current population serves as parents for the next generation.

        Args:
            population (list): list of agents (expected to be sorted by fitness and to be of type RnnAgent)
        
        Returns:
            list: list of new agents
        """

        descendants = []
        num_parents = int(self.population_size * self.elite_fraction)
        parents = population[:num_parents]

        # ELITISM
        # copy parents to the next generation, but set small weights to zero
        for parent in parents:
            child_model = Rnn(parent.model.params)
            for child_weights, parent_weights in zip(child_model.parameters(), parent.model.parameters()):
                parent_weights = self.zero_out_small_weights(parent_weights)
                child_weights.data.copy_(parent_weights)
            child = RnnAgent(parent.params, child_model)
            descendants.append(child)

        # MUTATION
        # mutate some of the best performing parents
        capped_elite_fraction = 0.2 if population[0].params.elite_fraction >= 0.2 else population[0].params.elite_fraction
        for _ in range(int(self.population_size * self.mutation_fraction)):
            parent = np.random.choice(parents[:int(capped_elite_fraction*self.population_size)])
            child_model = Rnn(parent.model.params)
            for child_weights, parent_weights in zip(child_model.parameters(), parent.model.parameters()):
                child_weights.data.copy_(parent_weights)
            child_model = self.mutate(child_model)
            child = RnnAgent(parent.params, child_model)
            descendants.append(child)

        # CROSSOVER
        # the rest is filled up with recombined agents from the best performing parents
        while(len(descendants) < self.population_size):
            parent1 = np.random.choice(descendants)
            parent2 = np.random.choice(descendants)
            child_model = self.crossover(parent1.model, parent2.model)
            child = RnnAgent(parent1.params, child_model)
            descendants.append(child)

        return descendants
    

    def mutate(self, model):
        """
        Mutate the model weights by adding gaussian noise to a fraction of the weights.
        Makes sure that the weights are within the range [-1, 1].

        Args:
            model (Rnn): model to be mutated

        Returns:
            Rnn: mutated model
        """
        for weights in model.parameters():
            # choose randomly which weights will be mutated
            mask = (torch.rand(weights.size()) < self.mutation_rate).float()
            # gaussion noise
            weights.data += torch.randn_like(weights) * mask * self.mutation_strength
            # set every mutated weight above 1 to 1 and below -1 to -1
            weights.data = torch.min(torch.ones_like(weights.data), weights.data)
            weights.data = torch.max(-torch.ones_like(weights.data), weights.data)
            weights = self.zero_out_small_weights(weights)
        return model

    def crossover(self, model1, model2):
        """
        Randomly recombine parent model weights.
        Weights are randomly chosen from either parent. 
        There is no guarantee that the same amount of weights from each parent will be chosen.

        Args:
            model1 (Rnn): first parent model
            model2 (Rnn): second parent model

        Returns:
            Rnn: child model
        """
        child_model = Rnn(model1.params)
        for child_weights, parent1_weights, parent2_weights in zip(child_model.parameters(), model1.parameters(), model2.parameters()):
            mask = (torch.rand(parent1_weights.data.size()) < 0.5).float()
            child_weights.data.copy_(mask * parent1_weights.data + (1 - mask) * parent2_weights.data)
        return child_model

    def zero_out_small_weights(self, weights):
        """
        Set weights that are smaller than the tolerance to zero.
        """
        close_to_zero = np.isclose(weights, 0, atol=self.tolerance)
        weights[close_to_zero] = 0
        return weights


if __name__ == '__main__':
    pass