import math
import random
from random import randint

from deap import base
from deap import creator
from deap import tools

import numpy as np
import matplotlib.pyplot as plt

import os

A = -32.768
B = 32.768
precision = 8

chromosome_size = math.ceil(
    math.log2((B - A) * (10 ** precision)) + math.log2(1)
)

try:
    os.mkdir("./binary")
except:
    print("Directory already exists.")

def generate_mean_plot(mean_list):
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(range(len(mean_list)), mean_list)
    plt.xlabel("epoch")
    plt.ylabel("mean")
    fig.savefig('./binary/mean.png')  # save the figure to file
    plt.close(fig)


def generate_standard_deviation_plot(standard_deviation_list):
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(range(len(standard_deviation_list)), standard_deviation_list)
    plt.xlabel("epoch")
    plt.ylabel("standard deviation")
    fig.savefig('./binary/standard_deviation.png')  # save the figure to file
    plt.close(fig)


def generate_best_value_plot(best_value):
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(range(len(best_value)), best_value)
    plt.xlabel("epoch")
    plt.ylabel("best value")
    fig.savefig('./binary/best_value.png')  # save the figure to file
    plt.close(fig)

def individual(icls):
    genome = list()
    for x in range(0, chromosome_size * 2):
        genome.append(randint(0, 1))

    return icls(genome)

def decodeInd(individual, a, b):
    decoded = list()
    chromosomes = [individual[:chromosome_size], individual[chromosome_size:]]
    for chromosome in chromosomes:
        binary_string_chromosome = ''.join(str(i) for i in chromosome)
        decimal_val = int(binary_string_chromosome, 2)
        decoded_chromosome = a + decimal_val * (b - a) / (2 ** chromosome_size - 1)
        decoded.append(decoded_chromosome)
    return decoded


def fitnessFunction(individual):
    ind = decodeInd(individual, A, B)
    # print(ind)

    result = -20 * np.exp(-0.2 * np.sqrt((1 / 2) * (ind[0] ** 2 + ind[1] ** 2))) - np.exp(
        (1 / 2) * (np.cos(2 * np.pi * ind[0]) + np.cos(2 * np.pi * ind[1]))) + 20 + np.exp(1)

    return result,

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


toolbox.register('individual', individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitnessFunction)

# selTournament(tournsize = ), selRandom(k = ), selBest(k = ), selWorst(k = ), selRoulette(k = )
toolbox.register("select", tools.selTournament, tournsize=3)

# cxOnePoint(ind1, ind2)
# cxUniform(ind1, ind2, indpb = Independent probability for each attribute to be exchanged)
# cxTwoPoint(ind1, ind2)
toolbox.register("mate", tools.cxTwoPoint)

# mutShuffleIndexes(indpd =), mutFlipBit(indpd = ),
toolbox.register("mutate", tools.mutFlipBit)

sizePopulation = 1000
probabilityMutation = 0.4
probabilityCrossover = 0.8
numberIteration = 100


pop = toolbox.population(n=sizePopulation)
fitness = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitness):
    ind.fitness.values = fit


g = 0
numberElitism = 1
mean_record = list()
std_record = list()
min_record = list()

while g < numberIteration:
    g = g + 1
    print("-- Generation %i --" % g)

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    listElitism = []

    for x in range(0, numberElitism):
        listElitism.append(tools.selBest(pop, 1)[0])

    # Apply crossover and mutation on the offsprings
    for child1, child2 in zip(offspring[::2], offspring[1::2]):

        # cross two individuals with probability CXPB
        if random.random() < probabilityCrossover:
            toolbox.mate(child1, child2)

            # fitness values of the children must be recalculated later
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        # mutate an individual with probability MUTPB
        if random.random() < probabilityMutation:
            toolbox.mutate(mutant, 0.5)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitness = map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitness):
        ind.fitness.values = fit

    print(" Evaluated %i individuals" % len(invalid_ind))
    pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    min_record.append(min(fits))
    mean_record.append(mean)
    std_record.append(std)

    print(" Min %s" % min(fits))
    print(" Max %s" % max(fits))
    print(" Avg %s" % mean)
    print(" Std %s" % std)
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s, \n %s" % (best_ind, best_ind.fitness.values, decodeInd(best_ind, A, B)))

    print("-- End of (successful) evolution --")

generate_mean_plot(mean_record)
generate_standard_deviation_plot(std_record)
generate_best_value_plot(min_record)

