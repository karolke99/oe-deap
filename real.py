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

try:
    os.mkdir("./real")
except:
    print("Directory already exists.")


def generate_mean_plot(mean_list):
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(range(len(mean_list)), mean_list)
    plt.xlabel("epoch")
    plt.ylabel("mean")
    fig.savefig('./real/mean.png')  # save the figure to file
    plt.close(fig)


def generate_standard_deviation_plot(standard_deviation_list):
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(range(len(standard_deviation_list)), standard_deviation_list)
    plt.xlabel("epoch")
    plt.ylabel("standard deviation")
    fig.savefig('./real/standard_deviation.png')  # save the figure to file
    plt.close(fig)


def generate_best_value_plot(best_value):
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(range(len(best_value)), best_value)
    plt.xlabel("epoch")
    plt.ylabel("best value")
    fig.savefig('./real/best_value.png')  # save the figure to file
    plt.close(fig)


def individual(icls):
    genome = list()
    genome.append(random.uniform(A, B))
    genome.append(random.uniform(A, B))

    return icls(genome)


def fitnessFunction(individual):
    ind = individual

    result = -20 * np.exp(-0.2 * np.sqrt((1 / 2) * (ind[0] ** 2 + ind[1] ** 2))) - np.exp(
        (1 / 2) * (np.cos(2 * np.pi * ind[0]) + np.cos(2 * np.pi * ind[1]))) + 20 + np.exp(1)

    return result,


def cxArithmetic(ind1, ind2, k=None):
    if k is None:
        k = np.random.uniform(0., 1.)

    while True:
        new_ind1 = [
            (k * ind1[0]) + ((1 - k) * ind2[0]),
            (k * ind1[1]) + ((1 - k) * ind2[1])
        ]

        new_ind2 = [
            ((1 - k) * ind1[0]) + (k * ind2[0]),
            ((1 - k) * ind1[1]) + (k * ind2[1])
        ]

        if any(ind1) < A or any(ind2) > B or any(ind2) < A or any(ind1) > B:
            k = np.random.uniform(0., 1.)
            continue
        else:
            break

    return new_ind1, new_ind2


def cxLinear(ind1, ind2):
    z = ([
        (0.5 * ind1[0]) + (0.5 * ind2[0]),
        (0.5 * ind1[1]) + (0.5 * ind2[1]),
    ])

    v = [
        (3. / 2.) * ind1[0] - (0.5 * ind2[0]),
        (3. / 2.) * ind1[1] - (0.5 * ind2[1])
    ]

    w = [
        (-0.5 * ind1[0]) + ((3. / 2.) * ind2[0]),
        (-0.5 * ind1[1]) + ((3. / 2.) * ind2[1])
    ]

    evaluated = [fitnessFunction(z), fitnessFunction(v), fitnessFunction(w)]
    print(f'evaluated in crossing: {evaluated}')
    min_idx = evaluated.index(min(evaluated))

    if min_idx == 0:
        return v, w
    elif min_idx == 1:
        return z, w
    else:
        return z, v


def cxAverage(ind1, ind2):
    new_ind = [
        (ind1[0] + ind2[0]) / 2,
        (ind1[1] + ind2[1]) / 2
    ]

    return new_ind, min(fitnessFunction(ind1), fitnessFunction(ind2))


def cxBlendCrossAlpha(ind1, ind2, alpha):
    if alpha is None:
        alpha = np.random.uniform(0., 1.)

    min_x = min(ind1[0], ind2[0])
    min_y = min(ind1[1], ind2[1])

    dx = abs(ind1[0] - ind2[0])
    dy = abs(ind1[1] - ind2[1])

    x1_new = np.random.uniform(min_x - (alpha * dx), min_x + (alpha * dx))
    y1_new = np.random.uniform(min_y - (alpha * dy), min_y + (alpha * dy))
    new_ind1 = [x1_new, y1_new]

    x2_new = np.random.uniform(min_x - (alpha * dx), min_x + (alpha * dx))
    y2_new = np.random.uniform(min_y - (alpha * dy), min_y + (alpha * dy))
    new_ind2 = [x2_new, y2_new]

    return new_ind1, new_ind2


def cxBlendCrossAlphaBeta(ind1, ind2, alpha, beta):
    if alpha is None:
        alpha = np.random.uniform(0., 1.)

    if beta is None:
        beta = np.random.uniform(0., 1.)

    min_x = min(ind1[0], ind2[0])
    min_y = min(ind1[1], ind2[1])

    dx = abs(ind1[0] - ind2[0])
    dy = abs(ind1[1] - ind2[1])

    x1_new = np.random.uniform(min_x - (alpha * dx), min_x + (beta * dx))
    y1_new = np.random.uniform(min_y - (alpha * dy), min_y + (beta * dy))
    new_ind1 = [x1_new, y1_new]

    x2_new = np.random.uniform(min_x - (alpha * dx), min_x + (beta * dx))
    y2_new = np.random.uniform(min_y - (alpha * dy), min_y + (beta * dy))
    new_ind2 = [x2_new, y2_new]

    return new_ind1, new_ind2


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
toolbox.register("mate", cxBlendCrossAlpha)

# mutGaussian(mu = 5, sigma = 10), mutUniformInt(low =, up =, indpd=)
toolbox.register("mutate", tools.mutFlipBit)

sizePopulation = 2
probabilityMutation = 0.4
probabilityCrossover = 0.8
numberIteration = 10

pop = toolbox.population(n=sizePopulation)
fitness = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitness):
    ind.fitness.values = fit

g = 0
numberElitism = 0
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
            toolbox.mate(child1, child2, 0.2)

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
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    print("-- End of (successful) evolution --")

generate_mean_plot(mean_record)
generate_standard_deviation_plot(std_record)
generate_best_value_plot(min_record)
