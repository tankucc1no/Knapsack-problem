import pandas as pd
import random
# import math
# import numpy as np
import time


# import matplotlib.pyplot as plt


def data_clean(data):
    for i in range(0, len(data), 3):
        data[i + 1] = float(data[i + 1].split(':')[1])
        data[i + 2] = float(data[i + 2].split(':')[1])
    data_list = []
    i = 0
    while i < len(data):
        data_list.append(data[i:i + 3])
        i += 3
    for i in data_list:
        i.pop(0)
    # Generate DF with indexes 1-100
    df = pd.DataFrame(data_list, index=list(range(1, 101)))
    df.columns = ['weight', 'value']
    return df


def fitness(values, idx):
    f = values
    return f


# def fitness(values, idx):
#     f = np.cos((((values - 3000) / 1600) + 2) * math.pi / 2) * 3 + np.sin(((len(idx) - 40) / 30) / 2 * math.pi)
#     return f


def weight(chromosomes):
    w = 0
    for i in chromosomes:
        w += df.loc[i, 'weight']
    return w


def solution_generation(df, p=1):  # Generate a solution, by default generate 1
    if p == 1:
        weight, values = 0, 0
        chromosomes = []
        random_index = random.sample(range(1, 101), 100)
        for i in random_index:
            if df.loc[i, 'weight'] + weight <= MAX_WEIGHT:
                weight += df.loc[i, 'weight']
                values += df.loc[i, 'value']
                chromosomes.append(i)
            else:
                break
        return values, chromosomes
    else:
        values, chromosomes = [], []
        for i in range(p):
            v, c = solution_generation(df)
            values.append(v)
            chromosomes.append(c)
        return values, chromosomes


# def binarn_tournament_selection(v, c, t=1):
#     if t == 1:
#         # Randomly choose a chromosome from the population twice, get the values and chromosomes.
#         idx = random.sample(range(len(v)), 2)
#         values, chromosomes = [], []
#         for i in idx:
#             values.append(v[i])
#             chromosomes.append(c[i])
#         # Make the fittest of these two (breaking ties randomly) become the selected parent.
#         if fitness(values[0], chromosomes[0]) == fitness(values[1], chromosomes[1]):
#             parent_idx = random.sample(range(2), 1)[0]
#         elif fitness(values[0], chromosomes[0]) > fitness(values[1], chromosomes[1]):
#             parent_idx = 0
#         else:
#             parent_idx = 1
#         return values[parent_idx], chromosomes[parent_idx]
#     else:
#         values, chromosomes = [], []
#         for i in range(t):
#             value, chromosome = binarn_tournament_selection(v, c)
#             values.append(value)
#             chromosomes.append(chromosome)
#         return values, chromosomes


def binarn_tournament_selection(v, c, t=2):
    assert (t >= 2), 'T is required to be bigger than or equal to 2'
    idx = random.sample(range(len(v)), t)
    fitness_list = []
    for i in idx:
        fitness_list.append(fitness(v[i], c[i]))
    max_idx = fitness_list.index(max(fitness_list))
    idx_a = idx[max_idx]
    fitness_list[max_idx] = -9
    idx_b = idx[fitness_list.index(max(fitness_list))]
    return [v[idx_a], v[idx_b]], [c[idx_a], c[idx_b]]


def crossover(df, chromosomes):
    # Random node generation (choose a shortest chromosome length)
    node = random.sample(range(min(len(chromosomes[0]), len(chromosomes[1]))), 1)[0]
    a = chromosomes[0][:node + 1]
    b = chromosomes[1][:node + 1]
    c = chromosomes[0][node + 1:]
    d = chromosomes[1][node + 1:]
    weight_a = 0
    weight_b = 0
    for i in a:
        weight_a += df.loc[i, 'weight']
    for i in b:
        weight_b += df.loc[i, 'weight']

    # Check the indexes that meet the conditions and fill in the gaps
    for i in d:
        if i not in a and weight_a + df.loc[i, 'weight'] <= MAX_WEIGHT:
            a.append(i)
            weight_a += df.loc[i, 'weight']
    for i in c:
        if i not in a and weight_a + df.loc[i, 'weight'] <= MAX_WEIGHT:
            a.append(i)
            weight_a += df.loc[i, 'weight']
    # Exchange of index lists
    for i in c:
        if i not in b and weight_b + df.loc[i, 'weight'] <= MAX_WEIGHT:
            b.append(i)
            weight_b += df.loc[i, 'weight']
    for i in d:
        if i not in a and weight_b + df.loc[i, 'weight'] <= MAX_WEIGHT:
            b.append(i)
            weight_b += df.loc[i, 'weight']
    value_a = 0
    for i in a:
        value_a += df.loc[i, 'value']

    value_b = 0
    for i in b:
        value_b += df.loc[i, 'value']

    # return two new children a and b.
    return [value_a, value_b], [a, b]


# Swap_mutation, M is the integer parameter which determines how many times it is repeated on a solution.
def swap_mutation(chromosomes, m=1, rate=1):
    if random.uniform(0, 1) <= rate:
        if m == 1:
            for i in chromosomes:
                idx = random.sample(range(len(i)), 2)
                i[idx[0]], i[idx[1]] = i[idx[1]], i[idx[0]]
            return chromosomes
        else:
            for i in range(m):
                swap_mutation(chromosomes)
            return chromosomes
    else:
        return chromosomes


# M_gene_mutation, when m=1, it comes to single_gene_mutation
def single_gene_mutation(values, chromosomes, m=1, rate=1):
    if random.uniform(0, 1) <= rate:
        for i in range(len(values)):
            if m == 1:
                i_df = random.sample(range(1, 101), 100)
                i_c = random.randint(0, len(chromosomes[i]) - 1)
                w = weight(chromosomes[i]) - df.loc[chromosomes[i][i_c], 'weight']
                for j in i_df:
                    if j not in chromosomes[i]:
                        if w + df.loc[j, 'weight'] <= MAX_WEIGHT:
                            values[i] -= df.loc[chromosomes[i][i_c], 'value']
                            chromosomes[i][i_c] = j
                            values[i] += df.loc[j, 'value']
                            break
                return values, chromosomes
            else:
                for k in range(m):
                    values, chromosomes = single_gene_mutation(values, chromosomes)
                return values, chromosomes
    else:
        return values, chromosomes


def weakest_replacement(values, chromosomes, mcv, mcc):
    l = []
    for i in range(len(values)):
        l.append(fitness(values[i], chromosomes[i]))
    for i in range(len(mcv)):
        if fitness(mcv[i], mcc[i]) > fitness(values[l.index(min(l))], chromosomes[l.index(min(l))]):
            values[l.index(min(l))] = mcv[i]
            chromosomes[l.index(min(l))] = mcc[i]


def main(df, values, chromosomes):
    # Use the binary tournament selection twice (with replacement) to select two parents a and b.
    parent_values, parent_chromosomes = binarn_tournament_selection(values, chromosomes, T)

    # run crossover on these parents to give 2 children, c and d.
    children_values, children_chromosomes = crossover(df, parent_chromosomes)

    # mutation
    # mutation_children_values, mutation_children_chromosomes = children_values, swap_mutation(children_chromosomes, M)
    mutation_children_values, mutation_children_chromosomes = single_gene_mutation(children_values,
                                                                                   children_chromosomes, M, RATE)

    weakest_replacement(values, chromosomes, mutation_children_values, mutation_children_chromosomes)


# main function
with open("BankProblem.txt", "r") as f:  # Open file
    data = f.read()  # Read the file
data_list = data.split('\n')
MAX_WEIGHT = int(data_list.pop(0).split(':')[1])  # Reading maximum weight
df = data_clean(data_list)

P = 500
T = 25
M = 1
RATE = 1
first_time = [[P, T, M, RATE], ]
process_time = []
max_values = []
for i in range(5):  # One trail will loop 5 times. Average the result for analysis
    start = time.process_time()
    values, chromosomes = solution_generation(df, P)
    for j in range(10000):
        main(df, values, chromosomes)
    end = time.process_time()
    process_time.append(end - start)
    max_values.append(max(values))
    # print(max_values)
first_time.append(process_time)
first_time.append(max_values)
print(first_time)
