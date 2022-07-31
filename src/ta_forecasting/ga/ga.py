import numpy as np


class GeneticAlgorithm:

    def __init__(self, dna_size, pop_size, cross_rate, mutation_rate):
        self.dna_size = dna_size
        self.pop_size = pop_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate

    @staticmethod
    def onemax(x):
        return -sum(x)

    def selection(self, populations, scores, k=3):
        selection_ix = np.random.randint(self.dna_size)
        for ix in np.random.randint(0, self.dna_size, k - 1):
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return populations[selection_ix]

    def crossover(self, parent1, parent2):
        child1, child2 = parent1.copy(), parent2.copy()
        if np.random.rand() < self.cross_rate:
            pt = np.random.randint(1, self.dna_size - 2)
            child1 = parent1[:pt] + parent2[pt:]
            child2 = parent2[:pt] + parent1[pt:]
        return [child1, child2]

    def mutation(self, bitstring):
        for i in range(self.dna_size):
            if np.random.rand() < self.mutation_rate:
                bitstring[i] = 1 - bitstring[i]

    def genetic_algorithm(self, objective, n_iter):
        populations = [np.random.randint(0, 2, self.dna_size).tolist() for _ in range(self.pop_size)]
        best_pop, best_score = populations[0], objective(populations[0])
        for generation_number in range(n_iter):
            scores = [objective(p) for p in populations]
            for i in range(self.pop_size):
                if scores[i] < best_score:
                    best_pop, best_score = populations[i], scores[i]
                    print(">%d, new best f(%s) = %.3f" % (generation_number, populations[i], scores[i]))
            selected = [self.selection(populations, scores) for _ in range(self.pop_size)]
            children = []
            for i in range(0, self.pop_size, 2):
                parent1, parent2 = selected[i], selected[i + 1]
                for child in self.crossover(parent1, parent2):
                    self.mutation(child)
                    children.append(child)
            populations = children
        return [best_pop, best_score]


if __name__ == '__main__':
    n_iter = 100
    n_bits = 20
    n_pop = 100
    r_cross = 0.9
    r_mut = 1.0 / float(n_bits)

    best, score = GeneticAlgorithm(n_bits, n_pop, r_cross, r_mut).genetic_algorithm(
        GeneticAlgorithm.onemax, n_iter,
    )

    print('Done!')
    print('f(%s) = %f' % (best, score))
