import pandas 
from  sklearn import linear_model 

df = pandas.read_csv("cars.csv")

def get_error(binary_features):
    # print("input",binary_features)
    # 7 is the start index for nomerical parameters. clean data to numeric only and ommit 7
    features=[index+7 for index,x in enumerate(binary_features, 0) if x==1]
    # print(df.columns.values)
    if (features==[]):
        # too big to br ignored in getting Min
        return 1000000
    X=df.iloc[: , features]

    #consider last column as y value
    y = df.iloc[: , len(df.columns)-1]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    y_errors=y

    
    # get errors for each data row
    for feature_index,feature in enumerate(features, 0):
        # print("index",feature_index)
        y_errors=y_errors-(df.iloc[: , feature].mul(regr.coef_[feature_index]))
    # print("y_errors",y_errors)
    # print("output",np.sqrt(sum(y_errors**2)))
    return np.sqrt(sum(y_errors**2))



#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
##############################Genetic Algorithm######################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################


import numpy as np

TARGET_PHRASE = '0000'       # target DNA
# 300
POP_SIZE = 300                    # population size
CROSS_RATE = 0.4                    # mating probability (DNA crossover)
MUTATION_RATE = 0.01                # mutation probability
# 1000 debug
N_GENERATIONS = 100


DNA_SIZE = len(TARGET_PHRASE)
# TARGET_ASCII = np.fromstring(TARGET_PHRASE, dtype=np.uint8)  # convert string to number
TARGET_ASCII=np.array([0,0,0,1,1,0,0,1,1,1,1,0,0,1,1,1])
print(TARGET_ASCII)
BOUND = [0, 1]


class GA(object):
    def __init__(self, DNA_size, DNA_bound, cross_rate, mutation_rate, pop_size):
        self.DNA_size = DNA_size
        DNA_bound[1] += 1
        self.DNA_bound = DNA_bound
        # print("self.DNA_bound",self.DNA_bound)
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        initial_pop= np.random.randint(*DNA_bound, size=(pop_size, DNA_size)).astype(np.int8)
        print("initial pop",initial_pop)
        self.pop = initial_pop  # int8 for convert to ASCII

    def translateDNA(self, DNA):                 # convert to readable string
        return DNA

    def get_fitness(self):                      # count how many character matches
        # match_count = (self.pop == TARGET_ASCII).sum(axis=1)
        # print('abc',(self.pop == TARGET_ASCII).sum(axis=1))
        list_of_errors=[get_error(dna) for dna in self.pop]
        result=np.array(list_of_errors)
        # print(type(self.pop))
        return result

    def select(self):
        fitness = self.get_fitness() + 1e-4     # add a small amount to avoid all zero fitness
        prob_matice=((fitness)/fitness.sum())
        prob_matice=((fitness.sum()-fitness)/fitness.sum())/(len(fitness)-1)
        # print("prob",prob_matice)
        # idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True)
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=prob_matice)
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)                        # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   # choose crossover points within string
            parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                child[point] = np.random.randint(*self.DNA_bound)  # choose a random ASCII index
        return child

    def evolve(self):
        pop = self.select()
        pop_copy = pop.copy()
        for parent in pop:  # for every pop as parent for next gen do both crossover and mutate
            # print("one of pop",parent)
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop

if __name__ == '__main__':
    ga = GA(DNA_size=DNA_SIZE, DNA_bound=BOUND, cross_rate=CROSS_RATE,
            mutation_rate=MUTATION_RATE, pop_size=POP_SIZE)

    for generation in range(N_GENERATIONS):
        fitness = ga.get_fitness()
        # print("pop",ga.pop)
        # print("fitness",fitness)
        best_DNA = ga.pop[np.argmin(fitness)]
        best_phrase = ga.translateDNA(best_DNA)
        print('Gen', generation, ': best features so far:', best_DNA)

        ga.evolve()