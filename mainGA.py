import ga
import pprint

def main():
    popSize = 10
    pop = ga.GA.selectPop(pop, popSize)

    for x in range(popSize): 
        print ("pop", [x], pop[x])

if __name__ == "__main__":
    main()