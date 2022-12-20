import torch
import pygad.torchga
import pygad
import sys

from walking import sim

from test_pygad_my_model import control_model, fitness_func, callback_generation

if len(sys.argv) > 1:
    saved_population = pygad.load(filename=sys.argv[1])
else:
    saved_population = pygad.load(filename="../species/jumping_boy.ga")

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = saved_population.best_solution()

sim(solution, solution_idx, control_model, True)