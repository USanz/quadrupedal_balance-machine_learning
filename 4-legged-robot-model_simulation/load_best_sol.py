import torch
import pygad.torchga
import pygad

from walking import sim

from test_pygad_my_model import control_model, fitness_func, callback_generation


saved_population = pygad.load(filename="./save_ga_instance.ga")

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = saved_population.best_solution()

sim(solution, solution_idx, control_model, True)