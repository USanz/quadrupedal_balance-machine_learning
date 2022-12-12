import torch
import pygad.torchga
import pygad

from walking import sim


class ControlNet(torch.nn.Module):
     def __init__(self):
         super(ControlNet, self).__init__()
         #self.flatten = nn.Flatten()
         self.linear_relu_stack = torch.nn.Sequential(
             torch.nn.Linear(19, 10),
             torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
             torch.nn.ReLU(),
             torch.nn.Linear(10, 10),
             torch.nn.ReLU(),
             torch.nn.Linear(10,12)
         )
 
     def forward(self, x):
         #x = self.flatten(x)
         logits = self.linear_relu_stack(x)
         return logits

def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, control_model

    return sim(solution, sol_idx, control_model, False)

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

# Create the PyTorch model
# this var should be imported when loading the saved model
control_model = ControlNet()


if __name__ == "__main__":
    
    # print(model)

    # Create an instance of the pygad.torchga.TorchGA class to build the initial population.
    torch_ga = pygad.torchga.TorchGA(model=control_model,
                            num_solutions=10)

    # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
    num_generations = 1 # Number of generations.
    num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
    initial_population = torch_ga.population_weights # Initial population of network weights

    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        initial_population=initial_population,
                        fitness_func=fitness_func,
                        on_generation=callback_generation)

    ga_instance.run()

    # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
    ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))


    #sim(solution, solution_idx, control_model, True)
    print("Solution is ", solution)

    ga_instance.save(filename="./save_ga_instance.ga")

    # # Make predictions based on the best solution.
    # predictions = pygad.torchga.predict(model=model,
    #                                     solution=solution,
    #                                     data=data_inputs)
    # print("Predictions : \n", predictions.detach().numpy())

    # abs_error = loss_function(predictions, data_outputs)
    # print("Absolute Error : ", abs_error.detach().numpy())