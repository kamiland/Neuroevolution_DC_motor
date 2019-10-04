import numpy as np
import math as m

from model_with_NN import max_weight
from model_with_NN import max_bias
from model_with_NN import max_control_constant

from model_with_NN import input_count
from model_with_NN import hidden_count
from model_with_NN import output_count

from model_with_NN import NeuralNetwork
from model_with_NN import Model


class GenethicAlgorithm:
    def __init__(self):
        self.population_count = 70
        self.DC_generation = [Model() for ii in range(self.population_count)]
        self.DC_next_generation = [Model() for ii in range(self.population_count)]

        self.DC_best = Model()

        self.best_fit = 0

    def make_simulation(self):
        for i in range(0, self.population_count):
            self.DC_generation[i].brain_simulation(100, 0.01)

    def make_next_generation(self):
        self.make_simulation()
        msx_fit = self.DC_generation[0].fitness
        for n in range(0, self.population_count):
            print("fitness int of", n, "object:",  self.DC_generation[n].fitness)
            if self.DC_generation[n].fitness > msx_fit:  # find smallest fit
                msx_fit = self.DC_generation[n].fitness

        print("max:", msx_fit)

        non_zero_count = self.normalize_fitness()

        for i in range(0, self.population_count):
            if non_zero_count > 4:      # some arbitrary value
                random_value = np.random.sample()
                if random_value >= 0.5:
                    self.pick_tweak(i)       # 50% chance of tweaking
                elif random_value >= 0.1:
                    self.pick_and_cross(i)   # 40% chance of crossing
                else:
                    self.mutatant(i)         # 10% chance of new random object

            elif non_zero_count > 0:    # when max 4 models have non zero fitness value
                random_value = np.random.sample()
                if random_value >= 0.7:
                    self.pick_tweak(i)   # 70% chance of tweaking
                else:
                    self.mutatant(i)     # 30% chance of new random object

            else:   # when all models have 0 fitness value
                self.mutatant(i)

        for i in range(0, self.population_count):
            self.DC_generation[i] = self.DC_next_generation[i]

    def pick_tweak(self, i):
        parent = Model()

        if np.random.sample() > 0.01:
            picked = False
            while True:
                x = m.floor((np.random.sample() * self.population_count))   # np.random.sample() will never
                if np.random.sample() <= self.DC_generation[x].fitness:     # return value = 1, so we will never
                    parent = self.DC_generation[x]                           # refer to models[population_count]
                    picked = True
                if picked:
                    break
        else:
            parent = self.DC_best   # 1% chance of picking "best" to tweak

        self.DC_next_generation[i] = self.tweak(parent, max_weight * 0.1, max_bias * 0.1, max_control_constant * 0.1)

    def pick_and_cross(self, i):
        parent_a = Model()
        parent_b = Model()
        picked = False

        while True:
            a = m.floor((np.random.sample() * self.population_count))
            if np.random.sample() <= self.DC_generation[a].fitness:
                parent_a = self.DC_generation[a]
                picked = True
            if picked:
                break
                
        if np.random.sample() > 0.01:
            picked = False
            while True:
                b = m.floor((np.random.sample() * self.population_count))
                if a != b and (np.random.sample() <= self.DC_generation[b].fitness):
                    parent_b = self.DC_generation[b]
                    picked = True
                if picked:
                    break
        else:
            parent_b = self.DC_best     # 1% chance of picking "best" as a pair to cross with

        self.DC_next_generation[i] = self.cross(parent_a, parent_b)

    def mutatant(self, i):
        if np.random.sample() > 0.01:
            mutant_brain = NeuralNetwork(input_count, hidden_count, output_count, True)
            self.DC_next_generation[i].set_brain(mutant_brain)
        else:
            self.DC_next_generation[i].set_brain(self.DC_best.brain)   # 1% chance of picking "best" as a mutant
            self.DC_next_generation[i].set_control_constant(self.DC_best.control_constant)

    @staticmethod
    def tweak(parent: Model, weight_tweak_range, bias_tweak_range, control_constant_tweak_range):
        # making a new brain (with no randomization) for a child
        tweak_brain = NeuralNetwork(input_count, hidden_count, output_count, False)

        # tweaking weights
        for i in range(0, len(tweak_brain.weights_ih)):
            for j in range(0, len(tweak_brain.weights_ih[0])):
                tweak_brain.weights_ih[i][j] = parent.brain.weights_ih[i][j] + np.random.normal()/3 * weight_tweak_range

        for i in range(0, len(tweak_brain.weights_ho)):
            for j in range(0, len(tweak_brain.weights_ho[0])):
                tweak_brain.weights_ho[i][j] = parent.brain.weights_ho[i][j] + np.random.normal()/3 * weight_tweak_range

        # tweaking biases
        for i in range(0, len(tweak_brain.biases_ih)):
            tweak_brain.biases_ih[i] = parent.brain.biases_ih[i] + np.random.normal()/3 * bias_tweak_range

        for i in range(0, len(tweak_brain.biases_ho)):
            tweak_brain.biases_ho[i] = parent.brain.biases_ho[i] + np.random.normal()/3 * bias_tweak_range

        tweak_control_constant = parent.control_constant + np.random.normal()/3 * control_constant_tweak_range

        child = Model()
        child.set_brain(tweak_brain)
        child.set_control_constant(tweak_control_constant)

        return child

    @staticmethod
    def cross(parent_a: Model, parent_b: Model):
        # making a new brain (with no randomization) for a child
        cross_brain = NeuralNetwork(input_count, hidden_count, output_count, False)

        # crossing weights
        for i in range(0, len(cross_brain.weights_ih)):
            for j in range(0, len(cross_brain.weights_ih[0])):
                random_value = np.random.sample()
                cross_brain.weights_ih[i][j] \
                    = random_value * parent_a.brain.weights_ih[i][j] + (1 - random_value) * parent_b.brain.weights_ih[i][j]

        for i in range(0, len(cross_brain.weights_ho)):
            for j in range(0, len(cross_brain.weights_ho[0])):
                random_value = np.random.sample()
                cross_brain.weights_ho[i][j] \
                    = random_value * parent_a.brain.weights_ho[i][j] + (1 - random_value) * parent_b.brain.weights_ho[i][j]

        # crossing biases */
        for i in range(0, len(cross_brain.biases_ih)):
            random_value = np.random.sample()
            cross_brain.biases_ih[i] \
                = random_value * parent_a.brain.biases_ih[i] + (1 - random_value) * parent_b.brain.biases_ih[i]

        for i in range(0, len(cross_brain.biases_ho)):
            random_value = np.random.sample()
            cross_brain.biases_ho[i] \
                = random_value * parent_a.brain.biases_ho[i] + (1 - random_value) * parent_b.brain.biases_ho[i]

        random_value = np.random.sample()
        cross_control_constant = random_value * parent_a.control_constant + (1 - random_value) * parent_b.control_constant

        child = Model()
        child.set_brain(cross_brain)
        child.set_control_constant(cross_control_constant)

        return child

    def normalize_fitness(self):
        non_zero = 0
        max_fit = 0.0
        for i in range(0, self.population_count):       # search generation and choose best model if exists
            if self.DC_generation[i].fitness > max_fit:
                max_fit = self.DC_generation[i].fitness
                if max_fit > self.best_fit:
                    self.best_fit = max_fit
                    self.DC_best = self.DC_generation[i]

        min_fit = max_fit

        for i in range(0, self.population_count):
            if self.DC_generation[i].fitness < min_fit:  # find smallest fit
                min_fit = self.DC_generation[i].fitness
            else:
                min_fit = min_fit

        if max_fit != 0 and (max_fit - min_fit) != 0:
            for i in range(0, self.population_count):
                self.DC_generation[i].fitness -= min_fit
                self.DC_generation[i].fitness = (self.DC_generation[i].fitness / (max_fit - min_fit))
                if self.DC_generation[i].fitness > 0:
                    non_zero = non_zero + 1

        return non_zero


def plot(*items):
    win = CurveDialog(edit=True, toolbar=True, wintitle="")
    plot = win.get_plot()
    for item in items:
        plot.add_item(item)
    win.show()
    win.exec_()


GA = GenethicAlgorithm()
if __name__ == '__main__':

    from guiqwt.plot import CurveDialog
    from guiqwt.builder import make
    import guidata

    _app = guidata.qapplication()

    # ==================================================================================================================
    for k in range(0, 25):
        print("\ngeneration:", k, ":\n")
        GA.make_next_generation()

    print(GA.DC_best.brain.weights_ih, "\n", GA.DC_best.brain.biases_ih, "\n")
    print(GA.DC_best.brain.weights_ho, "\n", GA.DC_best.brain.biases_ho, "\n")

    best_brain = GA.DC_best.brain
    test_model = Model()

    test_model.brain = GA.DC_best.brain
    test_model.control_constant = GA.DC_best.control_constant

    test_model.brain_simulation(150, 0.01)

    plot(make.curve(test_model.time, test_model.angular_velocity, color="b", title="angular_velocity"),
         make.curve(test_model.time, test_model.rotor_current, color="g", title="rotor_current"),
         make.curve(test_model.time, test_model.NN_output_vector, color="r", title="NN output"),
         make.curve(test_model.time, np.abs(test_model.angular_velocity_error), color="m", title="error"),
         make.curve(test_model.time, test_model.setpoint_vector, color="black", title="setpoint"),
         make.legend("BR"))
