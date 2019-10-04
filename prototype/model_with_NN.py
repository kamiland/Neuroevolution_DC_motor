import numpy as np

max_weight = 0.2
max_bias = 0.0
input_count = 3  # 2 states of DC motor + setpoint/error
hidden_count = 10  # some arbitrary value
output_count = 1  # one control force (output)
max_control_constant = 200


class NeuralNetwork:
    def __init__(self, _input_count, _hidden_count, _output_count, randomize):

        self.weights_ih = np.zeros((_hidden_count, _input_count))
        self.weights_ho = np.zeros((_output_count, _hidden_count))

        # self.input = np.zeros(input_count)

        self.hidden = np.zeros(_hidden_count)
        self.output = np.zeros(_output_count)

        self.biases_ih = np.zeros(_hidden_count)
        self.biases_ho = np.zeros(_output_count)

        self.input_count = _input_count
        self.hidden_count = _hidden_count
        self.output_count = _output_count

        if randomize:
            self.randomization(max_weight, max_bias)  # this values are only for testing right now

    def think(self, _input):
        if len(_input) == self.input_count:
            # activate( weights_ih * input + biases_ih )
            for i in range(0, self.hidden_count):
                temp = 0
                for j in range(0, self.input_count):
                    temp += _input[j] * self.weights_ih[i][j]

                self.hidden[i] = temp + self.biases_ih[i]

            self.ReLU(self.hidden)

            # activate( weights_ho * (input + biases_ho) )
            for i in range(0, self.output_count):
                temp = 0
                for j in range(0, self.hidden_count):
                    temp += self.hidden[j] * self.weights_ho[i][j]

                self.output[i] = temp + self.biases_ho[i]

            self.tanh(self.output)

        return self.output

    def randomize_weights(self, weights_range):
        for i in range(0, len(self.weights_ih)):
            for j in range(0, len(self.weights_ih[0])):
                self.weights_ih[i][j] = ((np.random.sample() * 2) - 1) * weights_range

        for i in range(0, len(self.weights_ho)):
            for j in range(0, len(self.weights_ho[0])):
                self.weights_ho[i][j] = ((np.random.sample() * 2) - 1) * weights_range

    def randomize_biases(self, biases_range):
        for i in range(0, len(self.biases_ih)):
            self.biases_ih[i] = ((np.random.sample() * 2) - 1) * biases_range

        for i in range(0, len(self.biases_ho)):
            self.biases_ho[i] = ((np.random.sample() * 2) - 1) * biases_range

    def randomization(self, weights_range, biases_range):
        self.randomize_weights(weights_range)
        self.randomize_biases(biases_range)

    def ReLU(self, arr):
        for i in range(0, len(arr)):
            if arr[i] < 0:
                arr[i] = 0

    def tanh(self, arr):
        for i in range(0, len(arr)):
            arr[i] = np.tanh(arr[i])

    def logistic(self, arr):
        for i in range(0, len(arr)):
            arr[i] = 1.0 / (1 + np.exp(-arr[i]))


class Model:
    def __init__(self):
        self.fitness = 0.0
        self.brain = NeuralNetwork(input_count, hidden_count, output_count, True)
        self.RK4 = Solver()
        self.rotor_current_error = []
        self.rotor_current = []
        self.angular_velocity_error = []
        self.angular_velocity_error_int = 0
        self.angular_velocity_error_int_vector = []
        self.angular_velocity = []
        self.NN_output_vector = []
        self.setpoint = 100
        self.input_vector = np.zeros(3)
        self.setpoint_vector = []
        self.time = []
        self.control_constant = np.random.sample() * max_control_constant

    def set_brain(self, _brain: NeuralNetwork):
        self.brain = _brain

    def get_fitness(self):
        return self.fitness

    def set_control_constant(self, _control_constant):
        self.control_constant = _control_constant

    def set_setpoint(self, _setpoint):
        self.setpoint = _setpoint

    def calculate_fitness(self):
        self.fitness = 1.0 / (1.0 + self.angular_velocity_error_int)

    def brain_simulation(self, number_of_probes, time_step):
        # self.rotor_current_error = []
        # self.rotor_current = []
        # self.angular_velocity_error = []
        # self.angular_velocity = []
        # self.time = []
        self.RK4.x[0] = 0.0  # rotor current
        self.RK4.x[1] = 0.0  # angular velocity

        for i in range(0, number_of_probes):

            # if i == 0:
            #     self.set_setpoint(100)

            self.setpoint_vector.append(self.setpoint)

            self.input_vector[0] = self.RK4.x[0]
            self.input_vector[1] = self.RK4.x[1]
            local_error = np.abs(self.setpoint - self.RK4.x[1])
            self.input_vector[2] = local_error

            local_nn_output = float(self.brain.think(self.input_vector))
            u = local_nn_output * self.control_constant
            self.NN_output_vector.append(u)

            self.RK4.x = self.RK4.calculate_next_step(u, time_step)

            self.rotor_current.append(self.RK4.x[0])
            self.angular_velocity.append(self.RK4.x[1])

            self.angular_velocity_error.append(self.setpoint - self.RK4.x[1])
            self.angular_velocity_error_int += np.square(self.setpoint - self.RK4.x[1])
            self.angular_velocity_error_int_vector.append(self.angular_velocity_error_int)

            self.time.append(time_step * i)

        self.calculate_fitness()

    # def simulation(self, number_of_probes, time_step):
    #     self.rotor_current_error = []
    #     self.rotor_current = []
    #     self.angular_velocity_error = []
    #     self.angular_velocity = []
    #     self.time = []
    #     self.RK4.x[0] = 0.0  # rotor current
    #     self.RK4.x[1] = 0.0  # angular velocity
    #
    #     for i in range(0, number_of_probes):
    #         self.RK4.x = self.RK4.calculate_next_step(self.setpoint, time_step)
    #         self.rotor_current.append(self.RK4.x[0])
    #         self.angular_velocity.append(self.RK4.x[1])
    #         self.angular_velocity_error.append(self.setpoint - self.RK4.x[1])
    #         self.angular_velocity_error_int += np.abs(self.angular_velocity_error[i])
    #         self.angular_velocity_error_int_vector.append(self.angular_velocity_error_int)
    #         self.time.append(time_step * i)
    #
    #     self.calculate_fitness()


class Solver:

    def __init__(self):
        self.Ra = 0.4  # Ra - rezystancja uzwojenia wirnika
        self.La = 0.02  # La - indukcyjność własna wirnika
        self.Rf = 65  # Rf - rezystancja obwodu wzbudzenia
        self.Lf = 65  # Lf - indukcyjność własna obwodu wzbudzenia
        self.J = 0.11  # J - moment bezwłądności
        self.B = 0.0053  # B - współczynnik tłumienia
        self.p = 2  # p - pary biegunów
        self.Laf = 0.363  # Laf - indukcyjność wzajemna
        self.Ufn = 110

        # Wartości początkowe dla obiektu silnika
        self.rotorCurrent = 0.0
        self.angularVelocity = 0.0

        self.x = np.array([0.0, 0.0])
        self.x[0] = self.rotorCurrent
        self.x[1] = self.angularVelocity

        self.ifn = self.Ufn / self.Rf
        self.Gaf = self.p * self.Laf * self.ifn

    # U - napięcie zasilania
    def calculate_rotor_current(self, x1, x2, U):
        return -(self.Ra / self.La) * x1 - (self.Gaf / self.La) * x2 + (1 / self.La) * U

    def calculate_angular_velocity(self, x1, x2, Tl):
        return (self.Gaf / self.J) * x1 - (self.B / self.J) * x2 + (1 / self.J) * Tl

    # implementacja algorytmu Rungego-Kutty dla obiektu silnika DC
    # https://pl.wikipedia.org/wiki/Algorytm_Rungego-Kutty
    def calculate_next_step(self, U, h):
        k = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

        # Wyznaczanie prądu obwodu wirnika
        k[0][0] = h * self.calculate_rotor_current(self.x[0], self.x[1], U)
        k[0][1] = h * self.calculate_rotor_current(self.x[0] + k[0][0] / 2, self.x[1] + k[0][0] / 2, U)
        k[0][2] = h * self.calculate_rotor_current(self.x[0] + k[0][1] / 2, self.x[1] + k[0][1] / 2, U)
        k[0][3] = h * self.calculate_rotor_current(self.x[0] + k[0][2], self.x[1] + k[0][2], U)

        # Wyznaczanie prędkości kątowej
        k[1][0] = h * self.calculate_angular_velocity(self.x[0], self.x[1], 0)
        k[1][1] = h * self.calculate_angular_velocity(self.x[0] + k[1][0] / 2, self.x[1] + k[1][0] / 2, 0)
        k[1][2] = h * self.calculate_angular_velocity(self.x[0] + k[1][1] / 2, self.x[1] + k[1][1] / 2, 0)
        k[1][3] = h * self.calculate_angular_velocity(self.x[0] + k[1][2], self.x[1] + k[1][2], 0)

        for i in range(0, 2):
            self.x[i] = self.x[i] + (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6

        return self.x
