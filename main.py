import random, copy

class Perceptron:

    def __init__(self, samples, outputs, learning_rate=0.1, epochs=1000, threshold=-1):
        self.samples = samples  # all samples
        self.outputs = outputs  # respective outputs for each sample
        self.learning_rate = learning_rate  # learning rate (between 0 and 1)
        self.epochs = epochs  # number of epochs
        self.threshold = threshold  # threshold
        self.num_samples = len(samples)  # quantity of samples
        self.num_elements = len(samples[0])  # quantity of elements per sample
        self.weights = []  # weight vector

    # function to train the network
    def train(self):
        # add -1 to each of the samples
        for sample in self.samples:
            sample.insert(0, -1)

        # initialize the weight vector with random values
        for i in range(self.num_elements):
            self.weights.append(random.random())

        # insert the threshold in the weight vector
        self.weights.insert(0, self.threshold)

        # initialize epoch counter
        num_epochs = 0

        while True:
            error = False  # initially no error

            # for all training samples
            for i in range(self.num_samples):
                u = 0

                '''
                    perform the summation, the limit (self.num_elements + 1)
                    is because -1 was inserted for each sample
                '''
                for j in range(self.num_elements + 1):
                    u += self.weights[j] * self.samples[i][j]

                # get the network output using the activation function
                y = self.sign(u)

                # check if the network output is different from the desired output
                if y != self.outputs[i]:
                    # calculate the error: subtraction between desired output and network output
                    error_aux = self.outputs[i] - y

                    # adjust the weights for each sample element
                    for j in range(self.num_elements + 1):
                        self.weights[j] = self.weights[j] + self.learning_rate * error_aux * self.samples[i][j]

                    error = True  # error still exists

            # increment the number of epochs
            num_epochs += 1

            # stopping criteria is by number of epochs or if no error exists
            if num_epochs > self.epochs or not error:
                break

    # function used to test the network
    # receives a sample to be classified and the class names
    # uses the sign function, if it's -1 then it's class1, otherwise it's class2
    def test(self, sample, class1, class2):
        # insert -1
        sample.insert(0, -1)

        # uses the weight vector that was adjusted during training
        u = 0
        for i in range(self.num_elements + 1):
            u += self.weights[i] * sample[i]

        # calculate the network output
        y = self.sign(u)

        # check which class it belongs to
        if y == -1:
            print('The sample belongs to class %s' % class1)
        else:
            print('The sample belongs to class %s' % class2)

    # activation function: bipolar step (sign)
    def sign(self, u):
        return 1 if u >= 0 else -1


print('\nA or B?\n')

# samples: a total of 4 samples
samples = [[0.1, 0.4, 0.7], [0.3, 0.7, 0.2], 
           [0.6, 0.9, 0.8], [0.5, 0.7, 0.1]]

# desired outputs for each sample
outputs = [1, -1, -1, 1]

# set of test samples
tests = copy.deepcopy(samples)

# create a Perceptron network
network = Perceptron(samples=samples, outputs=outputs,
                     learning_rate=0.1, epochs=1000)

# train the network
network.train()

# testing the network
for test in tests:
    network.test(test, 'A', 'B')
