import math
import numpy as np
import pickle
from intellino.core.neuron_cell import NeuronCells
from torchvision import datasets


# pickle data load
with open('./data_for_intellino/intellino_test_dataset.pkl', 'rb') as f:
    test_data = pickle.load(f)
# print(test_data[0])

with open('./data_for_intellino/intellino_train_dataset.pkl', 'rb') as f:
    train_data = pickle.load(f)
# print(len(train_data))


# intellino variable set
number_of_neuron_cells = 738
length_of_input_vector = 1024
resize_size = int(math.sqrt(length_of_input_vector))
neuron_cells = NeuronCells(number_of_neuron_cells=number_of_neuron_cells,
                           length_of_input_vector=length_of_input_vector,
                           measure="manhattan")


# train
for data, label in train_data:
    numpy_data = np.array(data[0])
    numpy_data = numpy_data.flatten()
    print(numpy_data)
    is_finish = neuron_cells.train(vector=numpy_data, target=label[0])
    if is_finish == True:
        break
