import numpy as np

# test
correct = 0

for data, label in test_data:
    numpy_data = np.array(data[0])
    numpy_data = numpy_data.flatten()
    # print(numpy_data)
    predict_label = neuron_cells.inference(vector=numpy_data)
    # print(f"label : {label[0]}, predict_label : {predict_label}")
    if predict_label == label:
        correct = correct + 1

test_accuracy = correct/len(test_data)*100
print(f'accuracy = {test_accuracy:.2f}%')
