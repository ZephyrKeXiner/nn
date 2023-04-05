import Nerualnetwork
import numpy as np

input_nodes = 784
hidden_nodes =500
output_nodes = 10
learning_rate = 0.3
nn = Nerualnetwork.NerualNetwork(input_nodes,output_nodes,hidden_nodes,learning_rate)

nn.hiddenweight = np.load('hiddenlayer_weight.npy')
nn.inputweight = np.load('inlayer_weight.npy')

test_data_file = open("training_set/test_1.csv","r")
test_data_list = test_data_file.readlines()
test_data_file.close()

for record in test_data_list:
    all_values = record.split(",")
    inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
    nn.show(inputs)
