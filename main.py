import Nerualnetwork
import numpy as np

input_nodes = 784
hidden_nodes =500
output_nodes = 10
learning_rate = 0.1
epoches = 15

data_file = open("training_set/mnist_train_small.csv","r")
data_list = data_file.readlines()
data_file.close()

nn = Nerualnetwork.NerualNetwork(input_nodes,output_nodes,hidden_nodes,learning_rate)
inlayer_weight = np.zeros((hidden_nodes,input_nodes))
hiddenlayer_weight = np.zeros((output_nodes,hidden_nodes))

for i in range(epoches):
    for record in data_list:
        all_values = record.split(",")
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99    
    
    
        inputs, hiddenlayer_outputs, outlayer_outputs = nn.forward_spread(inputs)
        hiddenlayer_error, outlayer_error = nn.error_comp(outlayer_outputs,targets)
        nn.gradintdecrease(inputs, 
                           hiddenlayer_error,
                           outlayer_error,
                           hiddenlayer_outputs,
                           outlayer_outputs)
    
    inlayer_weight = nn.inputweight
    hiddenlayer_weight = nn.hiddenweight

nn.hiddenweight = hiddenlayer_weight
nn.inputweight = inlayer_weight

np.save('hiddenlayer_weight.npy', hiddenlayer_weight)
np.save('inlayer_weight.npy', inlayer_weight)

test_data_file = open("training_set/mnist_test.csv","r")
test_data_list = test_data_file.readlines()
test_data_file.close()

for record in test_data_list:
    all_values = record.split(",")
    inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
    nn.show(inputs)
