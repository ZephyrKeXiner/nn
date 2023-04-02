import Nerualnetwork
import numpy as np
from PIL import Image

input_nodes = 784
hidden_nodes =1000
out_puts = 10
learning_rate = 0.3
correct = []
epoches = 12


nn = Nerualnetwork.NerualNetwork(input_nodes,hidden_nodes,out_puts,learning_rate)
for i in (epoches):
    inputs, hiddenlayer_outputs, outlayer_outputs = nn.forward_spread(input_list)
    hiddenlayer_error, outlayer_error = nn.error_comp(outlayer_outputs,correct)
    nn.gradintdecrease(inputs, 
                       hiddenlayer_error,
                       outlayer_error,
                       hiddenlayer_outputs,
                       outlayer_outputs)

final = nn.show()
