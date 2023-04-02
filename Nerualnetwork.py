import numpy as np
import scipy.special


class NerualNetwork():
    def __init__(self,input_nodes,output_nodes,hidden_nodes,learning_rate):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_nodes = hidden_nodes
        self.lrt = learning_rate

        self.inputweight = np.random.normal(0.0,5.0,(self.hidden_nodes,self.input_nodes))
        self.hiddenweight = np.random.normal(0.0,5.0,(self.output_nodes,self.hidden_nodes))

        self.activity_func = lambda x:scipy.special.expit(x)
    
    def forward_spread(self,inputs):
        inputs = np.array(inputs,ndmin=2).T
        hidden_inputs = np.dot(self.inputweight,inputs)
        hidden_outputs = self.activity_func(hidden_inputs)
        output_inputs = np.dot(self.hiddenweight,hidden_outputs)
        output_outputs = self.activity_func(output_inputs)

        print(output_outputs)