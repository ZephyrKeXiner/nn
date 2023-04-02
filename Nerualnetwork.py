import numpy as np
import scipy.special


class NerualNetwork():
    def __init__(self,inlayer_nodes, outlayer_nodes, hiddenlayer_nodes, learning_rate):
        self.inlayer_nodes = inlayer_nodes
        self.outlayer_nodes = outlayer_nodes
        self.hiddenlayer_nodes = hiddenlayer_nodes
        self.lrt = learning_rate

        self.inputweight = np.random.normal(0.0,5.0,(self.hiddenlayer_nodes, self.inlayer_nodes))
        self.hiddenweight = np.random.normal(0.0,5.0,(self.outlayer_nodes, self.hiddenlayer_nodes))

        self.activity_func = lambda x:scipy.special.expit(x)
    
    def forward_spread(self,inputs):
        inputs = np.array(inputs,ndmin=2).T
        hiddenlayer_inputs = np.dot(self.inputweight, inputs)
        hiddenlayer_outputs = self.activity_func(hiddenlayer_inputs)
        outlayer_inputs = np.dot(self.hiddenweight, hiddenlayer_outputs)
        outlayer_outputs = self.activity_func(outlayer_inputs)

        return inputs,hiddenlayer_outputs,outlayer_outputs

    def error_comp(self,outlayer_outputs,correct):
        correct = np.array(correct,ndmin=2).T
        outlayer_error = correct - outlayer_outputs
        hiddenlayer_error = np.dot(self.hiddenweight.T, outlayer_error)
        return hiddenlayer_error, outlayer_error
    
    def gradintdecrease(self, inlayer, hiddenlayer_errors, output_errors, hiddenlayer_outputs, outlayer_outputs):

        self.hiddenweight += self.lrt * np.dot((output_errors * outlayer_outputs * (1.0 - outlayer_outputs)),
                                                np.transpose(hiddenlayer_outputs))
        self.inputweight += self.lrt * np.dot((hiddenlayer_errors * hiddenlayer_outputs * (1.0 - hiddenlayer_outputs)),
                                               np.transpose(inlayer))

    def show(self,inputs):
        inputs = np.array(inputs,ndmin=2).T

        hidden_inputs = np.dot(self.inputweight, inputs)
        hidden_outputs = self.activity_func(hidden_inputs)

        final_inputs = np.dot(self.hiddenweight, hidden_outputs)
        final_outputs = self.activity_func(final_inputs)

        return final_outputs
