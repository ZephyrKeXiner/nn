import Nerualnetwork

input_nodes = 5
hidden_nodes =10
out_puts = 3
input_list =[4,2,5,2,4]

nn = Nerualnetwork.NerualNetwork(input_nodes,hidden_nodes,out_puts,0.3)

nn.forward_spread(input_list)
