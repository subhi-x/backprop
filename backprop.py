# Feed Forward Network
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
  return 1/(1+np.exp(-x))
def sigmoid_der(x):
  return sigmoid(x)*(1-sigmoid(x))

inputs = np.array([[0.05],[0.1]]) # x1 and x2
real_outputs = np.array([[0.01],[0.99]]) # y1 and y2
real_outputs= real_outputs.reshape(2,1)
weights_hidden = np.array([[0.15,0.2],[0.25,0.3]])
hidden_bias = 0.35

output_weights = np.array([[0.4,0.45],[0.5,0.55]])
output_bias = 0.6

lr=0.5

for epoch in range(20000):
  h = np.dot(weights_hidden,inputs)+hidden_bias
  h_activate = sigmoid(h)

  o = np.dot(output_weights,h_activate)+output_bias
  o_activate = sigmoid(o)
  MSE= np.square(np.subtract(real_outputs,o_activate)).mean()
  # print(MSE) (This was to check if there were any changes actually occuring in MSE)

  d1=o_activate-real_outputs
  d2=sigmoid_der(o)
  d3=h_activate
  er_1=d1*d2
  d3=d3.reshape(1,2)
  opt_err=np.dot(er_1,d3)
  der_dh=output_weights
  d4=np.dot(der_dh.T,er_1)
  d5=sigmoid_der(h)
  d6=inputs
  er_2=d4*d5
  d6=d6.reshape(1,2)
  inp_err=np.dot(er_2,d6)

  weights_hidden=weights_hidden - (lr*inp_err)
  output_weights=output_weights - (lr*opt_err)
  
  h = np.dot(weights_hidden,inputs)+hidden_bias
h_activate = sigmoid(h)

o = np.dot(output_weights,h_activate)+output_bias
o_activate = sigmoid(o)

print(o_activate)
