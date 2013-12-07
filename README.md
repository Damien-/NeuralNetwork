NeuralNetwork
=============

FF BP 

Neural Net

Implement the back propagation algorithm on page 734 of your textbook to train a neural net. 
Allow your neural net to have an arbitrary number of inputs, outputs, hidden layers, and number 
of nodes in each hidden layer. You can use the two files below to debug your neural net, but you
should only need a single hidden layer with a few hidden nodes for this sample data. 
You may have to experiment with the learning rate (alpha) to get a decent fit within the allowed time.
Select a random number between -0.1 and 0.1 as the starting weights for each connection in your neural net.
Your neural net should be fully connected and feed forward only.

Cross Validation

Use k-fold cross validation to train and evaluate the performance of your neural net using 
the below text file (7 inputs and 1 output). Partition your data randomly into k bins. 
To perform a single cross validation, select one of the bins to be the validation data
and the other k-1 bins are combined to be the training data. The performance of your neural 
net is measured by summing over the error for all k possible validation sets and averaging the result.

Architecture Optimization

Run cross validations for several different neural net architectures. That is, allow the number of 
hidden layers and the number of nodes in each hidden layer to vary. Too many hidden layers/hidden
nodes in your architecture will result in overfitting of the training data and increased validation
error. Report the architecture of the neural net that provided the lowest validation error overall.
You can find a good neural net architecture by trial and error. You need a good architecture but not
the absolute best for the next part of the assignment. To avoid extremely long running times, limit 
the number of hidden layers to 1 to 3, and the number of hidden nodes in each hidden layer to 1 through 5. 
You may also find it necessary to again experiment with the allowed time and the learning rate.

Neural Net Drivers

To use your back propagating neural net by itself (NeuralNetDriver), allow the user to specify an 
input file as a command line argument that has the following format:

data file name (in the same format as the data files given above)
number of hidden layers
number of nodes in hidden layer 1 (similar for additional hidden layers, assume at least one hidden layer)
learning rate
error tolerance (use to determine the allowed time)
Provide as output the original data (inputs and outputs) and the neural net's computed outputs (unscaled)
for each data point. Also, output the final error summed over all of the inputs.

To use your neural net to optimize the architecture (NeuralNetArchitectureDriver), allow the user to 
specify an input file as a command line argument that has the following format:

data file name (in the same format as the data files given above)
number of cross validation bins (k)
max number of hidden layers
max number of nodes in hidden layer 1 (similar for additional hidden layers, assume at least one hidden layer)
learning rate
error tolerance (use to determine the allowed time)

Provide the (averaged) k-fold cross validation error for the architecture specified in the input file.
