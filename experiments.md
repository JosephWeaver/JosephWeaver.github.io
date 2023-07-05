
# Overview

## Terms 
- **input_dim** - the input dimension is determined by the size of data, thus can only change when the data is changed.
- **output** - the output layer of the neural network is a vector of reduced size (over the input) that contains the same discrimination ability as the original set.
- **input** - some image or vector belonging to a category.
- **weights** - (denoted w) a random variable used to increase/decrease multiplicatively the importance of the input to a node.
- **bias** - (denoted v) a random variable used to transition/center (add/subtract) the input node before it is passed to the activation function. 

## Hyperparameters
- **batch_size** - the number of input vectors processed in a single step.
- **max_epochs** - the maximum number of iteration through the entire training set.
- **learning_rate** - a scaler on the weight update
    - **lr_schedule** - (learning rate schedule) a schedule of adjustments to the learning based on some crtieria such as current epoch.
- **num_gpu** - the degree to which 
- **optimizer** - the choice of SGD, AdamW, et al. for solving the parameters.  Note: the can be mutliple optimizers, each for a different set of parameters.
    - **weight_decay** - ?
- **output_dim** - the overall compression rate.
- **nlayers** - the number of hidden layers in the network
    - **layer_size** - the number of nodes in a layer, each layer maybe a different size.
    - **act_funcs** - (activation functions) The function is called at the end of a layer to indicate, so there maybe many different activation function at each layer.
- **batch_norm** - indicates whether we use batch normalization
- **skip_steps** - indicates if layers are connected to deeper layers.  e.g.  a skip value of 2 means layer 1 is fully connected to layer 3, and 2 to 4.



# Investigation: Multiple GPU effect on Runtime.
## Experiment 1 – Reduce 128d/5id 2000 Hyperspheres (hps/00040)
Using 2000 generated hyperspheres with 128 dimensions, 5 intrinsic dimensions and a batch size of 50 we see the following run times
n	p	d	batch size	gpu	runtime
2000	128	5	50	1	2.2
2000	128	5	50	2	2.1
2000	128	5	50	4	2.5

Analysis: For the small hypersphere we see an increase in execution time as we parallelized over multiple GPUS.
 
## Experiment 2 – Reduce 1024d/50id 20000 Hypersphere (hps/00043)
Using 2000 generated hyperspheres with 128 dimensions, 5 intrinsic dimensions and a batch size of 50 we see the following run times
epoch	n	p	d	batch size	gpu	Epoch 
runtime	total 
runtime
50	20000	1024	50	500	1	7	5m42s
50	20000	1024	50	500	2	6.5	5m16s
50	20000	1024	50	500	4	5.8	4m42s

Analysis: For the larger hypersphere we see an decrease in execution time as we parallelized over multiple GPUS.  This decrease was not a 2x per 2x gpu.  Increasing the GPU count inversely proportionally decrease steps total steps = (n*epochs)/(batch size * num_gpu).
•	As n increases, epoch time increases.  But, by how much?
•	As batch size increases, we expect that training time should decrease up to a certain point, but loss and ap maybe effected.
Questions:
•	H

## Experiment 3 – hyperparameter search batch_size and GPU (hps/00045)
Using a sizes similar to imagenet dataset (n=1300000,p=1536,d=50) we can investigate the 1,2,4 gpu over batch sizes of 128,256,1024,2056.  Note: I have not run the hyperball at such a large size, my have issue.

Issues: code errored on prototype tensorboard.add_graph call. commented out.

# Investigation: Various Learning Rates
## Experiments: Starting at 1e-2 step 50 by .9 (bnnidr/00047)
Objective:  See how various learning rates affect loss and average precision.  
Using the StepLR(step_size=50, gamma=.9).

 

## Experiment: Variable Learning Rate
Objective:  model
