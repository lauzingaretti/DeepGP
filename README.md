<h1 align="center">
  <br>
<img src="https://github.com/lauzingaretti/DeepGP/blob/master/Logo.png" width="500"></a>
  <br>
</h1>

<h3 align="center">Harnessing deep learning for complex traits Genomic Prediction (GP) in plant breeding programs </h3>

<p align="center">


<p align="center">
Laura Zingaretti (m.lau.zingaretti@gmail.com) and Miguel Pérez-Enciso (miguel.perez@uab.es)
</p>
<hr>
<p align="center">
A python based tool to implement Genomic Prediction Experiments using Deep Learning

</p>

### The Software

The DeepGP package implements Multilayer Perceptron Networks (MLP),
Convolutional Neural Network (CNN),
Ridge Regression and Lasso Regression to Genomic Prediction purposes.
Our package takes advantage of Autonomio Talos [Computer software] (2019) ( http://github.com/autonomio/talos) functionalities to
optimize the networks  parameters. It also implements uni and
multi-trait experiments with a set of normalization options.
We believe it can be a useful tool for people who want to
implement Deep Learning models for predicting complex traits in an
easy way.
For a more 'didactic' site on DL, please check: https://github.com/miguelperezenciso/DLpipeline
Citation: [Pérez-Enciso M, Zingaretti LM. 2019. A Guide on Deep Learning for Complex Trait Genomic Prediction. Genes, 10, 553.](https://www.mdpi.com/2073-4425/10/7/553)

### Introduction

Deep Learning (DL) techniques comprise a heterogeneous collection of Machine Learning
algorithms which have excelled at prediction tasks.
All DL algorithms employ multiple neuron layers and numerous
architectures have been proposed.
DL is relatively easy to implement (https://keras.io/why-use-keras/) but
is not 'plug and play' and the DL performance highly depends of the
hyperparameter combination.

Genomic Selection is the breeding strategy
consisting in predicting complex traits using
genomic-wide genetic markers and it is standard in many animal and
plant breeding schemes. Its powerful predictive ability makes
DL a suitable tool for prediction problems. Here, we deliver a tool
 to predicting complex traits
from whole genome SNP information in a easy way.
This is thanks to Keras API (https://keras.io/) and TensorFlow (https://www.tensorflow.org/), which allow all intricacies to be encapsulated through very simple statements. TensorFlow is a machine-learning library developed by Google.
We also take advantage of the scikit learn (https://scikit-learn.org/stable/)
to surrender classical penalized models for prediction purposes.

The novelty of our tool lies in the fact that it allows us to tune
the most important  hyperparameters in CNN and MLP architectures.

 ### GP main features

  - It allows tuning the MLP and CNN architectues for GP purposes
  - Our package uses a random search for hyperparameter optimization.
  - It is easy to implement
  - It delivers all the parameters to implemented models
   allowing you to explore the hyperparameters influence

### What is a hyperparameter in DL jargon?

  Hyperparameters are variables that need to set
  in order to apply a DL algorithm to a dataset. Even in a simple neural network,
  numerous hyperparameters need to be configured. The optimal combination of
  hyperparameters depends on the specific problem and a set of suitable hyperparameters
  are required each time. This makes our tool a good solution for applying DL to
  Genomic Prediction problems.

  ### Deep Learning hyperparameters

   |Term|Definition|
   |:----:|----------|
   |**Activation functions**|The mathematical function f that produces neuron’s output f(w’x + b), where w is a weights vector, x is an input vector, and b is bias, a scalar. Both w and b are to be estimated for all neurons.|
   |**Batch**|In **Stochastic Gradient Descent** algorithms, each of the sample partitions within a given **epoch**.|
   |**Dropout**|Dropout means that a given percentage of neurons output is set to zero. The percentage is kept constant, but the specific neurons are randomly sampled in every iteration. The goal of dropout is to avoid overfitting.|
   |**Kernel = Filter = Tensor**|In DL terminology, the kernel is a multidimensional array of weights.|
   |**Learning rate**|Specify the speed of gradient update.|
   |**Loss**|Loss function measures how differences between observed and predicted target variables are quantified.|
   |**Neural layer**|‘Neurons’ are arranged in layers, i.e., groups of neurons that take the output of previous group of neurons as input |
   |**Neuron**|The basic unit of a DL algorithm. A ‘neuron’ takes as input a list of variable values (x) multiplied by ‘weights’ (w) and, as output, produces a non-linear transformation f(w’x + b) where f is the activation function and b is the bias. Both w and b need to be estimated for each neuron such that the loss is minimized across the whole set of neurons.|
   |**Optimizer**|Algorithm to find weights (w and b) that minimize the loss function. Most DL optimizers are based on **Stochastic Gradient Descent** (SGD).|
   |**Pooling**|A pooling function substitutes the output of a network at a certain location with a summary statistic of the neighboring outputs. This is one of the crucial steps on the CNN architecture. The most common pooling operations are maximum, mean, median.|
   |**Weight regularization**|An excess of parameters (weights, w) may produce the phenomenon called ‘overfitting’, which means that the model adjusts to the observed data very well, but prediction of new unobserved data is very poor. To avoid this, weights are estimated subject to constraints, a strategy called ‘penalization’ or ‘regularization’. The two most frequent regularizations are the L1 and L2 norms, which set restrictions on the sum of absolute values of w (L1) or of the square values (L2)|

<hr>

### List of input parameters

The next two tables include the whole list of DeepGenomic.py input parameters to be parser using argparse.

|Input parameter | Description|
|:----:|----------|
|`Xtr`| Required: **True**.<br/> The training prediction matrix. It should be a .csv file with samples in rows and SNPs in column coded as 0, 1, to ploidy level. Data may contains row and colnames. <br/> **Usage:** `--Xtr /path/to/XtrainData.csv`  |
|`ytr`|Required: **True**. <br/>  The training y matrix containing traits to be predicted. It should be a .csv file with samples in rows and traits in columns. Data may contains row and colnames and rownames should match with Xtr. The matrix should contains at least one column which means you want evaluating a single trait. It allows both continous and categorical traits. <br/> **Usage:** `--ytr /path/to/ytrainData.csv` |
|`Xval`| Required: **False**.<br/> The validation set prediction matrix. It should be a .csv file with samples in rows and SNPs in column coded as 0, 1, to ploidy level. Data may contains row and colnames. This matrix have to have the same SPNs set than Xtr. If you don't give a validation set, the software internally split the data in train and validation set. <br/> **Usage:** `--Xval /path/to/XvalData.csv` |
|`yval`|Required: **False**.<br/> The validation y matrix containing traits to be predicted. It should be a .csv file with samples in rows and traits in columns. Data may contains row and colnames and rownames should match with Xval. The matrix should contains the same traits than ytr.<br/> **Usage:** ` --yval /path/to/yvalData.csv` |
|`cat`| Required: **False**.<br/> This argument facilitates the normalization of the training dataset by a categorical variable, i.e. if you have a breeding program and you need to categorize by trial. To use it, ytr have to contains an extra categorical column indicating observations belonging to each trial.<br/> **Usage:** `--cat Trial_Cat`, where Trial_Cat is the name of the categorical variable.|
|`scale`|Default: **False**.<br/> It is a Boolean argument indicating if the y matrix should be scaled. It only works if you write `--scale` argument when you run DeepGenomic.py.|
|`dummy`|Default: **False**.<br/> It is a Boolean argument indicating if the SNP matrix should be converted into OneHot enconding.<br/> **Usage:**  `--dummy` |
|`categorical`|Default: **False**.<br/> It is a Boolean argument indicating if traits in y matrix are categorical. <br/> **Usage:**  `--categorical` .|
|`trait`| Required: **False**.<br/> This argument indicates which trait you want evaluating. Use in case your matrix have more than 1 trait and you don't want evaluating all of them. <br/> **Usage:** `--trait name_of_the_trait` when you run DeepGenomic.py.|
|`output`| Required: **True**.<br/> This parameter indicates the folder which will contain the outputs (table with evaluated models, weights for the best models and a plot with the predictive performance of the model). <br/> **Usage:** `--output path/to/output/folder/`.|
|`Method`| Required: **True**.<br/> A mutually exclusive group indicating which model you want to run. <br/> **Usage:** `--mlp` for Multilayer Perceptron Networks. `--cnn` for Convolutional Neural Networks `--lasso` for classical lasso modeling and `--ridge` for ridge regression approach.|

The next table includes the parameters required if you want to evaluate a MLP (`--mlp` argument) or CNN (`--cnn` argument) neural networks, i.e. it is the process of hyperparameter tuning using the talos tool. All these parameters wouldn't work if you choose `--ridge` or `--lasso` as Method.  

|Input parameter | Description|
|:----:|----------|
|`lr`| Required: **False**.<br/> A list of learning rates to be considered in the process of Neural Network (NN) hyperparameters tuning. <br/> **Usage:** `--lr 0.001 0.025 0.1` or just lr: `-lr 0.0001` if you don't want to evaluate different learning rates. **Default**: `--lr 0.0025`.|
|`dr1`| Required: **False**.<br/> A list of dropouts in the first layer to be considered in the process of Neural Network (NN) hyperparameters tuning. <br/> **Usage:** `--dr1 0.001 0 0.1` or just `-- dr1 0`. **Default**: `--dr1 0`.|
|`dr2`| Required: **False**.<br/> A list of dropouts in the hidden layers to be considered in the process of Neural Network (NN) hyperparameters tuning. <br/> **Usage:** `--dr2 0.001 0 0.1` or just `-- dr2 0`. **Default**: `--dr2 0`.|
|`reg1`| Required: **False**.<br/> A list of L2 weight regularization to be used in the first layer in the process of Neural Network (NN) hyperparameters tuning. <br/> **Usage:** `--reg1 0.001 0 0.1` or just `-- reg1 0`. **Default**: `--reg1 0`.|
|`reg2`| Required: **False**.<br/> A list of L2 weight regularization to be used in the hidden layers in the process of Neural Network (NN) hyperparameters tuning. <br/> **Usage:** `--reg2 0.001 0 0.1` or just `-- reg2 0`. **Default**: `--reg2 0`.|
|`reg3`| Required: **False**.<br/> A list of L2 weight regularization to be used in the last layer in the process of Neural Network (NN) hyperparameters tuning. <br/> **Usage:** `--reg3 0.001 0 0.1` or just `-- reg3 0`. **Default**: `--reg3 0`.|
|`hn`| Required: **False**.<br/> A list of integers with the number of hidden neurons  to be evaluated in the process of Neural Network (NN) hyperparameters tuning. <br/> **Usage:** `--hn 8 16 32` or just `-- hn 64`. **Default**: `-hn 8`.|
|`hl`| Required: **False**.<br/> A list of integers with the number of hidden layers in the fully connected layers of both CNN or MLP networks. <br/> **Usage:** `--hl 1 5 7` or just `-- hl 3`. **Default**: `--hl 3`.|
|`N_neurons_FL`| Required: **False**.<br/> A list of integer containing the number of neurons considered in the first layer of the mlp architecture. <br/> **Usage:** `--N_neurons_FL  1 4 10` or just `--N_neurons_FL 32`. **Default**: `--N_neurons_FL 8`. It only works when `--mlp` is activated.|
|`epochs`| Required: **False**.<br/> A list of integers containing the number of epochs to be evaluated. <br/> **Usage:** `--epochs 50 100 150` or just `-- epochs 1000`. **Default**: `--epochs 50`.|
|`optimizer`| Required: **False**.<br/> A list of strings containing the optimizer function to be considered. You can choice between Adam, Nadam and sgd. <br/> **Usage:** `--optimizer Adam sgd` or just `-- optimizer Adam`. **Default**: `--optimizer Adam sgd Nadam`.|
|`bs`| Required: **False**.<br/> A list of integers containing the batch size to be considered. <br/> **Usage:** `--bs 16 32` or just `-- bs 32`. **Default**: `--bs 16`.|
|`act1`| Required: **False**.<br/> A list of str with the activation function to be considered. For a fully connected network, i.e. mlp it is the activation to be used in the fully connected layers and for a cnn it should be the activation for the Convolutional layer. You can chose one or more from the following list: relu, elu, linear, softmax, tanh, softplus. <br/> **Usage:** `--act1 tanh linear` or just `--act1 tanh`. **Default**: `--act1 relu, elu, linear, softmax, tanh, softplus`.|
|`act2`| Required: **False**.<br/> A list of str with the activation function to be considered in the fully connected layers of a CNN network.You can chose one or more from the following list: relu, elu, linear, softmax, tanh, softplus. <br/> **Usage:** `--act2 tanh linear` or just `--act2 tanh`. **Default**: `--act2 relu, elu, linear, softmax, tanh, softplus`. It only works when `--cnn` is activated.|
|`nconv`| Required: **False**.<br/> A list of integer with the number of convolutional layers to be considered. <br/> **Usage:** `--nconv 1 5` or just `--nconv 1`. **Default**: `--nconv 1`. It only works when `--cnn` is activated.|
|`ps`| Required: **False**.<br/> A list of integer containing pool size for convolutions. <br/> **Usage:** `--ps  1 5` or just `--ps 1`. **Default**: `--ps 1`. It only works when `--cnn` is activated.|
|`ks`| Required: **False**.<br/> A list of integer containing kernel size for convolutions. <br/> **Usage:** `--ks 3 5 7` or just `--ks 3`. **Default**: `--ks 3`. It only works when `--cnn` is activated.|
|`ns`| Required: **False**.<br/> A list of integer containing the number of strides for convolutions. <br/> **Usage:** `--ns 3 5 7` or just `--ns 3`. **Default**: `--ns 1`. It only works when `--cnn` is activated.|
|`nfilters`| Required: **False**.<br/> A list of integer containing the number of convolutional operations in each convolutional layer. <br/> **Usage:** `--nfilters 16 32 128` or just `--nfilters 32`. **Default**: `--nfilters 8`. It only works when `--cnn` is activated.|
|`gridProp`| Required: **False**.<br/> Proportion of random search for hyperparameter combination. Please be careful when selecting the prop and make sure you include at least one model to be evaluated, i.e. if the hyperparameters combination includes 10 models, gridProp should be at least 0.1.  <br/> **Usage:** `--gridProp 0.01`. **Default**: `--gridProp 1` which means all the models would be evaluated. It only works when `--cnn`  or --mlp is activated.|


<hr>

### 📈 Genomic Prediction Examples


Check in any of these items to access examples to run the desired DL option.


  - [1. MLP](https://github.com/lauzingaretti/DeepGS/blob/master/inst/md/MLP.md)
  - [2. CNN](https://github.com/lauzingaretti/DeepGS/blob/master/inst/md/CNN.md)
  - [3. Linear regression](https://github.com/lauzingaretti/DeepGS/blob/master/inst/md/LM.md)


<hr>

### 💾 Usage

Please clone this repo i the DeepGP folder

`git clone https://github.com/lauzingaretti/DeepGP.git DeepGP`

Install dependencies

`pip3 install -r requirements.txt`

`cd DeepGP`


`python3 Deep_genomic.py --Xtr /path/to/Xtrain.csv --Ytr  /path/to/Ytrain.csv   --Xval  /path/to/Xval.csv --yval  /path/to/Yval.csv  --trait name of the trait --gridProp 1 --output /path/to/outputFolder/ --cnn --lr 0.025 0.01 --dr1 0 0.01 --dr2 0 --reg1 0.0001 --nconv 3 --act1 linear tanh --ks 3 --hl 1 5  --optimizer Adam  --act2 tanh --dummy --scale`
`



### How to cite?

If you considered our program useful, please cite

`Zingaretti et al submitted ...`

<hr>

#
