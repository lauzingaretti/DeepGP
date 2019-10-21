### Evaluating a MLP architecture


Please clone this repo in the **DeepGP** folder.  

`git clone https://github.com/lauzingaretti/DeepGP.git DeepGP`

Make sure you have the dependencies installed. To install dependencies, please run the following command in bash:

`pip3 install -r requirements.txt`

usually as sudoer.

Go to the Project main folder:

`cd DeepGP`

Prepare the output folder where results should be stored:

`mkdir /path/to/output/Folder`

Running a CNN and tune hyperparameters by typing in bash:

`python3 Deep_genomic.py --Xtr /dataset_wheat/XWheatTrain.csv --ytr /dataset_wheat/YWheatTrain.csv  --Xval /dataset_wheat/XWheatVal.csv --yval /dataset_wheat/YWheatVal.csv  --trait 2  --output /path/to/output/Folder  --lr 0.025 0.01 --dr1 0 0.01 --dr2 0 0.01 --reg1 0.0001  --act2 linear tanh  --hl 1 5  --optimizer Adam --act1 tanh --mlp --gridProp 1
`

Once the running process finished, go to `/path/to/output/Folder`. You may have two folders inside the main directory. One of them is called `best_model`  containing a .zip file with all the weights of the selected model, i.e. the model with the best predictive accuracy.

You can recover the best model and test a new validation dataset by typing in python:

```python
import talos 
from talos import Restore
#make sure you are in the folder containing CNN_optimize.zip
model= Restore(scan_object,'MLP_optimize')
```

 You can use it as an usual keras object. Please check keras (https://keras.io/) and talos (https://autonomio.github.io/) documentation for more examples.

The alternative folder should be named as the architecture, i.e. MLP. This folder will contain a .csv file called cnn_prediction_talos.csv that has the hyperparameters and predictive abilities of the all evaluated models. An auxiliary folder named figures in this directory will contain the figures with the Predictive abilities for the validation set of the evaluated traits.  

Note that it can evaluate both diploids and polyploids organisms.
