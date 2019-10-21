### Evaluating a Linear model


Please clone this repo in the **DeepGP** folder.  

`git clone https://github.com/lauzingaretti/DeepGP.git DeepGP`

Make sure you have the dependencies installed. To install dependencies, please run in bash the following command:

`pip3 install -r requirements.txt`

usually as sudoer.

Go to the Project main folder:

`cd DeepGP`

Prepare the output folder where results should be stored:

`mkdir /path/to/output/Folder`

Runing Ridge regression by type in bash:

`python3 Deep_genomic.py --Xtr /dataset_wheat/XWheatTrain.csv --ytr /dataset_wheat/YWheatTrain.csv  --Xval /dataset_wheat/XWheatVal.csv --yval /dataset_wheat/YWheatVal.csv  --ridge
`
Alternatively, you can run lasso by:

`python3 Deep_genomic.py --Xtr /dataset_wheat/XWheatTrain.csv --ytr /dataset_wheat/YWheatTrain.csv  --Xval /dataset_wheat/XWheatVal.csv --yval /dataset_wheat/YWheatVal.csv  --lasso
`


Once the running process finished, go to `/path/to/output/Folder` and in a  folder called **ridge** or **lasso** respectively you can find the figures and predictive abilities.


Note that it can evaluate both diploids and polyploids organisms and although we are not interested in linear models themselves, we implement them in order to compare them with Dep architectures. 
