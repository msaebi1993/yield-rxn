
## Predicting Yield of Chemical Reactions using Deep Learning

### Installation

1. First install Anaconda. 
2. Create a conda environment with
```
conda create --name rxntorch python=3.6
```
3. Then, activate the new conda environment with
```
conda activate rxntorch
```
4. Install RDKit
```
conda install -c rdkit rdkit 
```
5. Installing PyTorch with a CUDA enabled version
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
6. Install scikit-learn with
```
conda install scikit-learn
```

Finally, clone this repository to your local machine.

### Getting Started


1. Run: 
```
 pip install requirements
```
2.To train the model, run:
```
python train_yield.py
```
Arguments:
-c: Path to the dataset
--train_split: train-test split ratio
-o: Model name
--epochs: Number of epochs
--seed: Random seed
--layers: Number of layes
--hidden: Hidden size for all layers
--lr_decay: Learning rate decay
--use_domain: Use chemical features or not
--batch_size: Size of mini-batch
--dropout_rate: Droput rate

If using chemical features (domain features) you need the json file containg the features. Otherwise, you can just use smiles strings.
