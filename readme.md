# Brain Network Transformer with Lazy DataLoader

This implementation of the Brain Network Transformer (BNT) introduces Lazy Dataloaders, enabling efficient training on custom datasets while minimizing memory usage.

## Usage

1. Change the *path* attribute in file *source/conf/dataset/custom.yaml* to the path of your dataset.

2. Run the following command to train the model.

```bash
python -m source --multirun model=bnt dataset=custom 
```

- **model**, default=(bnt,fbnetgen,brainnetcnn,transformer). Which model to use. The value is a list of model names. For example, bnt means Brain Network Transformer, fbnetgen means FBNetGen, brainnetcnn means BrainNetCNN, transformer means VanillaTF.

- **dataset**, default=(ABIDE,ABCD,custom). Which dataset to use. The value is a list of dataset names. For example, ABIDE means ABIDE, ABCD means ABCD. With the Custom Dataset you can add your own dataset with lazy dataloaders.


## Installation

```bash
conda create --name bnt python=3.9
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge wandb
pip install hydra-core --upgrade
conda install -c conda-forge scikit-learn
conda install -c conda-forge pandas
```

## Dependencies

  - python=3.9
  - cudatoolkit=11.3
  - torchvision=0.13.1
  - pytorch=1.12.1
  - torchaudio=0.12.1
  - wandb=0.13.1
  - scikit-learn=1.1.1
  - pandas=1.4.3
  - hydra-core=1.2.0
