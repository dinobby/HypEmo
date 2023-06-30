# HypEmo
The official implementation of ACL 2023 paper [Label-Aware Hyperbolic Embeddings for Fine-grained Emotion Classification](https://arxiv.org/abs/2306.14822)

This code is tested under Python3.10.11.

### Training:
First, install the packages via the following command:
```pip install -r requirements.txt```

you can optionally open `config.py` and change the dataset and hyperparameters.

Second, just run `train.py` to start training!

### Note
`train_label_embedding.py` contains the script for training the hyperbolic label embeddings.

This script originates from [https://github.com/dalab/hyperbolic_cones](https://github.com/dalab/hyperbolic_cones).

If you are using other datasets, you may run this script on your custom label to obtain hyperbolic embeddings.

Once it is done, you will get a `.bin` in the `label_tree` folder, and you can run the main script by `train.py`.

If you are not using a custom dataset, you can skip this section and directly run `train.py`.

### Credit
We have prepared all the processed data in the `data` folder, which is from [GoEmotion](https://arxiv.org/abs/2005.00547) and [EmpatheticDialogues](https://arxiv.org/abs/1811.00207). We also rely on [Hyperbolic cones](https://arxiv.org/abs/1804.01882) to learn hyperbolic embeddings.
