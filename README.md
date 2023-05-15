# HypEmo
The official implementation of ACL 2023 paper "Label-Aware Hyperbolic Embeddings for Fine-grained Emotion Classification."
This repo is still in construction and will be available soon!

### Notes:
1. run the following script to start training:

`python3 train_ED_bert.py` or `python3 train_GE_bert.py`

2. `hyper_emb` contains the script for training the hyperbolic label embeddings.

If a new dataset is employed, one should first train its label embbedding vai `animation_train.py`.

Once the hyperbolic label embedding is obtained, move the saved file to `\label_tree`.

3. Other directories in the home directory are for the poincare manifold, 

which is used to train HNN and measure the hyperbolic distance when training an end-to-end model.
