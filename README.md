# Language-Identification-for-Speech-SSL-Models

This is a implementation of language identification for evaluating speech self-supervised models. 
The evaluating dataset is identical to [ML-SUPERB](https://arxiv.org/abs/2305.10615), 
while this repository simply adds a linear layer after speech SSL mdoel instead of doing ASR-like training. 
The code is implemented under [S3PRL](https://github.com/s3prl/s3prl),
please follow the instructions below to prepare data and environment.

# Preparing 
1. Install [S3PRL](https://github.com/s3prl/s3prl)
2. Execute the following command to prepare LID data of [ML-SUPERB](https://arxiv.org/abs/2305.10615) 
  ```
  bash tidy_lid_data/preprocess.sh [OUT_DIR]
  ```
3. Modify "root_dir" to OUT_DIR in lid/config.yaml
4. Move lid/ folder to s3prl/s3prl/downstream/
5. cd s3prl/s3prl

# Training LID Downstream
An example of training LID downstream for HuBERT Base: 
```
python3 run_downstream.py -m train -n lid_hubert -d lid -u hubert_base 
```
Evaluating:
```
python3 run_downstream.py -m evaluate -e result/downstream/lid_hubert 
```

# Performance on test set 
DistilHuBERT: 51.26%
