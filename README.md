# AutoGCL: Automated Graph Contrastive Learning via Learnable View Generators (AAAI 2022)

Yin Yihang, Qingzhong Wang, Siyu Huang, Haoyi Xiong, Xiang Zhang

## Full Paper

Please check the our arXiv version [here](https://arxiv.org/abs/2109.10259) for the full paper with supplementary.

## Requirement

```shell
rdkit
pytorch 1.10.0
pytorch_geometric 2.0.2
```

## Dataset Preparation

### TUDataset and MNISTSuperpixel

```shell
$ python download_dataset.py
```

## Semi-supervised Learning

## Unsupervised Learning

### Run

```shell
$ sh un_exp.sh
```

## Transfer `Learning`

### Prepare the Dataset

```shell
$ cd transfer
$ wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
$ unzip chem_dataset.zip
$ rm -rf dataset/*/processed
```

### Run the Fine-tuning Experiments

```shell
$ sh run_chem.sh
```

## Citation

If you find this work helpful, please kindly cite our [paper](https://arxiv.org/abs/2109.10259).

```latex
@article{yin2021autogcl,
  title={AutoGCL: Automated Graph Contrastive Learning via Learnable View Generators},
  author={Yin, Yihang and Wang, Qingzhong and Huang, Siyu and Xiong, Haoyi and Zhang, Xiang},
  journal={arXiv preprint arXiv:2109.10259},
  year={2021}
}
```
