## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

Run this command to see the list of options for training RF.
```
python3 RFxp.py -h
```

Run this command to train RF for dataset ecoli (tree-depth=4, number-of-trees=20):

```
python3 RFxp.py -t -d 4 -n 20 -v -o tmp_models datasets/ecoli.csv
```
Run this command to train RF for the datasets used in the paper.
```
./train.sh
```

## Datasets

All datasets are contained in the `datasets` folder.

## Tested instances/features

You can find tested instances/features for deciding feature membership problem in the `samples` folder.
Besides, you can randomly pick 200 tested features by running:

```
python3 pick_test_feats.py -bench pmlb_cegar.txt 200
```
Likewise, to randomly pick 200 test instances, run this command:
```
python3 pick_test_insts.py -bench pmlb_cegar.txt 200
```

## Pre-trained Models

All pretrained models are contained in the `rf_models/RF2001` folder.
Files `train_20_trees_log.txt`, `train_30_trees_log.txt` and `train_100_trees_log.txt`
store summaries of RF model.

## QBF solvers

We include 2 QBF solvers, 1 preprocessor in the folder `qbf_solvers`.
There is a script `run_caqe.sh` running CAQE + Bloqqer with timeout.
 
## Algorithms

Please check `xrf/xforest.py`.
Function `fmp_2qbf_enc` implements the QBF method.
Function `query_fmp` implements the CEGAR method.

## Experiments

Run this command for reproducing the experiments using CEGAR (Table 1 of the paper).
```
python3 experiment-cegar.py -bench pmlb_qbf.txt 20
```
Run this command for reproducing the experiments using CEGAR (Table 2 of the paper).
```
python3 experiment-cegar.py -bench pmlb_cegar.txt 100
```

Run this command for reproducing the experiments using QBF (Table 1 of the paper).
Using DepQBF:
```
python3 experiment-qbf.py -bench pmlb_qbf.txt 20 depqbf
```
or using CAQE
```
python3 experiment-qbf.py -bench pmlb_qbf.txt 20 caqe
```

Run this command for reproducing the experiments using QBF with timeout (Table 2 of the paper).
Using DepQBF:
```
./run_qbf_timeout.sh 100 depqbf
```
or using CAQE
```
./run_qbf_timeout.sh 100 caqe
```

## Results
All the reported results in `results` folder:

## Citation

If you make use of this code in your own work, please cite our paper:
```
@article{huang2022frprf,
  title={Solving Explainability Queries with Quantification: The Case of Feature Relevancy},
  author={Huang, Xuanxiang and Izza, Yacine and Marques-Silva, Joao},
  journal={},
  year={}
}
```