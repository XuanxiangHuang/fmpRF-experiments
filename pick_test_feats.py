#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   pick tested features
#
################################################################################
import random
import csv
import numpy as np
import pandas as pd
import sys
################################################################################

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 3 and args[0] == '-bench':
        bench_name = args[1]
        num = int(args[2])
        with open(bench_name, 'r') as fp:
            datasets = fp.readlines()

        for ds in datasets:
            ds = ds.strip()
            print(f"############ {ds} ############")
            df = pd.read_csv(f'datasets/{ds}.csv')
            feature_names = list(df.columns)
            feature_names.pop()
            feature_name_indices = {name: index for index, name in enumerate(feature_names)}
            nv = len(feature_names)
            ids = [i for i in range(nv)]
            if len(df.index) >= num:
                print(f"Test instances: ", num)
                print("Features: ", nv)
                if num < nv:
                    feat = np.array(random.sample(ids, num))
                else:
                    feat = np.array(np.random.choice(ids, num))
                test_feat = feat.reshape(-1, 1)
            else:
                tmp_num = len(df.index)
                print(f"Test instances: ", tmp_num)
                print("Features: ", nv)
                if tmp_num < nv:
                    feat = np.array(random.sample(ids, tmp_num))
                else:
                    feat = np.array(np.random.choice(ids, tmp_num))
                test_feat = feat.reshape(-1, 1)

            with open(f"samples/test_feats/{ds}.csv", 'w') as f:
                write = csv.writer(f)
                write.writerows(test_feat)