#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Explaining RF with Anchors
#
################################################################################
from __future__ import print_function
import pickle
from xrf import XRF, Dataset
import time
import sys, csv
import pandas as pd
import numpy as np
from anchor import anchor_tabular
from math import ceil
from concurrent.futures import ThreadPoolExecutor, TimeoutError

np.random.seed(73)
################################################################################


# Function to wrap the explain_instance method
def run_explainer(explainer_, dpt, fm_explainer):
    return explainer_.explain_instance(np.asarray([dpt], dtype=np.float32), fm_explainer.f.predict, threshold=0.95)


def pickle_load_file(filename):
    try:
        f = open(filename, "rb")
        data = pickle.load(f)
        f.close()
        return data
    except Exception as e:
        print(e)
        print("Cannot load from file", filename)
        exit()


def rf_anchor(rf_name, data_in, model_in, insts_file, feats_file):
    answers = []
    T_time = []

    df = pd.read_csv(f"datasets/{rf_name}.csv")
    features = df.columns[:-1].tolist()
    label_name = df.columns[-1]
    label_values = np.unique(df[label_name].values.astype(int))
    X = df.iloc[:, :-1].values.astype(int)

    print("Features:", features)
    print("Label Name:", label_name)
    print("Label Values:", label_values)
    print("X Shape:", X.shape)

    ###################### read instance file ######################
    with open(insts_file, 'r') as fp:
        inst_lines = fp.readlines()
    ###################### read instance file ######################
    ########### read feature file ###########
    with open(feats_file, 'r') as fp:
        feat_lines = fp.readlines()
    ########### read feature file ###########
    assert len(inst_lines) == len(feat_lines)
    d_len = len(inst_lines)
    nv = len(data_in.feature_names)

    explainer = anchor_tabular.AnchorTabularExplainer(class_names=label_values,
                                                      feature_names=features,
                                                      train_data=X)
    timeout_seconds = 1200

    for i, s in enumerate(inst_lines):
        inst = [float(v.strip()) for v in s.split(',')]
        print(f"{rf_name}, {i}-th inst file out of {d_len}")
        f_id = int(feat_lines[i])
        assert 0 <= f_id <= nv-1
        print(f"CEGAR: query on feature {f_id} out of {nv} features:")
        rf_md = XRF(model_in, data_in.feature_names, data_in.target_name)

        time_solving_start = time.process_time()
        feature_indices = []

        # Use ThreadPoolExecutor for timeout handling
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_explainer, explainer, inst, rf_md)
            try:
                exp = future.result(timeout=timeout_seconds)  # Wait for the result with timeout

                # Process the result if it completes
                feature_indices = exp.features()
                print("Indices of features in explanation:", feature_indices)

                print('Anchor: %s' % (' AND '.join(exp.names())))
                print('Precision: %.2f' % exp.precision())
                print('Coverage: %.2f' % exp.coverage())

            except TimeoutError:
                print("The operation timed out.")

        time_solving_end = time.process_time()
        time_taken = time_solving_end - time_solving_start
        print(f"Time taken: {time_taken} seconds")

        if f_id in feature_indices:
            print('======== Answer Yes ========')
            answers.append(1)
        else:
            print('******** Answer No ********')
            answers.append(0)

        T_time.append(time_taken)

    exp_results = f"{rf_name} & {d_len} & "
    exp_results += "{0:.1f} & {1:.1f} & {2:.1f} & {3:.1f}\n" \
        .format(sum(T_time), max(T_time), min(T_time), sum(T_time) / d_len)

    print(exp_results)

    with open('results/anchor/rf_anchor.txt', 'a') as f:
        f.write(exp_results)

    results_df = pd.DataFrame({
        "answer": answers,
        "runtime": T_time
    })

    # Save the DataFrame to a CSV file
    results_df.to_csv(f"results/anchor/{rf_name}_results.csv", index=False)

    return


if __name__ == '__main__':
    args = sys.argv[1:]
    # example: python3 experiment-cegar.py -bench pmlb_cegar.txt 100
    # program -bench file-list num-of-trees
    if len(args) >= 3 and args[0] == '-bench':
        bench_name = args[1]
        n_trees = int(args[2])
        with open(bench_name, 'r') as fp:
            name_list = fp.readlines()

        for item in name_list:
            name = item.strip()
            print(f"############ {name} ############")
            data = Dataset(filename=f"datasets/{name}.csv", separator=',', use_categorical=False)
            md_file = f"rf_models/RFmv/{name}/{name}_nbestim_{n_trees}_maxdepth_4.mod.pkl"
            if name == "vowel":
                md_file = f"rf_models/RFmv/{name}/{name}_nbestim_{n_trees}_maxdepth_6.mod.pkl"
            elif name == "hayes_roth":
                md_file = f"rf_models/RFmv/{name}/{name}_nbestim_30_maxdepth_4.mod.pkl"
            print(f"Loading {md_file}")
            pk_rf = pickle_load_file(md_file)
            test_insts = f"samples/test_insts/{name}.csv"
            test_feats = f"samples/test_feats/{name}.csv"
            rf_anchor(name, data, pk_rf, test_insts, test_feats)
    exit(0)
