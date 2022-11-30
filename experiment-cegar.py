#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Reproduce experiments (CEGAR)
#
################################################################################
from math import ceil
import sys
import pickle
from xrf import XRF, Dataset
################################################################################


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


def rf_fmp(rf_name, data_in, model_in, insts_file, feats_file):
    name = rf_name
    dp = 0
    sz = 0
    yes_time = []
    no_time = []
    yes_num_sat_calls = []
    no_num_sat_calls = []
    sat_axps = []
    cnf_nv = []
    cnf_claus = []
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

    for i, s in enumerate(inst_lines):
        inst = [float(v.strip()) for v in s.split(',')]
        print(f"{name}, {i}-th inst file out of {d_len}")
        f_id = int(feat_lines[i])
        assert 0 <= f_id <= nv-1
        print(f"CEGAR: query on feature {f_id} out of {nv} features:")

        rf_md = XRF(model_in, data_in.feature_names, data_in.target_name)
        dp = rf_md.f.md
        sz = rf_md.f.sz
        sat_axp, time_i, nv_cnf, claus_cnf, sat_call = rf_md.query_fmp(inst, f_id)

        if len(sat_axp):
            sat_axps.append(sat_axp)
            yes_time.append(time_i)
            yes_num_sat_calls.append(sat_call)
        else:
            no_time.append(time_i)
            no_num_sat_calls.append(sat_call)

        cnf_nv.append(nv_cnf)
        cnf_claus.append(claus_cnf)

    exp_results = f"{name} & {d_len} & "
    exp_results += f"{nv} & #C & {dp} & {sz} & A% & "
    exp_results += "{0:.0f} & {1:.0f} & " \
        .format(sum(cnf_nv) / d_len, sum(cnf_claus) / d_len)
    exp_results += f"{len(sat_axps)} & "
    exp_results += f"{max([len(x) for x in sat_axps]):.0f} & "
    exp_results += f"{ceil(sum([len(x) for x in sat_axps]) / len(sat_axps)):.0f} & "
    exp_results += "{0:.3f} & {1:.3f} & {2:.3f} & {3:.3f} & " \
        .format(sum(yes_time), max(yes_time), min(yes_time), sum(yes_time) / len(yes_time))
    exp_results += "{0:.1f} & " \
        .format(sum(yes_num_sat_calls) / len(yes_num_sat_calls))
    exp_results += "{0:.3f} & {1:.3f} & {2:.3f} & {3:.3f} & " \
        .format(sum(no_time), max(no_time), min(no_time), sum(no_time) / len(no_time))
    exp_results += "{0:.1f}\n" \
        .format(sum(no_num_sat_calls) / len(no_num_sat_calls))

    print(exp_results)

    with open('results/rf_fmp_cegar.txt', 'a') as f:
        f.write(exp_results)

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
            rf_fmp(name, data, pk_rf, test_insts, test_feats)
    exit(0)
