#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Reproduce experiments (2QBF)
#
################################################################################
import sys
import pickle
from xrf import XRF, Dataset
import os, time
from math import ceil
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


def rf_fmp(rf_name, data_in, model_in, insts_file, feats_file, qbf_slv):
    name = rf_name
    dp = 0
    sz = 0
    yes_time = []
    no_time = []
    qbf_axps = []
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
        print(f"2QBF: query on feature {f_id} out of {nv} features:")

        rf_md = XRF(model_in, data_in.feature_names, data_in.target_name)
        rf_md_check = XRF(model_in, data_in.feature_names, data_in.target_name)

        dp = rf_md.f.md
        sz = rf_md.f.sz
        qbf_waxp = []
        qbf_axp = []
        exist_q, univ_q, exist_q2, claus = rf_md.fmp_2qbf_enc(inst, f_id)
        if exist_q is None:
            # no AXp containing target feature
            time_i = 0
            print('=============== no AXp exists ===============')
            print(f"Solving FMP (CPU) time: {time_i:.2f} secs")
            if len(qbf_waxp):
                qbf_axps.append(qbf_waxp)
                yes_time.append(time_i)
            else:
                no_time.append(time_i)
            cnf_nv.append(0)
            cnf_claus.append(0)
        else:
            fmp_yes = False
            ########## output the encoding to QDIMACS ##########
            # 0. header: p cnf nof_variables nof_clauses
            # 1. existential quantifier
            # 2. universal quantifier
            # 3. clauses
            assert 0 not in exist_q
            assert 0 not in univ_q
            assert 0 not in exist_q2
            enc_file = f"p cnf {claus.nv} {len(claus.clauses)}\n"
            enc_file += "e " + " ".join(map(str, exist_q)) + " 0\n"
            enc_file += "a " + " ".join(map(str, univ_q)) + " 0\n"
            enc_file += "e " + " ".join(map(str, exist_q2)) + " 0\n"
            for cls in claus:
                assert 0 not in cls
                enc_file += " ".join(map(str, cls)) + " 0\n"
            # create folder to save QBF files
            save_folder_name = f'qbf_encoding_files/{name}'
            try:
                os.stat(save_folder_name)
            except:
                os.makedirs(save_folder_name)
            file_name = f'{save_folder_name}/inst_{i}_feat_{f_id}.QDIMACS'
            with open(file_name, 'w') as f:
                f.write(enc_file)
            ########## output the encoding to QDIMACS ##########
            # output encoding for debugging
            # rf_md.enc.printLits(exist_q)
            # rf_md.enc.printLits(univ_q)
            # rf_md.enc.printLits(exist_q2)
            # for cls in claus:
            #     rf_md.enc.printLits(cls)
            time_solving_start = time.perf_counter()
            ########## calling QBF solver ##########
            # 1. calling qbf solver,
            # 2. parse shell output and extract a weak AXp
            if qbf_slv == "depqbf":
                # cmd = f"./qbf_solvers/depqbf --qdo --no-dynamic-nenofex {file_name}"
                cmd = f"./qbf_solvers/depqbf --max-secs=1200 {file_name}"
                os.system(f'{cmd} > qbf_solution.tmp')
                with open('qbf_solution.tmp', 'r') as f_qbf:
                    solution_lines = f_qbf.readlines()
                    if solution_lines[-1].startswith("UNSAT"):
                        print('=============== no AXp exists ===============')
                    else:
                        fmp_yes = True
                        print(f'***** AXps containing {f_id} *****')
                    # solution_lines = f_qbf.readlines()
                    # if solution_lines[0].startswith("s cnf 0"):
                    #     print('=============== no AXp exists ===============')
                    # else:
                    #     qbf_waxp = []
                    #     slt_id = []
                    #     for s_line in solution_lines[1:]:
                    #         ele = s_line.split(' ')
                    #         assert ele[0] == 'V'
                    #         if int(ele[1]) in exist_q:
                    #             slt_id.append(int(ele[1]))
                    #     for ele in slt_id:
                    #         slt_name = rf_md.enc.nameVar(ele)
                    #         qbf_waxp.append(int(slt_name[6:]))
                    #     qbf_waxp.sort()
                    #     assert f_id in qbf_waxp
            elif qbf_slv == "caqe":
                # --preprocessor bloqqer cannot work with --qdo
                print("Using preprocessor bloqqer:")
                cmd = f"cd qbf_solvers; " \
                      f"./run_caqe.sh ../{file_name} > ../qbf_solution.tmp; " \
                      f"cd ../"
                os.system(cmd)
                with open('qbf_solution.tmp', 'r') as f_qbf:
                    solution_lines = f_qbf.readlines()
                    if solution_lines[-1].startswith("c Unsatisfiable"):
                        print('=============== no AXp exists ===============')
                    else:
                        fmp_yes = True
                        print(f'***** AXps containing {f_id} *****')
                    # solution_lines = f_qbf.readlines()
                    # if solution_lines[1].startswith("s cnf 0"):
                    #     print('=============== no AXp exists ===============')
                    # else:
                    #     qbf_waxp = []
                    #     slt_id = []
                    #     for s_line in solution_lines[2:-2]:
                    #         ele = s_line.strip().split(' ')
                    #         assert ele[0] == 'V'
                    #         if int(ele[1]) in exist_q:
                    #             slt_id.append(int(ele[1]))
                    #     for ele in slt_id:
                    #         slt_name = rf_md.enc.nameVar(ele)
                    #         qbf_waxp.append(int(slt_name[6:]))
                    #     qbf_waxp.sort()
                    #     assert f_id in qbf_waxp

            time_solving_end = time.perf_counter()
            time_i = time_solving_end - time_solving_start
            print(f"Solving FMP (CPU) time: {time_i:.2f} secs")
            if qbf_waxp:
                print("##### check AXp #####")
                qbf_axp = rf_md_check.query_compute_axp(inst, f_id, qbf_waxp)

            if qbf_slv != "depqbf" and qbf_slv != "caqe":
                if len(qbf_axp):
                    qbf_axps.append(qbf_axp)
                    yes_time.append(time_i)
                else:
                    no_time.append(time_i)
                cnf_nv.append(claus.nv)
                cnf_claus.append(len(claus.clauses))
            else:
                if fmp_yes:
                    qbf_axps.append(qbf_axp)
                    yes_time.append(time_i)
                else:
                    no_time.append(time_i)
                cnf_nv.append(claus.nv)
                cnf_claus.append(len(claus.clauses))

    exp_results = f"{name} & {d_len} & "
    exp_results += f"{nv} & #C & {dp} & {sz} & A% & "
    exp_results += "{0:.0f} & {1:.0f} & " \
        .format(sum(cnf_nv) / d_len, sum(cnf_claus) / d_len)
    exp_results += f"{len(qbf_axps)} & "
    exp_results += f"{max([len(x) for x in qbf_axps]):.0f} & "
    exp_results += f"{ceil(sum([len(x) for x in qbf_axps]) / len(qbf_axps)):.0f} & "
    exp_results += "{0:.3f} & {1:.3f} & {2:.3f} & {3:.3f} & " \
        .format(sum(yes_time), max(yes_time), min(yes_time), sum(yes_time) / len(yes_time))
    exp_results += "{0:.3f} & {1:.3f} & {2:.3f} & {3:.3f}\n" \
        .format(sum(no_time), max(no_time), min(no_time), sum(no_time) / len(no_time))

    print(exp_results)

    with open('results/rf_fmp_2qbf.txt', 'a') as f:
        f.write(exp_results)
    return


if __name__ == '__main__':
    args = sys.argv[1:]
    # example: python3 experiment-qbf.py -bench pmlb_qbf.txt 20 caqe
    # program -bench file-list num-of-trees solver-name
    if len(args) >= 4 and args[0] == '-bench':
        bench_name = args[1]
        n_trees = int(args[2])
        qbf_slv_name = args[3]
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
            print(f"Using solver: {qbf_slv_name}\n")
            pk_rf = pickle_load_file(md_file)
            test_insts = f"samples/test_insts/{name}.csv"
            test_feats = f"samples/test_feats/{name}.csv"
            rf_fmp(name, data, pk_rf, test_insts, test_feats, qbf_slv_name)
    exit(0)
