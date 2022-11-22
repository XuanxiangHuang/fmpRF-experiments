#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Parse QBF timeout log files
#
################################################################################


def parse_res(slv_name, results_file):
    yes_time = []
    no_time = []

    with open(results_file, 'r') as fp:
        res_lines = fp.readlines()

    test_len = 0
    answer = None
    timeout_cnt = 0
    data_name = None
    print(f"############ {slv_name} ############")
    for i, s in enumerate(res_lines):
        if s.startswith(f"############ ") or i == len(res_lines)-1:
            if i > 0:
                exp_results = f"{data_name} & {test_len}({test_len - len(yes_time) - len(no_time)}) & "
                if len(yes_time):
                    exp_results += "{0:.2f} & {1:.2f} & {2:.2f} & {3:.2f} & " \
                        .format(sum(yes_time), max(yes_time), min(yes_time), sum(yes_time) / len(yes_time))
                else:
                    exp_results += "* & * & * & * & "
                if len(no_time):
                    exp_results += "{0:.2f} & {1:.2f} & {2:.2f} & {3:.2f}" \
                        .format(sum(no_time), max(no_time), min(no_time), sum(no_time) / len(no_time))
                else:
                    exp_results += "* & * & * & *"
                print(exp_results)

                yes_time = []
                no_time = []
                test_len = 0
                timeout_cnt = 0
            data_name = s.lstrip("############ ").rstrip(" ############\n")
        elif s.startswith(f"{data_name}, "):
            line = s.lstrip(f"{data_name},").split('-')
            test_len += 1
        elif s.startswith("*****"):
            answer = "yes"
        elif s.startswith("==============="):
            answer = "no"
        elif s.startswith(f"Solving FMP (CPU) time:"):
            line = s.rstrip(" secs\n").split(': ')
            time_i = float(line[1])
            if time_i >= 1200:
                timeout_cnt += 1
            elif answer == "yes":
                yes_time.append(time_i)
            elif answer == "no":
                no_time.append(time_i)
            else:
                assert False

    return


if __name__ == '__main__':
    parse_res("depqbf", "depqbf_100trees_failed.txt")
    parse_res("caqe", "caqe_bloqqer_100trees_failed.txt")
    parse_res("depqbf", "depqbf_crx_log.txt")
    parse_res("depqbf", "depqbf_glass2_log.txt")
    parse_res("caqe", "caqe_bloqqer_crx_log.txt")

    exit(0)
