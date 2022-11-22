#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   pick tested instances
#
################################################################################
import sys
import pandas as pd
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
            input_file = f'datasets/{ds}.csv'
            print(f"############ {ds} ############")
            df = pd.read_csv(input_file, sep=',')
            if len(df.index) > num:
                save_df = df.sample(n=num)
                save_df.to_csv(f'samples/test_insts/{ds}.csv', sep=',', index=False, header=False)
            else:
                save_df = df.sample(n=len(df.index))
                save_df.to_csv(f'samples/test_insts/{ds}.csv', sep=',', index=False, header=False)
