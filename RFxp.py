#!/usr/bin/env python3
#-*- coding:utf-8 -*-
##
## xprf.py
##

#
#==============================================================================
from __future__ import print_function
from xrf.data import Data
from options import Options
import os
import sys
import pickle
import resource


from xrf import XRF, RF2001, Dataset
import numpy as np



#
#==============================================================================
def show_info():
    """
        Print info message.
    """
    print("c RFxp: Random Forest explainer.")
    print('c')

    
#
#==============================================================================
def pickle_save_file(filename, data):
    try:
        f =  open(filename, "wb")
        pickle.dump(data, f)
        f.close()
    except:
        print("Cannot save to file", filename)
        exit()

def pickle_load_file(filename):
    try:
        f =  open(filename, "rb")
        data = pickle.load(f)
        f.close()
        return data
    except Exception as e:
        print(e)
        print("Cannot load from file", filename)
        exit()    
        
    
#
#==============================================================================
if __name__ == '__main__':
    # parsing command-line options
    options = Options(sys.argv)
    
    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    # showing head
    show_info()

        
        
    if options.files:
        cls = None
        xrf = None
        
        print("loading data ...")
        data = Dataset(filename=options.files[0], 
                    separator=options.separator, use_categorical = options.use_categorical)
            
        if options.train:
            '''
            data = Dataset(filename=options.files[0], mapfile=options.mapfile,
                    separator=options.separator,
                    use_categorical = options.use_categorical)
            '''        
            params = {'n_trees': options.n_estimators,
                        'depth': options.maxdepth}
            cls = RF2001(**params)
            train_accuracy, test_accuracy = cls.train(data)
            
            if options.verb == 1:
                print("----------------------")
                print("Train accuracy: {0:.2f}".format(100. * train_accuracy))
                print("Test accuracy: {0:.2f}".format(100. * test_accuracy))
                print("----------------------")           
            
            xrf = XRF(cls, data.feature_names, data.target_name, options.verb)
            #xrf.test_tree_ensemble()          
            
            bench_name = os.path.basename(options.files[0])
            assert (bench_name.endswith('.csv'))
            bench_name = os.path.splitext(bench_name)[0]
            bench_dir_name = options.output + "/RFmv/" + bench_name
            try:
                os.stat(bench_dir_name)
            except:
                os.makedirs(bench_dir_name)

            basename = (os.path.join(bench_dir_name, bench_name +
                            "_nbestim_" + str(options.n_estimators) +
                            "_maxdepth_" + str(options.maxdepth)))

            modfile =  basename + '.mod.pkl'
            print("saving  model to ", modfile)
            pickle_save_file(modfile, cls)        


        # read a sample from options.explain
        if options.explain:
            options.explain = [float(v.strip()) for v in options.explain.split(',')]
            
            if not xrf:
                print("loading model ...")
                cls = pickle_load_file(options.files[1])
                #print()
                #print("class skl:",cls.forest.classes_)
                #print("feat names:",data.feature_names)
                #print("extended name:",data.extended_feature_names_as_array_strings)
                #print("target:",data.target_name)
                #print()
                xrf = XRF(cls, data.feature_names, data.target_name, options.verb)
                if options.verb:
                    # print test accuracy of the RF model
                    _, X_test, _, y_test = data.train_test_split()
                    X_test = data.transform(X_test) 
                    cls.print_accuracy(X_test, y_test) 
            
            expl = xrf.explain(options.explain, options.xtype)
            
            print(f"expl len: {len(expl)}")
            
            del xrf.enc
            del xrf.x            
          
            