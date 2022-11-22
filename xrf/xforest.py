
#from sklearn.ensemble._voting import VotingClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
import numpy as np
import resource

import collections
from six.moves import range
import six
import math

from xrf.data import Data
from .tree import Forest

#from .encode import SATEncoder
from pysat.formula import CNF, WCNF, IDPool
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from pysat.examples.lbx import LBX
from pysat.examples.rc2 import RC2

import time
    

#
#==============================================================================
class Dataset(Data):
    """
        Class for representing dataset (transactions).
    """
    def __init__(self, filename=None, fpointer=None, mapfile=None,
            separator=' ', use_categorical = False):
        super().__init__(filename, fpointer, mapfile, separator, use_categorical)
        
        # split data into X and y
        self.feature_names = self.names[:-1]
        self.nb_features = len(self.feature_names)
        self.use_categorical = use_categorical
        
        samples = np.asarray(self.samps)
        if not all(c.isnumeric() for c in samples[:, -1]):            
            le = LabelEncoder()
            le.fit(samples[:, -1])
            samples[:, -1]= le.transform(samples[:, -1])
            self.class_names = le.classes_ 
            print(le.classes_)
            print(samples[1:4, :])
        
        samples = np.asarray(samples, dtype=np.float32)
        self.X = samples[:, 0: self.nb_features]
        self.y = samples[:, self.nb_features]
        self.num_class = len(set(self.y))
        self.target_name = list(range(self.num_class))          
        
        print("c nof features: {0}".format(self.nb_features))
        print("c nof classes: {0}".format(self.num_class))
        print("c nof samples: {0}".format(len(self.samps)))
        
        # check if we have info about categorical features
        if (self.use_categorical):
            self.target_name = self.class_names            
            
            self.binarizer = {}
            for i in self.categorical_features:
                self.binarizer.update({i: OneHotEncoder(categories='auto', sparse=False)})#,
                self.binarizer[i].fit(self.X[:,[i]])
        else:
            self.categorical_features = []
            self.categorical_names = []            
            self.binarizer = []           
        #feat map
        self.mapping_features()
        
        
            
    def train_test_split(self, test_size=0.2, seed=0):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=seed)
           

    def transform(self, x):
        if(len(x) == 0):
            return x
        if (len(x.shape) == 1):
            x = np.expand_dims(x, axis=0)
        if (self.use_categorical):
            assert(self.binarizer != [])
            tx = []
            for i in range(self.nb_features):
                #self.binarizer[i].drop = None
                if (i in self.categorical_features):
                    self.binarizer[i].drop = None
                    tx_aux = self.binarizer[i].transform(x[:,[i]])
                    tx_aux = np.vstack(tx_aux)
                    tx.append(tx_aux)
                else:
                    tx.append(x[:,[i]])
            tx = np.hstack(tx)
            return tx
        else:
            return x

    def transform_inverse(self, x):
        if(len(x) == 0):
            return x
        if (len(x.shape) == 1):
            x = np.expand_dims(x, axis=0)
        if (self.use_categorical):
            assert(self.binarizer != [])
            inverse_x = []
            for i, xi in enumerate(x):
                inverse_xi = np.zeros(self.nb_features)
                for f in range(self.nb_features):
                    if f in self.categorical_features:
                        nb_values = len(self.categorical_names[f])
                        v = xi[:nb_values]
                        v = np.expand_dims(v, axis=0)
                        iv = self.binarizer[f].inverse_transform(v)
                        inverse_xi[f] =iv
                        xi = xi[nb_values:]

                    else:
                        inverse_xi[f] = xi[0]
                        xi = xi[1:]
                inverse_x.append(inverse_xi)
            return inverse_x
        else:
            return x

    def transform_inverse_by_index(self, idx):
        if (idx in self.extended_feature_names):
            return self.extended_feature_names[idx]
        else:
            print("Warning there is no feature {} in the internal mapping".format(idx))
            return None

    def transform_by_value(self, feat_value_pair):
        if (feat_value_pair in self.extended_feature_names.values()):
            keys = (list(self.extended_feature_names.keys())[list( self.extended_feature_names.values()).index(feat_value_pair)])
            return keys
        else:
            print("Warning there is no value {} in the internal mapping".format(feat_value_pair))
            return None

    def mapping_features(self):
        self.extended_feature_names = {}
        self.extended_feature_names_as_array_strings = []
        counter = 0
        if (self.use_categorical):
            for i in range(self.nb_features):
                if (i in self.categorical_features):
                    for j, _ in enumerate(self.binarizer[i].categories_[0]):
                        self.extended_feature_names.update({counter:  (self.feature_names[i], j)})
                        self.extended_feature_names_as_array_strings.append("f{}_{}".format(i,j)) # str(self.feature_names[i]), j))
                        counter = counter + 1
                else:
                    self.extended_feature_names.update({counter: (self.feature_names[i], None)})
                    self.extended_feature_names_as_array_strings.append("f{}".format(i)) #(self.feature_names[i])
                    counter = counter + 1
        else:
            for i in range(self.nb_features):
                self.extended_feature_names.update({counter: (self.feature_names[i], None)})
                self.extended_feature_names_as_array_strings.append("f{}".format(i))#(self.feature_names[i])
                counter = counter + 1

    def readable_sample(self, x):
        readable_x = []
        for i, v in enumerate(x):
            if (i in self.categorical_features):
                readable_x.append(self.categorical_names[i][int(v)])
            else:
                readable_x.append(v)
        return np.asarray(readable_x)

    
    def test_encoding_transformes(self, X_train):
        # test encoding

        X = X_train[[0],:]

        print("Sample of length", len(X[0])," : ", X)
        enc_X = self.transform(X)
        print("Encoded sample of length", len(enc_X[0])," : ", enc_X)
        inv_X = self.transform_inverse(enc_X)
        print("Back to sample", inv_X)
        print("Readable sample", self.readable_sample(inv_X[0]))
        assert((inv_X == X).all())

        '''
        for i in range(len(self.extended_feature_names)):
            print(i, self.transform_inverse_by_index(i))
        for key, value in self.extended_feature_names.items():
            print(value, self.transform_by_value(value))   
        '''       
#
#==============================================================================
class XRF(object):
    """
        class to encode and explain Random Forest classifiers.
    """
    
    def __init__(self, model, feature_names, class_names, verb=0):
        self.cls = model
        #self.data = dataset
        self.verbose = verb
        self.feature_names = feature_names
        self.class_names = class_names
        self.fnames = [f'f{i}' for i in range(len(feature_names))]
        self.f = Forest(model, self.fnames)
        
        if self.verbose > 2:
            self.f.print_trees()
        if self.verbose:    
            print("c RF sz:", self.f.sz)
            print('c max-depth:', self.f.md)
            print('c nof DTs:', len(self.f.trees))
        
    def __del__(self):
        if 'enc' in dir(self):
            del self.enc
        if 'x' in dir(self):
            if self.x.slv is not None:
                self.x.slv.delete()
            del self.x
        del self.f
        self.f = None
        del self.cls
        self.cls = None
        
    def encode(self, inst):
        """
            Encode a tree ensemble trained previously.
        """
        if 'f' not in dir(self):
            self.f = Forest(self.cls, self.fnames)
            #self.f.print_tree()
            
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime            
            
        self.enc = SATEncoder(self.f, self.feature_names, len(self.class_names), self.fnames)
        
        #inst = self.data.transform(np.array(inst))[0]
        formula, _, _, _ = self.enc.encode(np.array(inst))
        
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - time        
        
        if self.verbose:
            print('c nof vars:', formula.nv) # number of variables 
            print('c nof clauses:', len(formula.clauses)) # number of clauses    
            print('c encoding time: {0:.3f}'.format(time))            
        
    def explain(self, inst, xtype='abd'):
        """
            Explain a prediction made for a given sample with a previously
            trained RF.
        """
        
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime          
        
        if 'enc' not in dir(self):
            self.encode(inst)
        
        #inpvals = self.data.readable_sample(inst)
        inpvals = np.asarray(inst)
        preamble = []
        for f, v in zip(self.feature_names, inpvals):
            if f not in str(v):
                preamble.append('{0} = {1}'.format(f, v))
            else:
                preamble.append(v)
                    
        inps = self.fnames # input (feature value) variables
        #print("inps: {0}".format(inps))
            
        self.x = SATExplainer(self.enc, inps, preamble, self.class_names, verb=self.verbose)
        #inst = self.data.transform(np.array(inst))[0]
        expl = self.x.explain(np.array(inst), xtype)

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - time 
        
        if self.verbose:
            print("c Total time: {0:.3f}".format(time))
            
        return expl
    
    def enumerate(self, inst, xtype='con', smallest=True):
        """
            list all XPs
        """
        if 'enc' not in dir(self):
            self.encode(inst)
            
        if 'x' not in dir(self):
            inpvals = np.asarray(inst)
            preamble = []
            for f, v in zip(self.feature_names, inpvals):
                if f not in str(v):
                    preamble.append('{0} = {1}'.format(f, v))
                else:
                    preamble.append(v)
                    
            inps = self.fnames
            self.x = SATExplainer(self.enc, inps, preamble, self.class_names)
            
        for expl in self.x.enumerate(np.array(inst), xtype, smallest):
            yield expl

    ##############################
    def query_axp(self, inst, axp):
        """
            AXp query: checking if a given explanation is an AXp.
            :param inst: given instance/sample
            :param axp: given AXp
        """
        if 'enc' not in dir(self):
            self.encode(inst)

        inpvals = np.asarray(inst)
        preamble = []
        for f, v in zip(self.feature_names, inpvals):
            if f not in str(v):
                preamble.append('{0} = {1}'.format(f, v))
            else:
                preamble.append(v)

        inps = self.fnames  # input (feature value) variables
        self.x = SATExplainer(self.enc, inps, preamble, self.class_names, verb=self.verbose)
        # preprocessing, this should be called only once
        self.x.prepare_selectors(inst)
        return self.x.is_axp(axp)

    def query_fmp(self, inst, feat_t):
        """
            Feature membership query: if there exist a weak AXp
            containing target feature `feat_t`?
            :param inst: given instance/sample
            :param feat_t: target feature
        """
        if 'enc' not in dir(self):
            self.encode(inst)

        inpvals = np.asarray(inst)
        preamble = []
        for f, v in zip(self.feature_names, inpvals):
            if f not in str(v):
                preamble.append('{0} = {1}'.format(f, v))
            else:
                preamble.append(v)

        inps = self.fnames  # input (feature value) variables
        self.x = SATExplainer(self.enc, inps, preamble, self.class_names, verb=self.verbose)
        # preprocessing, this should be called only once
        self.x.prepare_selectors(inst)
        time_solving_start = time.perf_counter()

        axp = []
        waxp, sat_calls, cegar_nvars, cegar_ncls = self.x.fmp_cegar(feat_t)
        if waxp:
            print(f"weak AXp: {waxp}")
            axp = self.x.extract_axp(waxp)
            assert feat_t in axp, f"target {feat_t} not in computed AXp"
            assert self.x.is_axp(axp), f"FMP error: incorrect AXp {axp}"
            print(f"AXp: {axp}")
        else:
            print('=============== no AXp exists ===============')
        print(f"#var in CEGAR loop: {cegar_nvars}")
        print(f"#clau in CEGAR loop: {cegar_ncls}")

        time_solving_end = time.perf_counter()
        solving_time = time_solving_end - time_solving_start
        print(f"Solving FMP (CPU) time: {solving_time:.2f} secs")
        print(f'Number of SAT calls: {sat_calls}')
        if sat_calls == 0:
            return axp, solving_time, cegar_nvars, cegar_ncls, sat_calls
        else:
            return axp, solving_time, self.enc.cnf.nv + cegar_nvars, len(self.enc.cnf.clauses) + cegar_ncls, sat_calls

    ##############################
    
    ##############################

    def fmp_2qbf_enc_all_intvs(self, inst, feat_t):
        """
            (deprecated)
            Encoding FMP of RF into 2QBF, and save the encoding to QDIMACS file.
            :param inst: given instance/sample
            :param feat_t: target feature
        """
        if 'enc' not in dir(self):
            if 'f' not in dir(self):
                self.f = Forest(self.cls, self.fnames)
            self.enc = SATEncoder(self.f, self.feature_names, len(self.class_names), self.fnames)
            cnf_RF, eq1_predicate = self.enc.qbf_encode_rf_all_intvs(np.array(inst))
            cnf_t = self.enc.qbf_encode_t(np.array(inst))

        ########## define quantifier ##########
        cnf = CNF()
        # the selector of target feature
        slt_t = None
        # selector s_i => given value v_i
        inps = self.fnames  # input (feature value) variables
        # preparing the selectors with the current sample
        # assums is the selectors
        assums = []  # var selectors to be used as assumptions
        sel2fid = {}  # selectors to original feature ids
        sel2vid = {}  # selectors to categorical feature ids
        sel2v = {}  # selectors to (categorical/interval) values
        pick_predicate = []
        ########## original RF ##########
        for i, (inp, val) in enumerate(zip(inps, inst), 1):
            if len(self.enc.intvs[inp]):
                v = next((intv for intv in self.enc.intvs[inp] if intv > val), None)
                assert (v is not None)
                selv = self.enc.newVar(f'selv_{inp}')
                assums.append(selv)
                pick_var = self.enc.newVar(f'pick_{inp}^RF')
                pick_predicate.append(pick_var)
                assert (selv not in sel2fid)
                sel2fid[selv] = int(inp[1:])
                sel2vid[selv] = [i - 1]
                for j, p in enumerate(self.enc.ivars[inp]):
                    cl = [-selv]
                    if j == self.enc.imaps[inp][v]:
                        cl += [p]
                        sel2v[selv] = p
                        cnf_RF.append(cl + [-pick_var])
                        cnf_RF.append([selv, pick_var])
                        cnf_RF.append([-p, pick_var])
                    else:
                        cl += [-p]
        ########## t-th RF ##########
        for i, (inp, val) in enumerate(zip(inps, inst), 1):
            if len(self.enc.intvs[inp]):
                v = next((intv for intv in self.enc.intvs[inp] if intv > val), None)
                selv = self.enc.newVar(f'selv_{inp}')
                assert selv in assums
                assert selv in sel2fid
                assert selv in sel2vid
                assert sel2fid[selv] == int(inp[1:])
                assert sel2vid[selv] == [i - 1]
                for j, p in enumerate(self.enc.ivars_t[inp]):
                    cl = [-selv]
                    if j == self.enc.imaps[inp][v]:
                        cl += [p]
                        str1 = self.enc.nameVar(sel2v[selv])
                        str2 = self.enc.nameVar(p)
                        assert str1.rstrip("^RF") == str2
                        if int(inp[1:]) != feat_t:
                            cnf_t.append(cl)
                        else:
                            slt_t = selv
                    else:
                        cl += [-p]

        # it is possible that a feature has 0 values,
        # hence there is no selector for it
        for i, inp in enumerate(inps):
            assert i == int(inp[1:])
            if len(self.enc.intvs[inp]) == 0:
                assert self.enc.newVar(f'selv_{inp}') not in assums
                if i == feat_t:
                    print("interested feature not in RF, so there is no AXp")
                    return None, None, None, None
            else:
                assert self.enc.newVar(f'selv_{inp}') in assums

        assert self.enc.newVar(f'selv_f{feat_t}') == slt_t
        winner = self.enc.newVar(f'winner_{self.enc.cmaj}^RF')
        # if /\ EqualsOne(..) /\ (s_i => z_j) => "same prediction"
        cnf.append([-ele for ele in eq1_predicate] +
                   [-ele for ele in pick_predicate] + [winner])
        cnf.append([slt_t])
        cnf.extend(cnf_RF.clauses)
        cnf.extend(cnf_t.clauses)
        # for intervals vars, add them to the second level of universal quantifier
        for_all = []
        for f, intvs in six.iteritems(self.enc.ivars):
            print(f, intvs)
            if not len(intvs):
                continue
            for_all.extend(intvs)
        # for the rest of vars in cnf_RF and cnf_t add them to the third level of exist quantifier
        rest = set()
        for cls in cnf_RF.clauses:
            for ele in cls:
                ele = abs(ele)
                if ele not in assums and ele not in for_all:
                    rest.add(ele)
        for cls in cnf_t.clauses:
            for ele in cls:
                ele = abs(ele)
                if ele not in assums and ele not in for_all:
                    rest.add(ele)
        exist_q2 = list(rest)
        exist_q2.sort()

        return assums, for_all, exist_q2, cnf

    def query_compute_axp(self, inst, feat_t, waxp):
        """
            Given an weak AXp containing feature t,
            extracting one AXp.
            :param inst: given instance/sample
            :param feat_t: target feature
            :param waxp: given weak AXp
        """
        if 'enc' not in dir(self):
            self.encode(inst)

        inpvals = np.asarray(inst)
        preamble = []
        for f, v in zip(self.feature_names, inpvals):
            if f not in str(v):
                preamble.append('{0} = {1}'.format(f, v))
            else:
                preamble.append(v)

        inps = self.fnames  # input (feature value) variables
        self.x = SATExplainer(self.enc, inps, preamble, self.class_names, verb=self.verbose)
        # preprocessing, this should be called only once
        self.x.prepare_selectors(inst)

        print(f"weak AXp: {waxp}")
        axp = self.x.extract_axp(waxp)
        assert feat_t in axp, f"target {feat_t} not in computed AXp"
        assert self.x.is_axp(axp), f"FMP error: incorrect AXp {axp}"
        print(f"AXp: {axp}")

        return axp

    def fmp_2qbf_enc(self, inst, feat_t):
        """
            Encoding FMP of RF into 2QBF, and save the encoding to QDIMACS file.
            :param inst: given instance/sample
            :param feat_t: target feature
        """
        if 'enc' not in dir(self):
            if 'f' not in dir(self):
                self.f = Forest(self.cls, self.fnames)
            self.enc = SATEncoder(self.f, self.feature_names, len(self.class_names), self.fnames)
            cnf_RF = self.enc.qbf_encode_rf(np.array(inst))
            cnf_t = self.enc.qbf_encode_t(np.array(inst))

        ########## feature values selectors vectors ##########
        # for value selectors vector <y_i> (auxiliary boolean vars)
        # add them to the second level of universal quantifier
        # feat_val_slts = []
        univ_q = []
        feat2val_slt = dict()
        for f, intvs in six.iteritems(self.enc.ivars):
            # self.enc.printLits(intvs)
            if not len(intvs):
                continue
            # feat_val_slt = self.enc.newVar(f'{f}_val_slt')
            # feat_val_slts.append(feat_val_slt)
            num_digits = math.ceil(math.log2(len(intvs)))
            aux_y = []
            for idx in range(num_digits):
                # the left most y_idx is the least bit
                aux_y.append(self.enc.newVar(f'{f}_y_{idx}'))
            feat2val_slt.update({f: aux_y})
            univ_q.extend(aux_y)
            for k, ele in enumerate(intvs):
                bin_k = [int(j) for j in list(np.binary_repr(k, width=num_digits))]
                # reverse the boolean encoding, so the left most var is the least bit
                bin_k.reverse()
                # print(f"{k}: {bin_k}")
                selv = self.enc.newVar(f'selv_{f}')
                tmp = []
                for kk, var in enumerate(bin_k):
                    assert var == 0 or var == 1
                    if var == 0:
                        tmp.append(-aux_y[kk])
                    else:
                        tmp.append(aux_y[kk])
                # self.enc.printLits([selv] + [-j for j in tmp] + [intvs[k]])
                cnf_RF.append([selv] + [-j for j in tmp] + [intvs[k]])
        # self.enc.printLits(feat_val_slts)
        # for item in feat2val_slt:
        #     self.enc.printLits(feat2val_slt[item])
        ########## selector to given values ##########
        cnf = CNF()
        # the selector of target feature
        slt_t = None
        # selector s_i => given value v_i
        inps = self.fnames  # input (feature value) variables
        # preparing the selectors with the current sample
        # assums is the selectors
        assums = []  # var selectors to be used as assumptions
        sel2fid = {}  # selectors to original feature ids
        sel2vid = {}  # selectors to categorical feature ids
        sel2v = {}  # selectors to (categorical/interval) values
        ########## original RF ##########
        for i, (inp, val) in enumerate(zip(inps, inst), 1):
            if len(self.enc.intvs[inp]):
                v = next((intv for intv in self.enc.intvs[inp] if intv > val), None)
                assert (v is not None)
                selv = self.enc.newVar(f'selv_{inp}')
                assums.append(selv)
                assert (selv not in sel2fid)
                sel2fid[selv] = int(inp[1:])
                sel2vid[selv] = [i-1]
                for j, p in enumerate(self.enc.ivars[inp]):
                    cl = [-selv]
                    if j == self.enc.imaps[inp][v]:
                        cl += [p]
                        sel2v[selv] = p
                        cnf_RF.append(cl)
                    else:
                        cl += [-p]
        ########## t-th RF ##########
        for i, (inp, val) in enumerate(zip(inps, inst), 1):
            if len(self.enc.intvs[inp]):
                v = next((intv for intv in self.enc.intvs[inp] if intv > val), None)
                selv = self.enc.newVar(f'selv_{inp}')
                assert selv in assums
                assert selv in sel2fid
                assert selv in sel2vid
                assert sel2fid[selv] == int(inp[1:])
                assert sel2vid[selv] == [i-1]
                for j, p in enumerate(self.enc.ivars_t[inp]):
                    cl = [-selv]
                    if j == self.enc.imaps[inp][v]:
                        cl += [p]
                        str1 = self.enc.nameVar(sel2v[selv])
                        str2 = self.enc.nameVar(p)
                        assert str1.rstrip("^RF") == str2
                        if int(inp[1:]) != feat_t:
                            cnf_t.append(cl)
                        else:
                            slt_t = selv
                    else:
                        cl += [-p]

        # it is possible that a feature has 0 values,
        # hence there is no selector for it
        for i, inp in enumerate(inps):
            assert i == int(inp[1:])
            if len(self.enc.intvs[inp]) == 0:
                assert self.enc.newVar(f'selv_{inp}') not in assums
                if i == feat_t:
                    print("interested feature not in RF, so there is no AXp")
                    return None, None, None, None
            else:
                assert self.enc.newVar(f'selv_{inp}') in assums

        assert self.enc.newVar(f'selv_f{feat_t}') == slt_t
        cnf.append([slt_t])
        cnf.extend(cnf_RF.clauses)
        cnf.extend(cnf_t.clauses)
        # for the rest of vars, add them to the third level of exist quantifier
        rest = set()
        for cls in cnf_RF.clauses:
            for ele in cls:
                ele = abs(ele)
                if ele not in assums and ele not in univ_q:
                    rest.add(ele)
        for cls in cnf_t.clauses:
            for ele in cls:
                ele = abs(ele)
                if ele not in assums and ele not in univ_q:
                    rest.add(ele)
        exist_q2 = list(rest)
        exist_q2.sort()

        return assums, univ_q, exist_q2, cnf

    ##############################
        
#
#==============================================================================
class SATEncoder(object):
    """
        Encoder of Random Forest classifier into SAT.
    """
    
    def __init__(self, forest, feats, nof_classes, extended_feature_names,  from_file=None):
        self.forest = forest
        #self.feats = {f: i for i, f in enumerate(feats)}
        self.num_class = nof_classes
        self.vpool = IDPool()
        self.extended_feature_names = extended_feature_names
        
        #encoding formula
        self.cnf = None

        # for interval-based encoding
        self.intvs, self.imaps, self.ivars, self.thvars = None, None, None, None
       
        
    def newVar(self, name):
        """
            If a variable named 'name' already exists then
            return its id; otherwise create a new var
        """
        if name in self.vpool.obj2id: #var has been already created 
            return self.vpool.obj2id[name]
        var = self.vpool.id('{0}'.format(name))
        return var
    
    def nameVar(self, vid):
        """
            input a var id and return a var name
        """
        return self.vpool.obj(abs(vid))
    
    def printLits(self, lits):
        print(["{0}{1}".format("-" if p<0 else "",self.vpool.obj(abs(p))) for p in lits])
    
    def traverse(self, tree, k, clause):
        """
            Traverse a tree and encode each node.
        """

        if tree.children:
            f = tree.name
            v = tree.threshold
            pos = neg = []
            if f in self.intvs:
                d = self.imaps[f][v]
                pos, neg = self.thvars[f][d], -self.thvars[f][d]
            else:
                var = self.newVar(tree.name)
                pos, neg = var, -var
                #print("{0} => {1}".format(tree.name, var))
                
            assert (pos and neg)
            self.traverse(tree.children[0], k, clause + [-neg])
            self.traverse(tree.children[1], k, clause + [-pos])            
        else:  # leaf node
            cvar = self.newVar('class{0}_tr{1}'.format(tree.values,k))
            self.cnf.append(clause + [cvar])
            #self.printLits(clause + [cvar])

    def compute_intervals(self):
        """
            Traverse all trees in the ensemble and extract intervals for each
            feature.

            At this point, the method only works for numerical datasets!
        """

        def traverse_intervals(tree):
            """
                Auxiliary function. Recursive tree traversal.
            """

            if tree.children:
                f = tree.name
                v = tree.threshold
                if f in self.intvs:
                    self.intvs[f].add(v)

                traverse_intervals(tree.children[0])
                traverse_intervals(tree.children[1])

        # initializing the intervals
        self.intvs = {'{0}'.format(f): set([]) for f in self.extended_feature_names if '_' not in f}

        for tree in self.forest.trees:
            traverse_intervals(tree)
                
        # OK, we got all intervals; let's sort the values
        self.intvs = {f: sorted(self.intvs[f]) + ([math.inf] if len(self.intvs[f]) else []) for f in six.iterkeys(self.intvs)}

        self.imaps, self.ivars = {}, {}
        self.thvars = {}
        for feat, intvs in six.iteritems(self.intvs):
            self.imaps[feat] = {}
            self.ivars[feat] = []
            self.thvars[feat] = []
            for i, ub in enumerate(intvs):
                self.imaps[feat][ub] = i

                ivar = self.newVar('{0}_intv{1}'.format(feat, i))
                self.ivars[feat].append(ivar)
                #print('{0}_intv{1}'.format(feat, i))
                
                if ub != math.inf:
                    #assert(i < len(intvs)-1)
                    thvar = self.newVar('{0}_th{1}'.format(feat, i))
                    self.thvars[feat].append(thvar)
                    #print('{0}_th{1}'.format(feat, i))



    def encode(self, sample):
        """
            Do the job.
        """
        
        ###print('Encode RF into SAT ...')

        self.cnf = CNF()
        # getting a tree ensemble
        #self.forest = Forest(self.model, self.extended_feature_names)
        num_tree = len(self.forest.trees)
        self.forest.predict_inst(sample)

        #introducing class variables
        #cvars = [self.newVar('class{0}'.format(i)) for i in range(self.num_class)]
        
        # define Tautology var
        vtaut = self.newVar('Tautology')
        self.cnf.append([vtaut])
            
        # introducing class-tree variables
        ctvars = [[] for t in range(num_tree)]
        for k in range(num_tree):
            for j in range(self.num_class):
                var = self.newVar('class{0}_tr{1}'.format(j,k))
                ctvars[k].append(var)       

        # traverse all trees and extract all possible intervals
        # for each feature
        ###print("compute intervarls ...")
        self.compute_intervals()
        
        #print(self.intvs)
        #print([len(self.intvs[f]) for f in self.intvs])
        #print(self.imaps) 
        #print(self.ivars)
        #print(self.thvars)
        #print(ctvars)
        
        
        ##print("encode trees ...")
        # traversing and encoding each tree
        for k, tree in enumerate(self.forest.trees):
            #print("Encode tree#{0}".format(k))
            # encoding the tree     
            self.traverse(tree, k, [])
            # exactly one class var is true
            #self.printLits(ctvars[k])
            card = CardEnc.atmost(lits=ctvars[k], vpool=self.vpool,encoding=EncType.cardnetwrk) 
            self.cnf.extend(card.clauses)
        
        
            
        # calculate the majority class   
        self.cmaj = self.forest.predict_inst(sample)       
        
        ##print("encode majority class ...")                
        #Cardinality constraint AtMostK to capture a j_th class
        
        if(self.num_class == 2):
            rhs = math.floor(num_tree / 2) + 1
            if(self.cmaj==1 and not num_tree%2):
                rhs = math.floor(num_tree / 2)      
            lhs = [ctvars[k][1 - self.cmaj] for k in range(num_tree)]
            atls = CardEnc.atleast(lits = lhs, bound = rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(atls)
        else: 
            zvars = []
            zvars.append([self.newVar('z_0_{0}'.format(k)) for k in range (num_tree) ])
            zvars.append([self.newVar('z_1_{0}'.format(k)) for k in range (num_tree) ])
            ##
            rhs = num_tree
            lhs0 = zvars[0] + [ - ctvars[k][self.cmaj] for k in range(num_tree)]
            ##self.printLits(lhs0)
            atls = CardEnc.atleast(lits = lhs0, bound = rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(atls)
            ##
            #rhs = num_tree - 1
            rhs = num_tree + 1
            ###########
            lhs1 =  zvars[1] + [ - ctvars[k][self.cmaj] for k in range(num_tree)]
            ##self.printLits(lhs1)
            atls = CardEnc.atleast(lits = lhs1, bound = rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(atls)            
            #
            pvars = [self.newVar('p_{0}'.format(k)) for k in range(self.num_class + 1)]
            ##self.printLits(pvars)
            for k,p in enumerate(pvars):
                for i in range(num_tree):
                    if k == 0:
                        z = zvars[0][i]
                        #self.cnf.append([-p, -z, vtaut])
                        self.cnf.append([-p, z, -vtaut])       
                        #self.printLits([-p, z, -vtaut])
                        #print()
                    elif k == self.cmaj+1:
                        z = zvars[1][i]
                        self.cnf.append([-p, z, -vtaut])       
                        
                        #self.printLits([-p, z, -vtaut])
                        #print()                       
                        
                    else:
                        z = zvars[0][i] if (k<self.cmaj+1) else zvars[1][i]
                        self.cnf.append([-p, -z, ctvars[i][k-1] ])
                        self.cnf.append([-p, z, -ctvars[i][k-1] ])  
                        
                        #self.printLits([-p, -z, ctvars[i][k-1] ])
                        #self.printLits([-p, z, -ctvars[i][k-1] ])
                        #print()
                        
            #
            self.cnf.append([-pvars[0], -pvars[self.cmaj+1]])
            ##
            lhs1 =  pvars[:(self.cmaj+1)]
            ##self.printLits(lhs1)
            eqls = CardEnc.equals(lits = lhs1, bound = 1, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(eqls)
            
            
            lhs2 = pvars[(self.cmaj + 1):]
            ##self.printLits(lhs2)
            eqls = CardEnc.equals(lits = lhs2, bound = 1, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(eqls)
                
        
            
        ##print("exactly-one feat const ...")
        # enforce exactly one of the feature values to be chosen
        # (for categorical features)
        categories = collections.defaultdict(lambda: [])
        for f in self.extended_feature_names:
            if '_' in f:
                categories[f.split('_')[0]].append(self.newVar(f))        
        for c, feats in six.iteritems(categories):
            # exactly-one feat is True
            self.cnf.append(feats)
            card = CardEnc.atmost(lits=feats, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(card.clauses)
        # lits of intervals   
        for f, intvs in six.iteritems(self.ivars):
            if not len(intvs):
                continue
            self.cnf.append(intvs) 
            card = CardEnc.atmost(lits=intvs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(card.clauses)
            #self.printLits(intvs)
        
            
        
        for f, threshold in six.iteritems(self.thvars):
            for j, thvar in enumerate(threshold):
                d = j+1
                pos, neg = self.ivars[f][d:], self.ivars[f][:d] 
                
                if j == 0:
                    assert(len(neg) == 1)
                    self.cnf.append([thvar, neg[-1]])
                    self.cnf.append([-thvar, -neg[-1]])
                else:
                    self.cnf.append([thvar, neg[-1], -threshold[j-1]])
                    self.cnf.append([-thvar, threshold[j-1]])
                    self.cnf.append([-thvar, -neg[-1]])
                
                if j == len(threshold) - 1:
                    assert(len(pos) == 1)
                    self.cnf.append([-thvar, pos[0]])
                    self.cnf.append([thvar, -pos[0]])
                else:
                    self.cnf.append([-thvar, pos[0], threshold[j+1]])
                    self.cnf.append([thvar, -pos[0]])
                    self.cnf.append([thvar, -threshold[j+1]])
          


        return self.cnf, self.intvs, self.imaps, self.ivars

    ##############################
    
    def qbf_encode_rf_all_intvs(self, sample):
        """
            (deprecated)
            Encoding the original RF.
            using interval variables.
            2QBF encoding, we use ^RF to denote the original RF,
        """

        ##################################################

        def _traverse(tree, k, clause):
            """
                Traverse a tree and encode each node.
            """
            if tree.children:
                f = tree.name
                v = tree.threshold
                if f in self.intvs:
                    d = self.imaps[f][v]
                    pos, neg = self.thvars[f][d], -self.thvars[f][d]
                else:
                    assert False, "_traverse: this might be problematic"
                    var = self.newVar(tree.name + "^RF")
                    pos, neg = var, -var
                assert (pos and neg)
                _traverse(tree.children[0], k, clause + [-neg])
                _traverse(tree.children[1], k, clause + [-pos])
            else:
                # leaf node, class, tree, r-th copy
                cvar = self.newVar(f'class{tree.values}_tr{k}^RF')
                cnf_RF.append(clause + [cvar])

        def _compute_intervals():
            """
                Traverse all trees in the ensemble and extract intervals for each
                feature.
                At this point, the method only works for numerical datasets!
            """

            def traverse_intervals(tree):
                """
                    Auxiliary function. Recursive tree traversal.
                """
                if tree.children:
                    f = tree.name
                    v = tree.threshold
                    if f in self.intvs:
                        self.intvs[f].add(v)
                    traverse_intervals(tree.children[0])
                    traverse_intervals(tree.children[1])

            # initializing the intervals
            self.intvs = {'{0}'.format(f): set([]) for f in self.extended_feature_names if '_' not in f}

            for tree in self.forest.trees:
                traverse_intervals(tree)
            # OK, we got all intervals; let's sort the values
            self.intvs = {f: sorted(self.intvs[f]) + ([math.inf] if len(self.intvs[f]) else []) for f in
                          six.iterkeys(self.intvs)}

            self.imaps = {}
            self.ivars, self.thvars = {}, {}
            for feat, intvs in six.iteritems(self.intvs):
                self.imaps[feat] = {}
                self.ivars[feat] = []
                self.thvars[feat] = []
                for i, ub in enumerate(intvs):
                    self.imaps[feat][ub] = i
                    ivar = self.newVar(f'{feat}_intv{i}^RF')
                    self.ivars[feat].append(ivar)
                    if ub != math.inf:
                        thvar = self.newVar(f'{feat}_th{i}^RF')
                        self.thvars[feat].append(thvar)

        ##################################################
        cnf_RF = CNF()
        num_tree = len(self.forest.trees)
        self.forest.predict_inst(sample)

        # introducing class-tree variables
        ctvars = [[] for _ in range(num_tree)]
        ########## the original RF ##########
        for k in range(num_tree):
            for j in range(self.num_class):
                var = self.newVar(f'class{j}_tr{k}^RF')
                ctvars[k].append(var)

        _compute_intervals()

        # traversing and encoding each tree
        for k, tree in enumerate(self.forest.trees):
            # encoding the tree
            _traverse(tree, k, [])
            # exactly one class var is true
            cnf_RF.append(ctvars[k])
            card = CardEnc.atmost(lits=ctvars[k], vpool=self.vpool, encoding=EncType.cardnetwrk)
            cnf_RF.extend(card.clauses)

        # calculate the majority class
        self.cmaj = self.forest.predict_inst(sample)
        winner = self.newVar(f'winner_{self.cmaj}^RF')
        if self.num_class == 2:
            # to keep the prediction of the original RF:
            # 1) c0 => c0: if num_tree = 2n, then sum(c0) >= n;
            #               if num_tree = 2n+1, then sum(c0) >= n+1
            # 2) c1 => c1: if num_tree = 2n, then sum(c1) >= n+1;
            #               if num_tree = 2n+1, then sum(c1) >= n+1
            rhs = math.floor(num_tree / 2) + 1
            if self.cmaj == 0 and not num_tree % 2:
                rhs = math.floor(num_tree / 2)
            lhs = [ctvars[k][self.cmaj] for k in range(num_tree)]
            atls = CardEnc.atleast(lits=lhs, bound=rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            # atls is #cmaj win #other-class constraint for each other class
            # associated each clause with an aux var
            win_aux_vars = []
            for idx, item in enumerate(atls):
                aux = self.newVar(f'win_{1-self.cmaj}_aux_{idx}^RF')
                win_aux_vars.append(aux)
                new_cls = item.copy()
                cnf_RF.append(new_cls + [-aux])
                for ele in item:
                    cnf_RF.append([-ele, aux])
            # associated each constraint with an win_{other-class}
            win_other = self.newVar(f'win_{1-self.cmaj}^RF')
            cnf_RF.append([-k for k in win_aux_vars] + [win_other])
            for ele in win_aux_vars:
                cnf_RF.append([-win_other, ele])
            cnf_RF.append([-winner, win_other])
            cnf_RF.append([-win_other, winner])
        else:
            assert False, "not implemented yet"
            for i in range(self.cmaj):
                rhs = num_tree + 1
                lhs = [ctvars[k][self.cmaj] for k in range(num_tree)] + [-ctvars[k][i] for k in range(num_tree)]
                atls = CardEnc.atleast(lits=lhs, bound=rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
                cnf_RF.extend(atls)
            for i in range(self.cmaj + 1, self.num_class):
                rhs = num_tree
                lhs = [ctvars[k][self.cmaj] for k in range(num_tree)] + [-ctvars[k][i] for k in range(num_tree)]
                atls = CardEnc.atleast(lits=lhs, bound=rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
                cnf_RF.extend(atls)

        # enforce exactly one of the feature values to be chosen
        # lits of intervals
        ###### the original RF ##########
        # auxiliary variables for EqualsOne() constraints
        eq1_predicate = []
        for f, intvs in six.iteritems(self.ivars):
            if not len(intvs):
                continue
            eq_i = self.newVar(f'eq_{f}^RF')
            eq1_predicate.append(eq_i)
            eq1_tmp = [intvs]
            card = CardEnc.atmost(lits=intvs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            eq1_tmp.extend(card.clauses)
            # eq1_tmp is EqualsOne constraint for each feature
            # associated each clause with an aux var
            eq1_aux_vars = []
            for idx, item in enumerate(eq1_tmp):
                aux = self.newVar(f'eq_{f}_aux_{idx}^RF')
                eq1_aux_vars.append(aux)
                new_cls = item.copy()
                cnf_RF.append(new_cls + [-aux])
                for ele in item:
                    cnf_RF.append([-ele, aux])
            # associated each constraint with an eq_i
            cnf_RF.append([-k for k in eq1_aux_vars] + [eq_i])
            for ele in eq1_aux_vars:
                cnf_RF.append([-eq_i, ele])
        # add EqualsOne_i to threshold clauses
        for f, threshold in six.iteritems(self.thvars):
            eq_i = self.newVar(f'eq_{f}^RF')
            assert eq_i in eq1_predicate
            for j, thvar in enumerate(threshold):
                d = j + 1
                pos, neg = self.ivars[f][d:], self.ivars[f][:d]
                if j == 0:
                    assert (len(neg) == 1)
                    cnf_RF.append([thvar, neg[-1]] + [-eq_i])
                    cnf_RF.append([-thvar, -neg[-1]] + [-eq_i])
                else:
                    cnf_RF.append([thvar, neg[-1], -threshold[j - 1]] + [-eq_i])
                    cnf_RF.append([-thvar, threshold[j - 1]] + [-eq_i])
                    cnf_RF.append([-thvar, -neg[-1]] + [-eq_i])

                if j == len(threshold) - 1:
                    assert (len(pos) == 1)
                    cnf_RF.append([-thvar, pos[0]] + [-eq_i])
                    cnf_RF.append([thvar, -pos[0]] + [-eq_i])
                else:
                    cnf_RF.append([-thvar, pos[0], threshold[j + 1]] + [-eq_i])
                    cnf_RF.append([thvar, -pos[0]] + [-eq_i])
                    cnf_RF.append([thvar, -threshold[j + 1]] + [-eq_i])

        return cnf_RF, eq1_predicate
    
    def qbf_encode_rf(self, sample):
        """
            Encoding the original RF.
            using auxiliary variables to pick a feature values if
            this feature is not fixed to the given value.
            2QBF encoding, we use ^RF to denote the original RF,
        """
        ##################################################

        def _traverse(tree, k, clause):
            """
                Traverse a tree and encode each node.
            """
            if tree.children:
                f = tree.name
                v = tree.threshold
                if f in self.intvs:
                    d = self.imaps[f][v]
                    pos, neg = self.thvars[f][d], -self.thvars[f][d]
                else:
                    assert False, "_traverse: this might be problematic"
                    var = self.newVar(tree.name + "^RF")
                    pos, neg = var, -var
                assert (pos and neg)
                _traverse(tree.children[0], k, clause + [-neg])
                _traverse(tree.children[1], k, clause + [-pos])
            else:
                # leaf node, class, tree, r-th copy
                cvar = self.newVar(f'class{tree.values}_tr{k}^RF')
                cnf_RF.append(clause + [cvar])

        def _compute_intervals():
            """
                Traverse all trees in the ensemble and extract intervals for each
                feature.

                At this point, the method only works for numerical datasets!
            """
            def traverse_intervals(tree):
                """
                    Auxiliary function. Recursive tree traversal.
                """
                if tree.children:
                    f = tree.name
                    v = tree.threshold
                    if f in self.intvs:
                        self.intvs[f].add(v)
                    traverse_intervals(tree.children[0])
                    traverse_intervals(tree.children[1])
            # initializing the intervals
            self.intvs = {'{0}'.format(f): set([]) for f in self.extended_feature_names if '_' not in f}

            for tree in self.forest.trees:
                traverse_intervals(tree)
            # OK, we got all intervals; let's sort the values
            self.intvs = {f: sorted(self.intvs[f]) + ([math.inf] if len(self.intvs[f]) else []) for f in
                          six.iterkeys(self.intvs)}

            self.imaps = {}
            self.ivars, self.thvars = {}, {}
            for feat, intvs in six.iteritems(self.intvs):
                self.imaps[feat] = {}
                self.ivars[feat] = []
                self.thvars[feat] = []
                for i, ub in enumerate(intvs):
                    self.imaps[feat][ub] = i
                    ivar = self.newVar(f'{feat}_intv{i}^RF')
                    self.ivars[feat].append(ivar)
                    if ub != math.inf:
                        thvar = self.newVar(f'{feat}_th{i}^RF')
                        self.thvars[feat].append(thvar)

        ##################################################
        cnf_RF = CNF()
        num_tree = len(self.forest.trees)
        self.forest.predict_inst(sample)

        # introducing class-tree variables
        ctvars = [[] for _ in range(num_tree)]
        ########## the original RF ##########
        for k in range(num_tree):
            for j in range(self.num_class):
                var = self.newVar(f'class{j}_tr{k}^RF')
                ctvars[k].append(var)

        _compute_intervals()

        # traversing and encoding each tree
        for k, tree in enumerate(self.forest.trees):
            # encoding the tree
            _traverse(tree, k, [])
            # exactly one class var is true
            card = CardEnc.atmost(lits=ctvars[k], vpool=self.vpool, encoding=EncType.cardnetwrk)
            cnf_RF.extend(card.clauses)

        # calculate the majority class
        self.cmaj = self.forest.predict_inst(sample)
        if self.num_class == 2:
            # to keep the prediction of the original RF:
            # 1) c0 => c0: if num_tree = 2n, then sum(c0) >= n;
            #               if num_tree = 2n+1, then sum(c0) >= n+1
            # 2) c1 => c1: if num_tree = 2n, then sum(c1) >= n+1;
            #               if num_tree = 2n+1, then sum(c1) >= n+1
            rhs = math.floor(num_tree / 2) + 1
            if self.cmaj == 0 and not num_tree % 2:
                rhs = math.floor(num_tree / 2)
            lhs = [ctvars[k][self.cmaj] for k in range(num_tree)]
            atls = CardEnc.atleast(lits=lhs, bound=rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            cnf_RF.extend(atls)
        else:
            for i in range(self.cmaj):
                rhs = num_tree+1
                lhs = [ctvars[k][self.cmaj] for k in range(num_tree)] + [-ctvars[k][i] for k in range(num_tree)]
                atls = CardEnc.atleast(lits=lhs, bound=rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
                cnf_RF.extend(atls)
            for i in range(self.cmaj+1, self.num_class):
                rhs = num_tree
                lhs = [ctvars[k][self.cmaj] for k in range(num_tree)] + [-ctvars[k][i] for k in range(num_tree)]
                atls = CardEnc.atleast(lits=lhs, bound=rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
                cnf_RF.extend(atls)

        # enforce exactly one of the feature values to be chosen
        # lits of intervals
        ###### the original RF ##########
        for f, intvs in six.iteritems(self.ivars):
            if not len(intvs):
                continue
            cnf_RF.append(intvs)
            card = CardEnc.atmost(lits=intvs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            cnf_RF.extend(card.clauses)
        for f, threshold in six.iteritems(self.thvars):
            for j, thvar in enumerate(threshold):
                d = j + 1
                pos, neg = self.ivars[f][d:], self.ivars[f][:d]
                if j == 0:
                    assert (len(neg) == 1)
                    cnf_RF.append([thvar, neg[-1]])
                    cnf_RF.append([-thvar, -neg[-1]])
                else:
                    cnf_RF.append([thvar, neg[-1], -threshold[j - 1]])
                    cnf_RF.append([-thvar, threshold[j - 1]])
                    cnf_RF.append([-thvar, -neg[-1]])

                if j == len(threshold) - 1:
                    assert (len(pos) == 1)
                    cnf_RF.append([-thvar, pos[0]])
                    cnf_RF.append([thvar, -pos[0]])
                else:
                    cnf_RF.append([-thvar, pos[0], threshold[j + 1]])
                    cnf_RF.append([thvar, -pos[0]])
                    cnf_RF.append([thvar, -threshold[j + 1]])
        return cnf_RF

    def qbf_encode_t(self, sample):
        """
            2QBF encoding, we use ^RF to denote the original RF,
        """
        # for t-th copy
        self.ivars_t, self.thvars_t = {}, {}
        ##################################################

        def _traverse(tree, k, clause):
            """
                Traverse a tree and encode each node.
            """
            if tree.children:
                f = tree.name
                v = tree.threshold
                if f in self.intvs:
                    d = self.imaps[f][v]
                    pos, neg = self.thvars_t[f][d], -self.thvars_t[f][d]
                else:
                    assert False, "_traverse: this might be problematic"
                    var = self.newVar(tree.name)
                    pos, neg = var, -var
                assert (pos and neg)
                _traverse(tree.children[0], k, clause + [-neg])
                _traverse(tree.children[1], k, clause + [-pos])
            else:
                # leaf node, class, tree, r-th copy
                cvar = self.newVar(f'class{tree.values}_tr{k}')
                cnf_t.append(clause + [cvar])

        def _compute_intervals():
            """
                Traverse all trees in the ensemble and extract intervals for each
                feature.

                At this point, the method only works for numerical datasets!
            """
            assert len(self.intvs)
            assert len(self.imaps)
            for feat, intvs in six.iteritems(self.intvs):
                # assert len(self.imaps[feat])  intvs=[] is possible
                self.ivars_t[feat] = []
                self.thvars_t[feat] = []
                for i, ub in enumerate(intvs):
                    assert self.imaps[feat][ub] == i
                    assert len(self.ivars[feat])
                    ivar_t = self.newVar(f'{feat}_intv{i}')
                    self.ivars_t[feat].append(ivar_t)
                    if ub != math.inf:
                        assert len(self.thvars[feat])
                        thvar_t = self.newVar(f'{feat}_th{i}')
                        self.thvars_t[feat].append(thvar_t)

        ##################################################
        cnf_t = CNF()
        num_tree = len(self.forest.trees)
        self.forest.predict_inst(sample)
        # define Tautology var
        vtaut = self.newVar('Tautology')
        cnf_t.append([vtaut])

        # introducing class-tree variables
        ctvars_t = [[] for _ in range(num_tree)]
        ########## t-th copy of RF ##########
        for k in range(num_tree):
            for j in range(self.num_class):
                var = self.newVar(f'class{j}_tr{k}')
                ctvars_t[k].append(var)

        _compute_intervals()

        for k, tree in enumerate(self.forest.trees):
            # encoding the tree
            _traverse(tree, k, [])
            # exactly one class var is true
            card = CardEnc.atmost(lits=ctvars_t[k], vpool=self.vpool, encoding=EncType.cardnetwrk)
            cnf_t.extend(card.clauses)

        # calculate the majority class
        self.cmaj = self.forest.predict_inst(sample)

        if self.num_class == 2:
            # to change the prediction of t-th copy of RF:
            # 1) c0 => c1: if num_tree = 2n, then sum(c1) >= n+1;
            #               if num_tree = 2n+1, then sum(c1) >= n+1.
            # 2) c1 => c0: if num_tree = 2n, then sum(c0) >= n;
            #               if num_tree = 2n+1, then sum(c0) >= n+1
            rhs = math.floor(num_tree / 2) + 1
            if self.cmaj == 1 and not num_tree % 2:
                rhs = math.floor(num_tree / 2)
            lhs = [ctvars_t[k][1 - self.cmaj] for k in range(num_tree)]
            atls = CardEnc.atleast(lits=lhs, bound=rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            cnf_t.extend(atls)
        else:
            zvars = []
            zvars.append([self.newVar('z_0_{0}'.format(k)) for k in range(num_tree)])
            zvars.append([self.newVar('z_1_{0}'.format(k)) for k in range(num_tree)])
            rhs = num_tree
            lhs0 = zvars[0] + [- ctvars_t[k][self.cmaj] for k in range(num_tree)]
            atls = CardEnc.atleast(lits=lhs0, bound=rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            cnf_t.extend(atls)
            rhs = num_tree + 1
            lhs1 = zvars[1] + [- ctvars_t[k][self.cmaj] for k in range(num_tree)]
            atls = CardEnc.atleast(lits=lhs1, bound=rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            cnf_t.extend(atls)
            pvars = [self.newVar('p_{0}'.format(k)) for k in range(self.num_class + 1)]
            for k, p in enumerate(pvars):
                for i in range(num_tree):
                    if k == 0:
                        z = zvars[0][i]
                        cnf_t.append([-p, z, -vtaut])
                    elif k == self.cmaj + 1:
                        z = zvars[1][i]
                        cnf_t.append([-p, z, -vtaut])
                    else:
                        z = zvars[0][i] if (k < self.cmaj + 1) else zvars[1][i]
                        cnf_t.append([-p, -z, ctvars_t[i][k - 1]])
                        cnf_t.append([-p, z, -ctvars_t[i][k - 1]])
            cnf_t.append([-pvars[0], -pvars[self.cmaj + 1]])
            lhs1 = pvars[:(self.cmaj + 1)]
            eqls = CardEnc.equals(lits=lhs1, bound=1, vpool=self.vpool, encoding=EncType.cardnetwrk)
            cnf_t.extend(eqls)

            lhs2 = pvars[(self.cmaj + 1):]
            eqls = CardEnc.equals(lits=lhs2, bound=1, vpool=self.vpool, encoding=EncType.cardnetwrk)
            cnf_t.extend(eqls)

        # enforce exactly one of the feature values to be chosen
        # (for categorical features)
        categories = collections.defaultdict(lambda: [])
        for f in self.extended_feature_names:
            if '_' in f:
                assert False, "qbf_encode: categorical not considered yet?"
                categories[f.split('_')[0]].append(self.newVar(f))
        for c, feats in six.iteritems(categories):
            # exactly-one feat is True
            cnf_t.append(feats)
            card = CardEnc.atmost(lits=feats, vpool=self.vpool, encoding=EncType.cardnetwrk)
            cnf_t.extend(card.clauses)
        # lits of intervals
        ########## t-th copy of RF ##########
        for f, intvs in six.iteritems(self.ivars_t):
            if not len(intvs):
                continue
            cnf_t.append(intvs)
            card = CardEnc.atmost(lits=intvs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            cnf_t.extend(card.clauses)
        for f, threshold in six.iteritems(self.thvars_t):
            for j, thvar in enumerate(threshold):
                d = j + 1
                pos, neg = self.ivars_t[f][d:], self.ivars_t[f][:d]
                if j == 0:
                    assert (len(neg) == 1)
                    cnf_t.append([thvar, neg[-1]])
                    cnf_t.append([-thvar, -neg[-1]])
                else:
                    cnf_t.append([thvar, neg[-1], -threshold[j - 1]])
                    cnf_t.append([-thvar, threshold[j - 1]])
                    cnf_t.append([-thvar, -neg[-1]])

                if j == len(threshold) - 1:
                    assert (len(pos) == 1)
                    cnf_t.append([-thvar, pos[0]])
                    cnf_t.append([thvar, -pos[0]])
                else:
                    cnf_t.append([-thvar, pos[0], threshold[j + 1]])
                    cnf_t.append([thvar, -pos[0]])
                    cnf_t.append([thvar, -threshold[j + 1]])
        return cnf_t

    ##############################

#
#==============================================================================
class SATExplainer(object):
    """
        An SAT-inspired minimal explanation extractor for Random Forest models.
    """

    def __init__(self, sat_enc, inps, preamble, target_name, verb=1):
        """
            Constructor.
        """
        self.enc = sat_enc
        self.inps = inps  # input (feature value) variables
        self.target_name = target_name
        self.preamble = preamble
        self.verbose = verb
        self.slv = None    
      
    def prepare_selectors(self, sample):
        # adapt the solver to deal with the current sample
        #self.csel = []
        self.assums = []  # var selectors to be used as assumptions
        self.sel2fid = {}  # selectors to original feature ids
        self.sel2vid = {}  # selectors to categorical feature ids
        self.sel2v = {} # selectors to (categorical/interval) values
        
        #for i in range(self.enc.num_class):
        #    self.csel.append(self.enc.newVar('class{0}'.format(i)))
        #self.csel = self.enc.newVar('class{0}'.format(self.enc.cmaj))
               
        # preparing the selectors
        for i, (inp, val) in enumerate(zip(self.inps, sample), 1):
            if '_' in inp:
                # binarized (OHE) features
                assert (inp not in self.enc.intvs)
                
                feat = inp.split('_')[0]
                selv = self.enc.newVar('selv_{0}'.format(feat))
            
                self.assums.append(selv)   
                if selv not in self.sel2fid:
                    self.sel2fid[selv] = int(feat[1:])
                    self.sel2vid[selv] = [i - 1]
                else:
                    self.sel2vid[selv].append(i - 1)
                    
                p = self.enc.newVar(inp) 
                if not val:
                    p = -p
                else:
                    self.sel2v[selv] = p
                    
                self.enc.cnf.append([-selv, p])
                #self.enc.printLits([-selv, p])
                    
            elif len(self.enc.intvs[inp]):
                #v = None
                #for intv in self.enc.intvs[inp]:
                #    if intv > val:
                #        v = intv
                #        break         
                v = next((intv for intv in self.enc.intvs[inp] if intv > val), None)     
                assert(v is not None)
                
                selv = self.enc.newVar('selv_{0}'.format(inp))
                self.assums.append(selv)  
                
                assert (selv not in self.sel2fid)
                self.sel2fid[selv] = int(inp[1:])
                self.sel2vid[selv] = [i - 1]
                            
                for j,p in enumerate(self.enc.ivars[inp]):
                    cl = [-selv]
                    if j == self.enc.imaps[inp][v]:
                        cl += [p]
                        self.sel2v[selv] = p
                    else:
                        cl += [-p]
                    
                    self.enc.cnf.append(cl)
                    #self.enc.printLits(cl)


    
    def explain(self, sample, xtype='abd', smallest=False):
        """
            Hypotheses minimization.
        """
        if self.verbose:
            print('  explaining:  "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.target_name[self.enc.cmaj]))
                    
        
        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime
        
        self.prepare_selectors(sample)
        
        if xtype == 'abd':
            # abductive (PI-) explanation
            expl = self.compute_axp() 
        else:
            # contrastive explanation
            expl = self.compute_cxp()
 
        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
    
        # delete sat solver
        self.slv.delete()
        self.slv = None
        
        if self.verbose:
            print('  time: {0:.3f}'.format(self.time))

        return expl

    def compute_axp(self, smallest=False):
        """
            Compute an Abductive eXplanation
        """         
        self.assums = sorted(set(self.assums))
        if self.verbose:
            print('  # hypos:', len(self.assums))
        
        #create a SAT solver
        self.slv = Solver(name="glucose3")
        
        # pass a CNF formula
        self.slv.append_formula(self.enc.cnf)    

        def minimal():
            vtaut = self.enc.newVar('Tautology')
            # simple deletion-based linear search
            for i, p in enumerate(self.assums):
                to_test = [vtaut] + self.assums[:i] + self.assums[(i + 1):] + [-p, -self.sel2v[p]]
                sat = self.slv.solve(assumptions=to_test)
                if not sat:
                    self.assums[i] = -p
            return
        
        if not smallest:
            minimal()
        else:
            raise NotImplementedError('Smallest explanation is not yet implemented.')
            #self.compute_smallest()

        expl = sorted([self.sel2fid[h] for h in self.assums if h>0 ])
        assert len(expl), 'Abductive explanation cannot be an empty-set! otherwise RF fcn is const, i.e. predicts only one class'
        
        if self.verbose:
            print("expl-selctors: ", expl)
            preamble = [self.preamble[i] for i in expl]
            print('  explanation: "IF {0} THEN {1}"'.format(' AND '.join(preamble), self.target_name[self.enc.cmaj]))
            print('  # hypos left:', len(expl))

        return expl
        
    def compute_cxp(self, smallest=True):
        """
            Compute a Contrastive eXplanation
        """         
        self.assums = sorted(set(self.assums))
        if self.verbose:
            print('  # hypos:', len(self.assums))   
    
        wcnf = WCNF()
        for cl in self.enc.cnf:
            wcnf.append(cl)    
        for p in self.assums:
            wcnf.append([p], weight=1)
            
        if not smallest:
            # mcs solver
            self.slv = LBX(wcnf, use_cld=True, solver_name='g3')
            mcs = self.slv.compute()
            expl = sorted([self.sel2fid[self.assums[i-1]] for i in mcs])
        else:
            # mxsat solver
            self.slv = RC2(wcnf)
            model = self.slv.compute()
            model = [p for p in model if abs(p) in self.assums]            
            expl = sorted([self.sel2fid[-p] for p in model if p<0 ])
       
        assert len(expl), 'Contrastive explanation cannot be an empty-set!'         
        if self.verbose:
            print("expl-selctors: ", expl)
            preamble = [self.preamble[i] for i in expl]
            pred = self.target_name[self.enc.cmaj]
            print(f'  explanation: "IF {" AND ".join([f"!({p})" for p in preamble])} THEN !(class = {pred})"')
            
        return expl    
    
    def enumerate(self, sample, xtype='con', smallest=True):
        """
            list all CXp's or AXp's
        """
        if xtype == 'abd':
            raise NotImplementedError('Enumerate abductive explanations is not yet implemented.')
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime
        
        if 'assums' not in dir(self):
            self.prepare_selectors(sample)
            self.assums = sorted(set(self.assums))
            #
            
        # compute CXp's/AE's    
        if self.slv is None:    
            wcnf = WCNF()
            for cl in self.enc.cnf:
                wcnf.append(cl)    
            for p in self.assums:
                wcnf.append([p], weight=1)
            if smallest:    
                # incremental maxsat solver    
                self.slv = RC2(wcnf, adapt=True, exhaust=True, minz=True)
            else:
                # mcs solver
                self.slv = LBX(wcnf, use_cld=True, solver_name='g3')
                #self.slv = MCSls(wcnf, use_cld=True, solver_name='g3')                
                
        if smallest:    
            print('smallest')
            for model in self.slv.enumerate(block=-1):
                #model = [p for p in model if abs(p) in self.assums]
                expl = sorted([self.sel2fid[-p] for p in model if (p<0 and (-p in self.assums))])
                cxp_feats = [f'f{j}' for j in expl]
                advx = []
                for f in cxp_feats:
                    ps = [p for p in model if (p>0 and (p in self.enc.ivars[f]))]
                    assert(len(ps) == 1)
                    advx.append(tuple([f,self.enc.nameVar(ps[0])]))   
                #yield expl
                print(cxp_feats, advx)
                yield advx
        else:
            print('LBX')
            for mcs in self.slv.enumerate():
                expl = sorted([self.sel2fid[self.assums[i-1]] for i in mcs])
                assumptions = [-p if(i in mcs) else p for i,p in enumerate(self.assums, 1)]
                #for k, model in enumerate(self.slv.oracle.enum_models(assumptions), 1):
                assert (self.slv.oracle.solve(assumptions))
                model = self.slv.oracle.get_model()
                cxp_feats = [f'f{j}' for j in expl]
                advx = []
                for f in cxp_feats:
                    ps = [p for p in model if (p>0 and (p in self.enc.ivars[f]))]
                    assert(len(ps) == 1)
                    advx.append(tuple([f,self.enc.nameVar(ps[0])]))
                yield advx
                self.slv.block(mcs)
                #yield expl
                
                
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - time 
        if self.verbose:
            print('c expl time: {0:.3f}'.format(time))
        #
        self.slv.delete()
        self.slv = None

    ##############################

    def is_axp(self, axp):
        """
            Checking if a given explanation is an AXp.
            :param axp: given AXp
        """
        assert len(self.assums)
        self.assums = sorted(set(self.assums))
        assums = self.assums[:]
        ########## preprocessing ##########
        for i, slt in enumerate(assums):
            feat = self.sel2fid[slt]
            if feat not in axp:
                assums[i] = -assums[i]
        for i, slt in enumerate(assums):
            feat = self.sel2fid[abs(slt)]
            if feat in axp:
                assert assums[i] > 0
            else:
                assert assums[i] < 0
        ########## preprocessing ##########
        self.slv = Solver(name="glucose3")
        # pass a CNF formula
        self.slv.append_formula(self.enc.cnf)
        vtaut = self.enc.newVar('Tautology')
        # if the CNF encoding is UNSAT then we got a weak AXp,
        # otherwise not a weak AXp
        if self.slv.solve(assumptions=[vtaut] + assums):
            print("is_axp: not a weak AXp")
            self.slv.delete()
            self.slv = None
            return False
        # 2) no feature can be removed
        # for i, slt in enumerate(assums):
        #     if slt > 0:
        #         assums[i] = -slt
        #         # if the encoding is still UNSAT after we free a feature,
        #         # then the given weak AXp is not subset-minimal.
        #         if not self.slv.solve(assumptions=[vtaut] + assums):
        #             print("is_axp: not subset-minimal weak AXp")
        #             self.slv.delete()
        #             self.slv = None
        #             return False
        #
        # self.slv.delete()
        # self.slv = None
        # return True
        self.slv.add_clause([-p for p in assums if p > 0])
        sat = self.slv.solve(assumptions=[vtaut] + [p for p in assums if p < 0])
        self.slv.delete()
        self.slv = None
        if not sat:
            print("is_axp: not subset-minimal weak AXp")
        return sat

    def is_waxp(self, waxp):
        """
            Checking if a given explanation is a weak AXp.
            :param sample: given instance/sample
            :param waxp: given weak AXp
        """
        assert len(self.assums)
        self.assums = sorted(set(self.assums))
        assums = self.assums[:]
        ########## preprocessing ##########
        for i, slt in enumerate(assums):
            feat = self.sel2fid[slt]
            if feat not in waxp:
                assums[i] = -assums[i]
        for i, slt in enumerate(assums):
            feat = self.sel2fid[abs(slt)]
            if feat in waxp:
                assert assums[i] > 0
            else:
                assert assums[i] < 0
        ########## preprocessing ##########
        vtaut = self.enc.newVar('Tautology')
        # if the CNF encoding is UNSAT then we got a weak AXp,
        # otherwise not a weak AXp
        return not self.slv.solve(assumptions=[vtaut] + assums)

    def extract_axp(self, waxp):
        """
            Extracting an AXp from given weak AXp.
            :param sample: given instance/sample
            :param waxp: given weak AXp
        """
        assert len(self.assums)
        self.assums = sorted(set(self.assums))
        assums = self.assums[:]
        ########## preprocessing ##########
        for i, slt in enumerate(assums):
            feat = self.sel2fid[slt]
            if feat not in waxp:
                assums[i] = -assums[i]
        for i, slt in enumerate(assums):
            feat = self.sel2fid[abs(slt)]
            if feat in waxp:
                assert assums[i] > 0
            else:
                assert assums[i] < 0
        ########## preprocessing ##########
        self.slv = Solver(name="glucose3")
        # pass a CNF formula
        self.slv.append_formula(self.enc.cnf)
        vtaut = self.enc.newVar('Tautology')
        # it is a weak axp
        assert not self.slv.solve(assumptions=[vtaut] + assums)
        # remove redundant feature
        for i, slt in enumerate(assums):
            if slt > 0:
                assums[i] = -slt
                if self.slv.solve(assumptions=[vtaut] + assums):
                    # it is not a weak axp when slt if free, so make it fix
                    assums[i] = slt
        axp = sorted([self.sel2fid[h] for h in assums if h > 0])

        self.slv.delete()
        self.slv = None
        return axp

    def new_pos_claus(self, drop):
        """
            Minimising drop set.
            :param sample: given instance/sample
            :param drop: given drop set
        """
        assert len(self.assums)
        self.assums = sorted(set(self.assums))
        assums = self.assums[:]
        ########## preprocessing ##########
        for i, slt in enumerate(assums):
            feat = self.sel2fid[slt]
            if feat in drop:
                assums[i] = -assums[i]
        for i, slt in enumerate(assums):
            feat = self.sel2fid[abs(slt)]
            if feat in drop:
                assert assums[i] < 0
            else:
                assert assums[i] > 0
        ########## preprocessing ##########
        vtaut = self.enc.newVar('Tautology')
        # it should represent a weak CXp
        assert self.slv.solve(assumptions=[vtaut] + assums)
        # minimize drop features with specified step
        step = int(len(drop) / 2)
        for i, slt in enumerate(assums):
            if slt < 0 and step > 0:
                # fix free feature
                assums[i] = -slt
                if not self.slv.solve(assumptions=[vtaut] + assums):
                    # it is not a weak cxp when slt is fix, so make it free
                    assums[i] = slt
                step -= 1

        new_drop = sorted([self.sel2fid[abs(h)] for h in assums if h < 0])
        assert 0 < len(new_drop) <= len(drop)

        return new_drop

    def new_neg_claus(self, pick):
        """
            Minimising pick set.
            :param sample: given instance/sample
            :param pick: given pick set
        """
        assert len(self.assums)
        self.assums = sorted(set(self.assums))
        assums = self.assums[:]
        ########## preprocessing ##########
        for i, slt in enumerate(assums):
            feat = self.sel2fid[slt]
            if feat not in pick:
                assums[i] = -assums[i]
        for i, slt in enumerate(assums):
            feat = self.sel2fid[abs(slt)]
            if feat in pick:
                assert assums[i] > 0
            else:
                assert assums[i] < 0
        ########## preprocessing ##########
        vtaut = self.enc.newVar('Tautology')
        # it is a weak axp
        assert not self.slv.solve(assumptions=[vtaut] + assums)
        # minimize fixed features with specified step
        step = int(len(pick) / 2)
        for i, slt in enumerate(assums):
            if slt > 0 and step > 0:
                assums[i] = -slt
                if self.slv.solve(assumptions=[vtaut] + assums):
                    # it is not a weak axp when slt if free, so make it fix
                    assums[i] = slt
                step -= 1

        new_pick = sorted([self.sel2fid[h] for h in assums if h > 0])
        assert 0 < len(new_pick) <= len(pick)

        return new_pick

    def fmp_cegar(self, feat_t):
        """
            A CEGAR based-approach for deciding feature membership problem (FMP)
        """
        self.slv = Solver(name="glucose3")
        # pass a CNF formula
        self.slv.append_formula(self.enc.cnf)
        #########################################
        vpool = IDPool()

        def new_var(name):
            """
                Inner function,
                Find or new a PySAT variable.
                See PySat.

                :param name: name of variable
                :return: index of variable
            """
            return vpool.id(f'{name}')

        #########################################

        nv = len(self.inps)
        slts = [new_var(f's_{i}') for i in range(nv)]
        # some selectors are free/drop from the very beginning
        no_appear_fid = []
        no_appear_cls = []
        for i, inp in enumerate(self.inps):
            assert i == int(inp[1:])
            if len(self.enc.intvs[inp]) == 0:
                assert self.enc.newVar(f'selv_{inp}') not in self.assums
                if i == feat_t:
                    return [], 0, 0, 0
                no_appear_fid.append(i)
                no_appear_cls.append([-slts[i]])
            else:
                assert self.enc.newVar(f'selv_{inp}') in self.assums
        waxp = []
        # solver for answering feature membership query
        fmp_slv = Solver(name="Glucose4")
        slv_calls = 0
        if len(no_appear_cls):
            fmp_slv.append_formula(no_appear_cls)
        while fmp_slv.solve(assumptions=[slts[feat_t]]):
            slv_calls += 1
            ##############################
            pick = []
            model = fmp_slv.get_model()
            assert model
            for lit in model:
                name = vpool.obj(abs(lit)).split(sep='_')
                if name[0] == 's':
                    if lit > 0 and int(name[1]) != feat_t:
                        pick.append(int(name[1]))
            assert feat_t not in pick
            drop = []
            for i in range(nv):
                if i == feat_t or i in no_appear_fid:
                    # skip target feature and those already free
                    continue
                if i not in pick:
                    drop.append(i)
            assert feat_t not in drop
            ##############################
            if self.is_waxp(pick + [feat_t]):
                # weak axp candidate
                if self.is_waxp(pick):
                    # pick is itself a weak axp, reduce pick set and block it
                    core = self.slv.get_core()
                    new_pick = sorted([self.sel2fid[h] for h in core if h in self.assums and h > 0])
                    assert 0 < len(new_pick) <= len(pick)
                    # new_pick = self.new_neg_claus(pick)
                    fmp_slv.add_clause([-slts[i] for i in new_pick])
                else:
                    # waxp is a weak axp
                    waxp = pick + [feat_t]
                    break
            else:
                # pick + feat_t is not a weak axp, add more features
                new_drop = self.new_pos_claus(drop)
                ct_examp = [slts[i] for i in new_drop]
                assert ct_examp != []
                fmp_slv.add_clause(ct_examp)

        self.slv.delete()
        self.slv = None
        # return number of variables/clauses used in CEGAR loop
        cegar_nvars = fmp_slv.nof_vars()
        cegar_ncls = fmp_slv.nof_clauses()
        fmp_slv.delete()
        waxp.sort()
        return waxp, slv_calls, cegar_nvars, cegar_ncls

    ##############################