from sklearn.ensemble._voting import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import sys
import os
import resource

import collections
from itertools import combinations
from six.moves import range
import six
import math



#
#==============================================================================
class VotingRF(VotingClassifier):
    """
        Majority rule classifier
    """
    
    def fit(self, X, y, sample_weight=None):
        self.estimators_ = []
        for _, est in self.estimators:
            self.estimators_.append(est)
            
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_   
        
            
    def predict(self, X):
        """Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        maj : array-like of shape (n_samples,)
            Predicted class labels.
        """
        #check_is_fitted(self)
        
        # 'hard' voting
        predictions = self._predict(X)
        predictions =  np.asarray(predictions, np.int64) #NEED TO BE CHECKED
        maj = np.apply_along_axis(
            lambda x: np.argmax(
                np.bincount(x, weights=self._weights_not_none)),
            axis=1, arr=predictions)
   
        maj = self.le_.inverse_transform(maj)

        return maj
    
        
#
#==============================================================================
class RF2001(object):
    """
        The main class to train Random Forest Classifier (RFC).
    """

    def __init__(self, **options):
        """
            Constructor.
        """    
        self.forest = None
        self.voting = None
              
        param_dist = {'n_estimators':options['n_trees'],
                      'max_depth':options['depth'],
                      'criterion':'entropy',
                      'random_state':324089}
        
        self.forest = RandomForestClassifier(**param_dist)
        
    def fit(self, X_train, y_train):
        """
            building Breiman'01 Random Forest 
            (similar to train(dataset) fnc) 
        """
        self.forest.fit(X_train,y_train)
        rtrees = [ ('dt', dt) for i, dt in enumerate(self.forest.estimators_)]
        self.voting = VotingRF(estimators=rtrees)
        self.voting.fit(X_train,y_train)
        
        return self
        
        
    def train(self, dataset, verb=0):
        """
            Train a random forest.
        """
        
        X_train, X_test, y_train, y_test = dataset.train_test_split()
            
        X_train = dataset.transform(X_train)
        X_test = dataset.transform(X_test)
        
        print("Build a random forest.")
        self.forest.fit(X_train,y_train)
        
        rtrees = [ ('dt', dt) for i, dt in enumerate(self.forest.estimators_)]
        self.voting = VotingRF(estimators=rtrees)
        self.voting.fit(X_train,y_train)
        
        train_acc = accuracy_score(self.predict(X_train), y_train)
        test_acc = accuracy_score(self.predict(X_test), y_test)

        if verb > 1:
            self.print_acc_vote(X_train, X_test, y_train, y_test)
            self.print_acc_prob(X_train, X_test, y_train, y_test)
        
        return train_acc, test_acc
    
    def predict(self, X):
        return self.voting.predict(X)
    
    def predict_prob(self, X):
        self.forest.predict(X)
        
    def estimators(self):
        assert(self.forest.estimators_ is not None)
        return self.forest.estimators_
        
    def n_estimators(self):
        return self.forest.n_estimators
    
    def print_accuracy(self, X_test, y_test):  
        test_acc = accuracy_score(self.predict(X_test), y_test)
        print("c Model accuracy: {0:.2f}".format(100. * test_acc))
        #print("----------------------")  