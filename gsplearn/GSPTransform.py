#!/usr/bin/python
# -*- coding: latin-1 -*-

__author__ = 'mathildemenoret'
__instit__ = 'imt_atlantique'
__funding__ = 'cominlabs-neuralcommunication'

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.neighbors import radius_neighbors_graph
from pygsp import graphs, operators
from nilearn.connectome import ConnectivityMeasure

def sparsify(matrix,k) :           
    sort_idx = np.argsort(np.abs(matrix),None)
    idx=sort_idx[:int(sort_idx.size*k)]# get indexes of sort order (high to low)
    matrix.put(idx, 0)
    return matrix
      
def create_graph(rest,kind,method,coords,spars,geo_alpha):            
        
    if 'geometric' in kind:
        if spars==0.:
            radius=1000.
            mode='distance'
        else: 
            radius=spars
            if 'binary' in method:
                mode = 'connectivity'              
            elif 'distance' in method:
                mode= 'distance'
        A = radius_neighbors_graph(coords, radius, mode=mode, include_self=False)
        if 'binary' in method:
            GW=A.toarray()
        else:        
            W=A.toarray()
            GW=W.copy()
            for i in range(W.shape[0]):
                for j in range(W.shape[0]):
                    if W[i,j]!=0:
                        GW[i,j]=np.exp(-geo_alpha*(W[i,j]**2))
                    else:
                        GW[i,j]=0

    else:
        if 'func' in kind:
            radius=1000.
            mode='connectivity'
            thr=spars            
        elif 'mixed' in kind:
            if spars!=0.:
                radius=spars
                mode ='connectivity'
            else:
               # tg=True
                mode='distance'
                radius=1000. 
        A = radius_neighbors_graph(coords, radius, mode=mode, include_self=False)
        W=A.toarray()
        if 'correlation' in method:
            correlation_measure = ConnectivityMeasure(kind='correlation')
        elif 'covariance' in method:
            correlation_measure = ConnectivityMeasure(kind='covariance')
        correlation_matrix = correlation_measure.fit_transform([rest])[0]
        if 'thr' in locals():
            if 0.< thr < 1.:
                correlation_matrix = sparsify(correlation_matrix,spars)
#        if 'tg' in globals():
#            nW=W/(W.mean()*2)
#            GW=W.copy()
#            for i in range(W.shape[0]):
#                for j in range(W.shape[0]):
#                    if W[i,j]!=0:
#                        GW[i,j]=np.exp(-(1-correlation_matrix[i,j])**100/0.2)*np.exp(-10*(nW[i,j]**1))
#                    else:
#                        GW[i,j]=0
#        else:    
        GW=W.copy()
        for i in range(W.shape[0]):
            for j in range(W.shape[0]):
                if W[i,j]!=0:
                    GW[i,j]=correlation_matrix[i,j]#np.abs(correlation_matrix[i,j])
                else:
                    GW[i,j]=0
            

    G=graphs.Graph(W=GW,coords=coords)#
    G.compute_fourier_basis()
     
    return G

class GraphTransformer(BaseEstimator, TransformerMixin):
    """ A class that compute different kind of graphs and tranform data
    
    Parameters
    ----------
    kind : str {"geometric", "functional", "mixed"}
     
    rest : array-like = [n_samples, n_features]
           The data used to build functional or mixed graph
    
    coords: numpy array [n_feature, n_dim]
            Coordinates (1D,2D,3D) - same n_features as rest and X
            For geometric or mixed graph  
      
    method : str{"covariance","correlation","distance","binary"}
        Methods for functionnal & mixed (covariance, correlation...)
        Methods for geometric (binary or distance)
        
  
    spars : float 
        Threshold sparsity: 0: Full (Geometric)
                            Between 0 and 1: sparse (Functional)
                            Float (Geometric and Mixed): Radius of neighbourood
        
    geo_alpha: float
        Parameter for distance gaussian kernel (default 0.0001)
        For geometric or mixed graph
    
    
        
    Attributes
    ----------
    input_shape : tuple
        The shape the data passed to :meth:`fit`
        
    G : Graph (structure as PyGSP)
    """
    
    def __init__(self, rest=[0,0], coords=[0], kind='geometric',
                 method='distance',spars=0, geo_alpha=0.0001):
        self.kind = kind
        self.method=method
        self.spars=spars
        self.geo_alpha=geo_alpha
        self.rest=rest
        self.coords=coords
        
        
    def fit(self, X, y=None):
        """Create Graph.
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
          
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        
        self.G = create_graph(self.rest,self.kind,self.method,self.coords,self.spars,self.geo_alpha)
        # Return the transformer
        return self

    def transform(self, X):
        """ Transform input data as Graph Fourrier transform.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The array containing the element-wise square roots of the values
            in `X`
        """
        # Check is fit had been called
        check_is_fitted(self, ['G'])
        # Input validation
        X = check_array(X)
        # Check that the input is of the same shape as the one passed
        # during fit.
#        if X.shape != self.input_shape_:
#            raise ValueError('Shape of input is different from what was seen'
#                             'in `fit`')
        X_hat=operators.gft(self.G, X.T)        
        return X_hat.T
        
        
        
    def inverse_transform(self, X_hat, copy=None):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        check_is_fitted(self, 'scale_')

        # Input validation
        X_hat = check_array(X_hat)
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X_hat.shape != self.input_shape_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        X=operators.igft(self.G, X_hat.T)        
        return X.T
        

