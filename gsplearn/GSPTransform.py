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
import scipy.io as sio

def sparsify(matrix,k) :           
    sort_idx = np.argsort(np.abs(matrix),None)
    idx=sort_idx[:int(sort_idx.size*k)]# get indexes of sort order (high to low)
    matrix.put(idx, 0)
    return matrix

def f(a,N):
    return np.argsort(a)[::-1][:N]
      
def create_graph(rest,kind,method,coords,spars,geo_alpha,verbose=0,w_name=None,absolute=False):            
    tmp = radius_neighbors_graph(coords, 10000., mode='distance', include_self=False).toarray()  
    if 'geometric' in kind:
        if spars==1:
            radius=10000.
            mode='distance'
        else: 
            if 0.< spars < 1.:
                radius=spars*tmp.max()
            else: radius=spars
            if verbose>0:
                print('radius:',radius)
            
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
            if verbose>0:
                print('mean',np.mean(GW[np.nonzero(GW)]),'min',np.min(GW[np.nonzero(GW)]),'max',np.max(GW))
    elif 'kalofolias' in kind:
        GW=sio.loadmat(w_name)['W']
    else:
        if 'func' in kind:
            radius=1000.
            mode='connectivity'
            thr=spars            
        elif 'mixed' in kind:
            if spars!=0.:
                radius=spars*tmp.max()
                if verbose>0:
                    print('radius:',radius)
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
        if absolute==True:
            correlation_matrix =np.abs(correlation_matrix)
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
    kind : str {"geometric", "functional", "mixed","kalofolias"}
     
    rest : array-like = [n_samples, n_features]
           The data used to build functional or mixed graph
    
    coords: numpy array [n_feature, n_dim]
            Coordinates (1D,2D,3D) - same n_features as rest and X
            For geometric or mixed graph  
      
    method : str{"covariance","correlation","distance","binary"}
        Methods for functionnal & mixed (covariance, correlation...)
        Methods for geometric (binary or distance)
        
  
    spars : float 
        Threshold sparsity: 1: Full (Geometric)
                            Between 0 and 1: sparse (Functional)
                            Float (Geometric and Mixed): Radius of neighbourood
        
    geo_alpha: float
        Parameter for distance gaussian kernel (default 0.0001)
        For geometric or mixed graph
    
    verbose: 0 or 1
    
    w_name : only for Kalofolias - str containing path to graph W in mat file 
    
    absolute : True or False (work with absolute functional measures (corr or cov))    
    
    Attributes
    ----------
    input_shape : tuple
        The shape the data passed to :meth:`fit`
        
    G : Graph (structure as PyGSP)
    """
    
    def __init__(self, rest=[0,0], coords=[0], kind='geometric',
                 method='distance',spars=0, geo_alpha=0.0001, verbose=0, w_name=None, absolute=False):
        self.kind = kind
        self.method=method
        self.spars=spars
        self.geo_alpha=geo_alpha
        self.rest=rest
        self.coords=coords
        self.verbose=verbose
        self.w_name=w_name
        self.absolute=absolute
        
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
        if self.kind == None:
            self.G=[]
        else:
            self.G = create_graph(self.rest,self.kind,self.method,self.coords,self.spars,self.geo_alpha,self.verbose,self.w_name)
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
            The array containing the X transformed in Graph Space
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
        if self.kind == None:
            X_hat = X.T
        else:
            X_hat=operators.gft(self.G, X.T)        
        return X_hat.T
        
        
        
    def inverse_transform(self, X_hat, copy=None):
        """Scale back the data to the original representation

        Parameters
        ----------
        X_hat : array-like, shape [n_samples, n_features]
            The data in graph space.
            
        Returns
        ----------
        X : array-like of shape = [n_samples, n_features]
            Data in the original space.
        """
        check_is_fitted(self, ['G'])

        # Input validation
        X_hat = check_array(X_hat)
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X_hat.shape != self.input_shape_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        X=operators.igft(self.G, X_hat.T)        
        return X.T
    
    def sample(self, X, k, fmin, fmax):
        """ GraphSampling - select the k vertices
        where the signal energy (weighted coherence) is the most concentrated in data X for
        a frequency band of interest delimited by indices (fmin, fmax), 
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        
        k : number of vertices to extract
        
        fmin : Frequency min (int)
        
        fmax: Frequency max (int or 'max')
        
        Returns
        -------
        indices: indices of selected vertices
        
        X_sampled: array of int of shape = [n_samples, k]
            The array containing the reduced data 'X'
        """
        check_is_fitted(self, ['G'])
        
        if fmax=='max':
            Uk = self.G.U[:, fmin:]        
        else: 
            Uk = self.G.U[:,fmin:fmax]

        weight_opt = (Uk**2).sum(1)/(Uk[:]**2).sum();   
        indices=f(weight_opt, k)
        X_sampled=X.T[indices]      
    
        return X_sampled.T, indices
