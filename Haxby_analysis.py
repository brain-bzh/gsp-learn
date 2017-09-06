# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 11:21:15 2017

@author: mmenoret
"""

from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from sklearn.externals.joblib import Memory
import nibabel as nib 
from nilearn.plotting import find_xyz_cut_coords
from nilearn.image import math_img
from gsplearn.GSPTransform import GraphTransformer
from sklearn.pipeline import Pipeline   
from sklearn.svm import SVC
import pandas as pd
from sklearn import preprocessing 
import numpy as np
from sklearn.linear_model import LogisticRegression
from nilearn.datasets import load_mni152_brain_mask,load_mni152_template
from nilearn.image import resample_img
from nilearn import plotting, image 
from nibabel.nifti1 import Nifti1Image
from sklearn.cross_validation import LeaveOneLabelOut, cross_val_score, permutation_test_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import pickle


def data_behaviour(suj):
    haxby = datasets.fetch_haxby(data_dir='D:/', subjects=6)
    # Load the behavioral data
    labels = np.recfromcsv(haxby.session_target[suj], delimiter=" ")
    y = labels['labels']
    session = labels['chunks']
    return y, session


def get_masker_coord(atlasname):
    """ Get coordinates of parcellation (3D) from a brain atlas image 
    defined by labels (integers)
    
    Parameters:
    ---------
        atlasname : string - pathway of the atlas
            OR is atlas is BASC
                tuple  or list as 
                filename[0]='BASC'
                filename[1]='sym' or 'asym'
                filename[2]= str of nb of parcel (version): '444'
    """
    
    if 'BASC' in atlasname:    
        basc = datasets.fetch_atlas_basc_multiscale_2015(version=atlasname[1])['scale'+atlasname[2]]
        atlasname=basc
        
    nib_parcel = nib.load(atlasname)
    labels_data = nib_parcel.get_data()         
    #fetch all possible label values 
    all_labels = np.unique(labels_data)
    # remove the 0. value which correspond to voxels out of ROIs
    all_labels = all_labels[1:]
#    bari_labels = np.zeros((all_labels.shape[0],3))
#    ## go through all labels 
#    for i,curlabel in enumerate(all_labels):
#        vox_in_label = np.stack(np.argwhere(labels_data == curlabel))
#        bari_labels[i] = vox_in_label.mean(axis=0)
#        
    allcoords=[]
    for i,curlabel in enumerate(all_labels):
        img_curlab = math_img(formula="img==%d"%curlabel,img=atlasname)
        allcoords.append(find_xyz_cut_coords(img_curlab))
    allcoords=np.array(allcoords)
    return  allcoords  

########################################


# Number of k - feature selections
k=50
ncomp=k
# Frequencies for Graph Sampling    
fmin=222
fmax='max'
   

result_scores = {}

##### Parameters for Classification & Dimension Reduction
feature_selection = SelectKBest(f_classif, k=k)
scaler = preprocessing.StandardScaler()
svm= SVC(C=1., kernel="linear")  
logistic = LogisticRegression(C=1., penalty="l1")
logistic_l2 = LogisticRegression(C=1., penalty="l2")
# A dictionary, to hold all our classifiers
classifiers = {'SVC': svm,
           'log_l1': logistic,
           'log_l2': logistic_l2
           }
pca = PCA(n_components=k,svd_solver = 'full')
ica=FastICA(n_components=k)

reductionlist = { 'anova'+str(k):feature_selection,
             'ica'+str(k):ica,
             'pca'+str(k):pca,
            } 

### Parameters for Graph
# Get coordinates of atlas for geometric graphs
atlasname=['BASC','sym','444']
coords=get_masker_coord(atlasname)
# Graph to build with dictionary of parameters
graphsname = {'g_kalofolias':{'kind':'kalofolias'},
              'g_semilocal_cov':{'kind':'mixed','method':'covariance','spars':0.2},
              'g_covariance':{'kind':'functional','method':'covariance','spars':1},
              'g_correlation':{'kind':'functional','method':'correlation','spars':1},
              'g_full':{'kind':'geometric','method':'distance','spars':1},
              'g_fundis':{'kind':'mixed','method':'correlation','spars':1},
              'g_distgeo':{'kind':'geometric','method':'distance','spars':0.2},
                        } 
verbose=0
absolute=True
geo_alpha=0.0001

##### Begin Analysis

for suj in range(6):
    result_scores[suj] = {}
    y, session=data_behaviour(suj)
    data_name= 'D:/haxby2001/mni/roi_mni_'+str(suj)+'.npz'
    fmri=np.load(data_name)['roi']
    rest_mask = y == b'rest'
    condition_mask = np.logical_or(y == b'cat', y == b'face')
    rest=fmri[rest_mask]
    cond= fmri[condition_mask]
    session_label=session[condition_mask]
    cv = LeaveOneLabelOut(session_label/2)
    y = y[condition_mask]
    y = (y == b'cat')
    
    w_name='D:/haxby2001/rest/'+str(suj)+'_w_kalofolias.mat'       

           
    for graph_name, param in sorted(graphsname.items()):  
        if 'method' not in param:
            param['method']=False
        if 'spars' not in param:
            param['spars'] =1
        gr=GraphTransformer(rest=rest, coords=coords, 
                                    verbose=verbose,kind=param['kind'],
                                    method=param['method'],
                                    spars=param['spars'],
                                    w_name=w_name)
        gr.fit(cond)
    
        pipeline_graph_anova = Pipeline([('graph',gr),('anova', feature_selection), ('scale', scaler),('classif_name', svm)])
        pipeline = Pipeline([('scaler',scaler), ('svm', svm)])
        
        # Classification with graph sampling
        cond_sampled, idex=gr.sample(cond,k,fmin,fmax)
        classifiers_scores_sampled = cross_val_score(
                    pipeline, cond_sampled, y,cv=cv)
        result_scores[suj][graph_name+'_fmri_sampled'+str(fmin)+str(fmax)] = classifiers_scores_sampled.mean()  
        
        # Classification with graph transform + anova
        classifiers_scores_graphspace = cross_val_score(
                    pipeline_graph_anova, cond, y,cv=cv)
        result_scores[suj][graph_name+'_'+str(k)] = classifiers_scores_graphspace.mean()  
        
        # Classification with graph transform + selection frequency
        cond_transf=gr.transform(cond)[:,:-k]
        classifiers_scores_graphspace = cross_val_score(
                    pipeline, cond_transf, y,cv=cv)
        result_scores[suj][graph_name+'_hf'+str(k)] = classifiers_scores_graphspace.mean()  

    classifiers_scores = cross_val_score(
                    pipeline, cond, y, cv=cv)                
    result_scores[suj]['fmri'] = classifiers_scores.mean()
    
    for red_name, red in sorted(reductionlist.items()):
        pipeline_red=Pipeline([('scaler',scaler),('reduction',red), ('svm', svm)])
        classifiers_scores_red= cross_val_score(
                            pipeline_red, cond, y, cv=cv)
        result_scores[suj][red_name] = classifiers_scores_red.mean()

import pickle
import pandas as pd
pickle.dump(result_scores, open( "Haxby_result_svm_housecat"+str(k)+".p", "wb" ) )

Haxby_result =pd.DataFrame.from_dict(result_scores).transpose()
test=Haxby_result.copy()
for names in Haxby_result.keys():
    for i in range(6):
        test[names][i]=Haxby_result[names][i].mean()

test.to_csv('F:/new_Haxby_svm_catface_'+str(ncomp)+'_freq'+str(fmin)+str(fmax)+'_k'+str(k)+'.csv',index=False)
