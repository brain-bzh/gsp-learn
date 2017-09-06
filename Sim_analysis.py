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
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from sklearn.externals.joblib import Memory
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


# Number of simulated "subjects" (set of parameters)
suj=89
# Number of simulations par "subjects/set of parameters"
meas=20
# Number of k - feature selections
k=50
ncomp=k
# Frequencies for Graph Sampling    
fmin=222
fmax='max'
   
# Load Conditions file & Cross validation procedure
eventname='F:/sim/event_sim.csv'
label=np.array(np.recfromcsv(eventname,names='s'))['s']
# Create session file
block=np.repeat(np.r_[1:8],61)[0:421]
index=[]
for x in range(label.size):
    if label[x]!=label[x-1]:
        index.append(x)
label=np.delete(label,index,0)
block=np.delete(block,index,0)
rest_mask = label == b'Rest'
y = label[np.logical_not(rest_mask)]
y = (y == b'Face')
block=block[np.logical_not(rest_mask)]
cv = LeaveOneLabelOut(block/2)

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
              'g_semilocal_cov':{'kind':'mixed','method':'covariance','spars':0.5},
              'g_covariance':{'kind':'functional','method':'covariance','spars':1},
              'g_correlation':{'kind':'functional','method':'correlation','spars':1},
              'g_full':{'kind':'geometric','method':'distance','spars':1},
              'g_fundis':{'kind':'mixed','method':'correlation','spars':1},
              'g_distgeo':{'kind':'geometric','method':'distance','spars':0.5},
                        } 
verbose=1
absolute=True
geo_alpha=0.0001

###### Create file and variable result to save results 
##### (can be loaded if preexisting with less subjects)
import os.path   
picklename='F:/sim/result/pickleresults_s'+str(ncomp)+'_freq'+str(fmin)+str(fmax)+'_k'+str(k)+'.p'
if os.path.exists(picklename):
    result=pickle.load(open(picklename)) 
    sujlist=np.arange(np.where(result['fmri'].mean(axis=1))[0].max()+1,suj) 
    ns=suj-np.shape(result['fmri'])[0]
    for name,data in sorted(result.items()):
        result[name] = np.append(result[name],np.zeros((ns,20)),axis=0)
     
else:
    sujlist=range(suj)
    result={}
    result['fmri']= np.zeros((suj,meas))
    for red_name, red in sorted(reductionlist.items()): 
        result[red_name]= np.zeros((suj,meas))
    for graph_name in sorted(graphsname): 
        result[graph_name+'_fmri_sampled'+str(fmin)+str(fmax)]= np.zeros((suj,meas))
        result[graph_name+'_'+str(k)]= np.zeros((suj,meas))
        result[graph_name+'_hf'+str(k)]= np.zeros((suj,meas))

##### Begin Analysis
        
for s in sujlist:
    for n in range(meas):
        # Path of PARCELLATED data ['roi'] (shape [n_sample,n_feature])
        sim_filename='F:/sim/mni/roi_mni_'+str(s+1)+'_'+ str(n+1)+ '.npz'   
        if os.path.exists(sim_filename):
            roi=np.load(sim_filename)['roi']
            roi=np.delete(roi,index,0)

            # Separate rest from other conditions
            rest=roi[rest_mask]
            # Path of W for Kalofolias (matlab)
            w_name='F:/sim/rest/sim_'+str(s+1)+'_'+str(n+1)+'_w_kalofolias.mat'
            cond=roi[np.logical_not(rest_mask)]
            
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
                result[graph_name+'_fmri_sampled'+str(fmin)+str(fmax)][s,n] = classifiers_scores_sampled.mean()  
                
                # Classification with graph transform + anova
                classifiers_scores_graphspace = cross_val_score(
                            pipeline_graph_anova, cond, y,cv=cv)
                result[graph_name+'_'+str(k)][s,n] = classifiers_scores_graphspace.mean()  
                
                # Classification with graph transform + selection frequency
                cond_transf=gr.transform(cond)[:,:-k]
                classifiers_scores_graphspace = cross_val_score(
                            pipeline, cond_transf, y,cv=cv)
                result[graph_name+'_hf'+str(k)][s,n] = classifiers_scores_graphspace.mean()  
        
            classifiers_scores = cross_val_score(
                            pipeline, cond, y, cv=cv)                
            result['fmri'][s,n] = classifiers_scores.mean()
            
            for red_name, red in sorted(reductionlist.items()):
                pipeline_red=Pipeline([('scaler',scaler),('reduction',red), ('svm', svm)])
                classifiers_scores_red= cross_val_score(
                                    pipeline_red, cond, y, cv=cv)
                result[red_name][s,n] = classifiers_scores_red.mean()

    presult={}        #
    for name,data in sorted(result.items()):
        presult[name] = data[s,:]
    presult=pd.DataFrame(presult)
    presult.to_csv('F:/sim/result/result_suj'+str(s+1)+'_s'+str(ncomp)+'_freq'+str(fmin)+str(fmax)+'_k'+str(k)+'.csv',index=False,sep='\t')
    pickle.dump(result, open('F:/sim/result/pickleresults_s'+str(ncomp)+'_freq'+str(fmin)+str(fmax)+'_k'+str(k)+'.p', "wb" ) )


if suj>1:
    allsuj_result={}
    for name,data in sorted(result.items()):
        allsuj_result[name] = data.mean(1)
    allsuj_result=pd.DataFrame(allsuj_result)
    allsuj_result.to_csv('F:/sim/result/result_all_s'+str(ncomp)+'_freq'+str(fmin)+str(fmax)+'_k'+str(k)+'.csv',index=False,sep='\t')

