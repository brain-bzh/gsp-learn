#!/usr/bin/python
# -*- coding: latin-1 -*-

__author__ = 'mathildemenoret'
__instit__ = 'imt_atlantique'
__funding__ = 'cominlabs-neuralcommunication'


  
import numpy as np
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img

def plot_eigenvector(graph,nb,mask1,mask2=None,bg_img=False):
    U=graph.U
    visu=mask1.inverse_transform(U.T)
    if mask2:
        visu=mask2.inverse_transform(visu)
    if bg_img:
        for i in range(nb):
            plot_stat_map(index_img(visu, i),vmax=0.04,bg_img=bg_img)
    else:
        for i in range(nb):
            plot_stat_map(index_img(visu, i),vmax=0.04)

    
    
    
def plot_selectedregions(pipeline,masker,weights=None,anova_name='anova',mask2=None,bg_img=False,display_mode='z'):
    if anova_name:
        #index= pipeline.named_steps[anova_name].get_support()
        if weights is None:
            weights=np.ones(pipeline.named_steps[anova_name].k)
        weight_full=pipeline.named_steps[anova_name].inverse_transform(weights)
    else: weight_full=weights
    weight_img = masker.inverse_transform(weight_full)
    if mask2:
        weight_img=mask2.inverse_transform(weight_img)
    
    if bg_img:
        plot_stat_map(weight_img, title='SVM weights',bg_img=bg_img,display_mode=display_mode,cmap='bwr')
    else:
        plot_stat_map(weight_img, title='SVM weights',display_mode=display_mode,cmap='bwr')
        


    
    
    
# TEST
##############################################
#from nilearn import datasets
#from nilearn.input_data import NiftiLabelsMasker
#from nilearn.datasets import load_mni152_brain_mask
#from sklearn.externals.joblib import Memory
#mem = Memory('nilearn_cache')
#
#coords=get_masker_coord('BASC')
#rest_filename=  fold+'mni/roirest_'+smt+'_'+n+'.npz' 
#gr=GraphTransformer(rest=rest, coords=coords, kind='geometric',
#                     method='distance',spars=0,geo_alpha=0.001)
#basc = datasets.fetch_atlas_basc_multiscale_2015(version='sym')['scale444']
#brainmask = load_mni152_brain_mask()
#masker = NiftiLabelsMasker(labels_img = basc, mask_img = brainmask, 
#                           memory=mem, memory_level=1, verbose=0,
#                           detrend=True, standardize=False,  
#                           high_pass=0.01,t_r=2.28,
#                           resampling_target='labels')
#masker.fit()
#gr=GraphTransformer(rest=rest, coords=coords, kind='geometric',
#                     method='distance',spars=0,geo_alpha=0.0001)
#gr.fit(np.zeros((444,100)))
#plot_eigenvector(gr.G,4,masker)