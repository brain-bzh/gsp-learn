# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 11:21:15 2017

@author: mmenoret
"""

########################################

import numpy as np
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from sklearn.externals.joblib import Memory
from nilearn.datasets import load_mni152_brain_mask,load_mni152_template
from nilearn import image 
from nibabel.nifti1 import Nifti1Image
import os.path
import scipy.io as sio 

suj=89
meas=20

# Load Conditions file
eventname='F:/sim/event_sim.csv'
label=np.array(np.recfromcsv(eventname,names='s'))['s']
# Create session file
rest_mask = label == b'Rest'


# Prepare masker & Correct affine
template = load_mni152_template()
basc = datasets.fetch_atlas_basc_multiscale_2015(version='sym')['scale444']
orig_filename='F:/sim/template/restbaseline.nii.gz'
orig_img= image.load_img(orig_filename)
brainmask = load_mni152_brain_mask()
mem = Memory('nilearn_cache')
masker = NiftiLabelsMasker(labels_img = basc, mask_img = brainmask, 
                           memory=mem, memory_level=1, verbose=0,
                           detrend=False, standardize=False,  
                           high_pass=0.01,t_r=2,
                           resampling_target='labels')
masker.fit()

# Prep Classification
   
for s in np.arange(84,89):#range(suj):#
    for n in range(meas):
        sim_filename='F:/sim/sim_'+str(s+1)+'_'+ str(n+1)+ '.nii.gz'
        if os.path.exists(sim_filename): 
            sim_img = image.load_img(sim_filename)
            sim=sim_img.get_data()
            data_sim=Nifti1Image(sim,orig_img.affine)
            roi = masker.transform(data_sim)
            save_name= 'F:/sim/mni/roi_mni_'+str(s+1)+'_'+str(n+1)+'.npz'
            np.savez_compressed(save_name,roi=roi) 
            rest=roi[rest_mask]
            sio.savemat('F:/sim/rest/sim_'+str(s+1)+'_'+ str(n+1)+'_rest.mat', {'rest':rest})

        
