# TV-BASED DEEP 3D SELF SUPER-RESOLUTION FOR FMRI
DL Self Super Resolution

This repository implements the code for the paper submitted to the International Symposium on Biomedical Imaging (ISBI) 2025

# AFNI commands for preprocessing:

Mask:

3dAutomask -prefix sub_mask.nii.gz input.nii.gz

Preprocessing:

\>3dcalc -overwrite -a 'input.nii.gz[5..$]' -expr 'a' -prefix sub_tcat.nii.gz  
\>3dvolreg -overwrite -verbose -Fourier -prefix sub_volreg.nii.gz -base 0 -1Dfile sub_motion.1D -1Dmatrix_save sub_motion -maxdisp1D sub_maxdisp.1D sub_tcat.nii.gz  
\>1d_tool.py -infile sub_motion.1D -censor_motion 0.3 sub  
\>1d_tool.py -infile sub_motion.1D -derivative -demean -write sub_motion.deriv.1D  
\>1d_tool.py -infile sub_motion.1D -demean -write sub_motion.demean.1D  
\>3dTproject -overwrite -input sub_volreg.nii.gz -prefix sub_tproject.nii.gz -polort 4 -censor sub_censor_censor.1D -cenmode KILL -ort sub_motion.demean.1D -ort sub_motion.deriv.1D -passband 0.005 0.2 -mask sub_mask.nii.gz  

Correlation maps are calculated using AFNI InstaCorr with:  
Blur = 3mm, seed rad = 3

