Update on your side:
base_path in SRcodeTV_train.py, SRcodeTV_NoResize_train.py and Test_TVnoResize.py

What to run:
# TV experiments (Unsupervised)# These first reduce the image, by the chosen factor and then go back to 1mm, so we have a GT to evaluate.
python -u SRcodeTV_train.py --train_data_path=[BRAN_TRAINPATH] --test_data_path=[BRAN_TESTPATH] --factor=1.25 
python -u SRcodeTV_train.py --train_data_path=[BRAN_TRAINPATH] --test_data_path=[BRAN_TESTPATH] --factor=1.5
python -u SRcodeTV_train.py --train_data_path=[BRAN_TRAINPATH] --test_data_path=[BRAN_TESTPATH] --factor=1.75
python -u SRcodeTV_train.py --train_data_path=[BRAN_TRAINPATH] --test_data_path=[BRAN_TESTPATH] --factor=2

It is expected that train_path is a folder containing only the training subjects and test_path containing only the test/validation subjects.
The same can also run with the parameter --flavor=Supervised or --flavor=Eugenio to make it supervised (MSE loss and L1-0.3SSIM, respectively). 

I've also included TVlauncher.sh that does all of those in one, but that might take quite a long computation time. Hopefully one after another, I'm not very good at bash scripting. You can also modify it to run one flavor at a time.

#Without resizing. This one goes from 1mm to 0.8mm#Trains on FC only. Validation is done with the same unsupervised loss function, so we just know that we are not doing something unexpected on those images.

python -u SRcodeTV_NoResize_train.py --train_data_path=[BRAN_TRAINPATH] --test_data_path=[BRAN_TESTPATH] --factor=1.25

Once this one is trained. Inference is done with:

python -u Test_TVnoResize.py --checkpoint_path=[BRANPATH_best.tar] --test_data_path=[BRAN_TESTPATH] --save_path=[where you want the images] --factor=1.25

This last one copies all files in test_data_path but .nii* files and then superresolve the .nii* files of selected stimuli. 