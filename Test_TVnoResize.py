import nibabel as nib
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
import sys
import os
base_path= '/scratch/fperezbueno/Boston'
sys.path.append(base_path)
from utils import load_volume, save_volume, align_volume_to_ref, myzoom_torch, show_overview, TVloss_3d, save_log
import datetime
from models import SRmodel
# from generators import basic_loader
from utils import MRIdataset_NoDownsize, ssim3D, ensure_directory
from options import set_opts
import torch.optim as optim
import shutil
# save_data_path ='/scratch/fperezbueno/Boston/PruebaFMRICompleto'
    
def main():
    
    
    args = set_opts()

    print('Arguments:')
    for arg in vars(args):
        print('{:<25s}: {:s}'.format(arg, str(getattr(args, arg))))  
    
    if torch.cuda.is_available():
      device='cuda'
    else:
      device='cpu'
      print("************** WARNING, GPU NOT AVAILABLE. Running on CPU **************************")
    n_frames=5
    # Constants
    num_filters = 96  # you might need to modify this if you are using a different setting
    num_residual_blocks = 12 # you might need to modify this if you are using a different setting
    kernel_size = 3
    use_global_residual = False
    # ref_res = 1.5
    lr=args.lr
    lr_gamma=args.lr_decay
    lr_steps=1000
    TV_weight=args.TV_weight
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    save_data_path =args.save_path
    if not os.path.exists(save_data_path):
        os.mkdir(save_data_path) 
#     validation_stimuli=['motion','color']
    validation_stimuli = args.validation_stimuli

    
    factor=args.factor
    down_ratios= (1/factor)*np.ones(3)
    up_ratios = factor*np.ones(3)
    pixel_down_ratio,pixel_up_ratio=factor.as_integer_ratio() #1.25 -> 5,4
    
    overlap = 10
    if overlap%factor!=0: overlap+= pixel_down_ratio-overlap%pixel_down_ratio 
    lr_overlap=overlap/factor

#     lr_side=12*8 
    lr_side=args.n_slices

    resume= True
    checkpoint_file = args.checkpoint_path

    model_name= 'SRTV_NoResize'
        

    def resume_checkpoint(check_file,device='gpu'):
        if os.path.isfile(check_file):
            print("=> loading checkpoint '{}'".format(check_file))
            checkpoint = torch.load(check_file, map_location=device)
            # args.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
              .format(42, checkpoint['epoch']))
            epoch= checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(check_file))
            
    model = SRmodel(num_filters, num_residual_blocks, kernel_size, use_global_residual).to(device)
    

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if resume:
        resume_checkpoint(checkpoint_file, device=device)
        


#SCAN TEST_DATA_PATH AND COPY EVERYTHING BUT .NII files
    everything = glob.glob(test_data_path+'/*')
    for anything in everything:
    #     print(anything)
        everything += glob.glob(anything+'/*')
        new_thing = anything.replace(test_data_path, save_path)
        #new thing can be either folder or file
        if os.path.isdir(anything):
            if not os.path.exists(new_thing):
                os.mkdir(new_thing) 
        elif '.nii' not in new_thing: 
            shutil.copy(anything, new_thing)
    print('Validation folder duplicated without .nii files in:', save_path) 
        
    
    validation_list=[]
    print('training with stimuli:')
    for stimuli in validation_stimuli[:]:
        print(stimuli, end=',')
        validation_list = validation_list + glob.glob(test_data_path +'/*/'+stimuli+'/*/f.nii*') #subject/stimuli/sesion/f.nii or f.nii.gz
    print('')
    print('validation_list len:',len(validation_list))
    
    


    for input_file in validation_list[0:]:
        
        img_load, aff = load_volume(input_file)
#         img_load=img_load[:,:,:,0:3]
        
        maxi = np.max(img_load)
        x_len, y_len, z_len, t_len = img_load.shape
        
        #ensure upscalable dimensions
        image = np.zeros([x_len+x_len%pixel_up_ratio, y_len+y_len%pixel_up_ratio, z_len+z_len%pixel_up_ratio, t_len])
        image[:x_len, :y_len,:z_len,:]=img_load[:x_len, :y_len,:z_len,:]
        x_len, y_len, z_len, t_len = image.shape

        print('Processing:', input_file, 'with vols:', image.shape)
        result= torch.zeros(np.array([factor*x_len, factor*y_len, factor*z_len, t_len],dtype='int').tolist())
        total_slices = z_len
        
        aff_upscaled = aff.copy()
        for j in range(3):
            aff_upscaled[:-1, j] = aff_upscaled[:-1, j] / up_ratios[j]
        aff_upscaled[:-1, -1] = aff_upscaled[:-1, -1] - np.matmul(aff_upscaled[:-1, :-1], 0.5 * (up_ratios - 1))


        for vol in range(t_len):
            #init loop h
            flag_h=False
            lr_init_h=0
            lr_stop_h= np.clip(lr_side,0,image.shape[0])
            
            while not flag_h:
                if lr_stop_h==image.shape[0]: flag_h=True #last?

                #calculate index hr
                hr_init_h=int(factor*lr_init_h+overlap)
                if lr_init_h==0: hr_init_h=0
                hr_stop_h=int(factor*lr_stop_h-overlap)
                if lr_stop_h==image.shape[0]: hr_stop_h=int(factor*lr_stop_h)

                #init loop w
                flag_w=False
                lr_init_w=0
                lr_stop_w= np.clip(lr_side,0,image.shape[1])
                
                while not flag_w:
                    if lr_stop_w==image.shape[1]: flag_w=True

                    hr_init_w=int(factor*lr_init_w+overlap)
                    if lr_init_w==0: hr_init_w=0
                    hr_stop_w=int(factor*lr_stop_w-overlap)
                    if lr_stop_w==image.shape[1]: hr_stop_w=int(factor*lr_stop_w)

                    #init loop d
                    flag_d=False
                    lr_init_d=0
                    lr_stop_d= np.clip(lr_side,0,image.shape[2])
                    
                    while not flag_d:
                        if lr_stop_d==image.shape[2]: flag_d=True

                        hr_init_d=int(factor*lr_init_d+overlap)
                        if lr_init_d==0: hr_init_d=0
                        hr_stop_d=int(factor*lr_stop_d-overlap)
                        if lr_stop_d==image.shape[2]: hr_stop_d=int(factor*lr_stop_d)
        #                 print('hr_init_d:',hr_init_d, hr_stop_d)

                        #observe
                        hr = np.squeeze(image[lr_init_h:lr_stop_h,lr_init_w:lr_stop_w,lr_init_d:lr_stop_d,vol:vol+1])
                        hr = torch.tensor(hr, device=device).float()
                        hr = (hr / maxi)
#                         x_len, y_len, z_len, t_len = observed.shape
            #             print(lr_init_h, lr_stop_h, end=' - ')

                        #upsample 
#                         upsampled=torch.ones(np.array([factor*x_len, factor*y_len, factor*z_len],dtype='int').tolist())
                        with torch.no_grad():
                            upscaled = myzoom_torch(hr, up_ratios, device=device).float()
                            upscaled=upscaled.to(device)
                            pred = model(upscaled[None, None, ...])
                    

                        #save only useful voxels
                        result[hr_init_h:hr_stop_h,
                               hr_init_w:hr_stop_w,
                               hr_init_d:hr_stop_d,vol] = pred[0,0,overlap*(lr_init_h!=0):upscaled.shape[0]-overlap*(not flag_h),
                                                                   overlap*(lr_init_w!=0):upscaled.shape[1]-overlap*(not flag_w),
                                                                   overlap*(lr_init_d!=0):upscaled.shape[2]-overlap*(not flag_d)]
                        
                        
                        #update d
                        lr_init_d=int(lr_stop_d-2*lr_overlap)
                        lr_stop_d=int(np.clip(lr_init_d+lr_side,0,image.shape[2]))
            
                    #update w
                    lr_init_w=int(lr_stop_w-2*lr_overlap)
                    lr_stop_w=int(np.clip(lr_init_w+lr_side,0,image.shape[1]))

                #update h
                lr_init_h=int(lr_stop_h-2*lr_overlap)
                lr_stop_h=int(np.clip(lr_init_h+lr_side,0,image.shape[0]))
            print(vol+1,'/',t_len, 'vol completed')

        #Force 0.8mm output from 0.5mm
#         new_result = myzoom_torch(result, 0.625*np.ones(3), device='cpu').float()
#         result=new_result
        result= maxi*result
        outfilename=(input_file.replace(train_data_path,save_data_path))
        ensure_directory(outfilename)
        save_volume(result, aff_upscaled, outfilename)
        
if __name__ == "__main__":
    main()
    print('COMPLETED')