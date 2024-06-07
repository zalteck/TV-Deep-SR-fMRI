#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nibabel as nib
import glob
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
import torch
import torch.optim as optim
import sys
import os
import random
base_path= '/scratch/fperezbueno/Boston'
sys.path.append(base_path)
from utils import load_volume, save_volume, align_volume_to_ref, myzoom_torch, show_overview, TVloss_3d, save_log
import datetime
from models import SRmodel
# from generators import basic_loader
from utils import MRIdataset, ssim3D
from options import set_opts
from torch.nn import L1Loss

def seed_everything(seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

class AverageMeter(object):
        """Computes and stores the average and current value"""
        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

# def empty_folder(path):
#     files= glob.glob(path+'/*')
#     for f in files:
#         if not os.path.isdir(f):
#             os.remove(f)
#         else:
#             empty_folder(f)

        
def main():
#     train_stimuli = ['resting', 'stereo']
#     validation_stimuli=['motion','color']
    
    args = set_opts()

    print('Arguments:')
    for arg in vars(args):
        print('{:<25s}: {:s}'.format(arg, str(getattr(args, arg))))
    

    seed_everything()

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
    n_epochs = args.epochs
    lr=args.lr
    lr_gamma=args.lr_decay
    lr_steps=1000
    TV_weight=args.TV_weight
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    save_path= args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path) 
        
    train_stimuli = args.training_stimuli
    validation_stimuli = args.validation_stimuli
    factor=args.factor
    down_ratios= (1/factor)*np.ones(3)
    up_ratios = factor*np.ones(3)

    flavor = args.flavor
    
    if flavor not in ['TV','Eugenio','Supervised']:
        print('Error: Flavor not implemented, please choose from TV, Eugenio, or Supervised')
        sys.exit()
    model_name= 'SR_'+flavor+'_'+str(factor)
    if flavor=='Eugenio':
        TV_weight=0.3
        

    resume= False
    checkpoint_file = args.checkpoint_path
#     log_file='/scratch/fperezbueno/Boston/Output_OpenNeuro_NoGlobRes0.001_0.01_1000/logOpenNeuro_NoGlobRes0.001_0.01_10001404_1545.csv'

    # In[3]: LOADING DATA
#     OpNeuro_path= '/scratch/fperezbueno/Boston/OpenNeuro'
# WORKING HERE: BUILD DATA READER
    #Build training
    
#     valid_subjects=['01','02','03','04','05','06','09','10','14','15','16','17','18','20','21','22']

    training_list=[]
    print('training with stimuli:')
    for stimuli in train_stimuli[:]:
        print(stimuli, end=',')
#         training_list= training_list + glob.glob(OpNeuro_path+'/sub-'+subject+'/*/fu*/*full*bold*')
        training_list = training_list + glob.glob(train_data_path +'/*/'+stimuli+'/*/f.nii*') #subject/stimuli/sesion/f.nii or f.nii.gz
    print('')
    print('training_list len:',len(training_list))


    validation_list=[]
    print('training with stimuli:')
    for stimuli in validation_stimuli[:]:
        print(stimuli, end=',')
#         training_list= training_list + glob.glob(OpNeuro_path+'/sub-'+subject+'/*/fu*/*full*bold*')
        validation_list = validation_list + glob.glob(test_data_path +'/*/'+stimuli+'/*/f.nii*') #subject/stimuli/sesion/f.nii or f.nii.gz
    print('')
    print('validation_list len:',len(validation_list))
    

    device2='cpu'
    train_dataset = MRIdataset(training_list, device=device2, time_choice='random', scalefactor=factor)
    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True)

    val_dataset = MRIdataset(validation_list, device=device2, scalefactor=factor)
    val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=1, shuffle=True,
            num_workers=1, pin_memory=True)


    model = SRmodel(num_filters, num_residual_blocks, kernel_size, use_global_residual).to(device)
    criterion = torch.nn.MSELoss()
    loss_fn = L1Loss()
    # criterion(pred[0,0,:,:,:], upscaled_patch[:,:,:,c])

    import torch.optim as optim

    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # In[8]:

    

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
        lr_new = lr * (lr_gamma ** (epoch // lr_steps))
        # print('lr:',lr_new)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_new

    def save_checkpoint(state, filename='checkpoint.pth.tar'):
        """
        Save the training model
        """
        torch.save(state, filename)

    def resume_checkpoint(check_file):
      if os.path.isfile(check_file):
        print("=> loading checkpoint '{}'".format(check_file))
        checkpoint = torch.load(check_file)
        # args.start_epoch = checkpoint['epoch']
        best = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print('WARNING: NO OPTIMIZER CHECKPOINT *************')
        print("=> loaded checkpoint '{}' (epoch {})"
          .format(42, checkpoint['epoch']))
        epoch= checkpoint['epoch']
      else:
        print("=> no checkpoint found at '{}'".format(check_file))

    


    # In[ ]:



#     model_name= model_name+str(lr)+'_'+str(TV_weight)#+'_'+str(lr_steps)
    epoch=0

    # base_path= '/home/fperezbueno/Boston'
    out_path= save_path + '/'+model_name+'/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    model_storage = out_path + '/Checkpoints/' #'/content/drive/MyDrive/BCBL/SRcode/Checkpoints/'
    if not os.path.exists(model_storage):
        os.mkdir(model_storage)
    log_name= 'log'+model_name+'.csv' #+datetime.datetime.now().strftime("%d%m_%H%M")
    print('results in:',out_path+log_name)
    # empty_folder(out_path)

    best = 100
    aff2=np.ones([4,4])
    save_freq = 500
    clip_grad=1e5
    
    # resume= False
    if resume:
        resume_checkpoint(checkpoint_file)
    #     epoch=int(checkpoint_file.rsplit('_',1)[-1].replace('.tar',''))
        print('resuming at epoch:',epoch,'with best:',best)
        log_name=log_file.replace(out_path,'')




    for epoch in range(epoch,n_epochs+1):
    #     epoch+=1
        adjust_learning_rate(optimizer, epoch)
        losses = AverageMeter()
        val_train_loss = AverageMeter()
        # print('Iter:',iter)
        for iter, (input, upscaled, target) in enumerate(train_loader):
            input=input.to(device) #torch.Size([1, x/2, y/2, z/2])
            upscaled=upscaled.to(device) #torch.Size([1, x, y, z])
            target=target.to(device) #torch.Size([1, x, y, z])
            optimizer.zero_grad()
    #         input, upscaled, target = next(gen)

            pred = model(upscaled[None, ...]) #torch.Size([1, 1, x, y, z])
            downgraded = myzoom_torch(pred[0,0,:,:,:], down_ratios, device=device) #torch.Size([x/2, y/2, z/2])
            
#             print('input shape', input.shape)
#             print('upscaled shape', upscaled.shape)
#             print('target shape', target.shape)
#             print('pred shape', pred.shape)
#             print('down shape', downgraded.shape)
# #             input shape torch.Size([1, 86, 86, 48])
# #             upscaled shape torch.Size([1, 129, 129, 72])
# #             target shape torch.Size([1, 129, 129, 72])
# #             pred shape torch.Size([1, 1, 129, 129, 72])
# #             down shape torch.Size([86, 86, 48])

            if flavor=='TV':
                mse_loss = criterion(downgraded[None,...], input)
                reg_loss = TVloss_3d(pred)
                loss = mse_loss + TV_weight*reg_loss
            elif flavor=='Supervised':
                mse_loss = criterion(pred[0,:,:,:,:].float(), target.float())
                reg_loss=torch.zeros(1)
                loss = mse_loss
            elif flavor=='Eugenio':
                l1_loss = loss_fn(pred.float(), target.float())
                mse_loss = l1_loss
                reg_loss = ssim3D(pred.double(), target.double())
                loss = l1_loss - 0.3*reg_loss
                
            losses.update(loss.item(),1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
    #         running_loss +=loss.item()
            mse_val = criterion(pred[0,:,:,:,:], target)
            val_train_loss.update(mse_val.item(),1)
            print(datetime.datetime.now().strftime("[%d/%m %H:%M]") + 'Ep:',epoch,
                  'Iter:',iter,'-- loss:',loss.item(), 'mse:',mse_loss.item(),'TV:', TV_weight*reg_loss.item(), 'val:',mse_val.item())

            # writer.add_scalar('Loss/loss', loss.item(), iter)
            # writer.add_scalar('Loss/mse', mse_loss.item(), iter)
            # writer.add_scalar('Loss/TV', TV_weight*reg_loss.item(), iter)

            save_log(out_path+log_name,iter, loss.item(), mse_loss.item(), TV_weight*reg_loss.item(), mse_val.item(),
                    name_list=['iter', 'loss', 'mse_loss', 'weighted_reg_loss', 'gt_mse_loss', 'None', 'None'])

        save_log(out_path+'EPOCH'+log_name,epoch, losses.avg, val_train_loss.avg, 0, 0,
                 name_list=['epoch', 'loss', 'gt_mse_loss', 'None', 'None', 'None', 'None'])

        if epoch > 5 and epoch % save_freq == 0:
    #       save_volume(torch.unsqueeze(torch.squeeze(pred), dim=3).detach().cpu().numpy(), aff2, out_path+'epoch_'+str(epoch)+'.nii.gz')
          save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_prec1': best,
            }, filename=os.path.join(model_storage, 'checkpoint_{}.tar'.format(epoch)))

        # VALIDATE
        with torch.no_grad():
            gt_loss= AverageMeter()
            val_loss= AverageMeter()
            ssim_loss= AverageMeter()
            for iter, (input, upscaled, target) in enumerate(val_loader):
                input=input.to(device)
                upscaled=upscaled.to(device)
                target=target.to(device)
                pred = model(upscaled[None, ...])
                mse_gt = criterion(pred[0,:,:,:,:], target)
                gt_loss.update(mse_gt.item(),1)
                ssim_loss.update(ssim3D(pred.double(), target.double()).item(), 1)
                
                downgraded = myzoom_torch(pred[0,0,:,:,:], down_ratios, device=device)
                mse_val = criterion(downgraded[None,...], input)
                val_loss.update(mse_val.item(), 1)
            save_log(out_path+'VALIDATION'+log_name,epoch, val_loss.avg, gt_loss.avg, ssim_loss.avg, 0,
                     name_list=['epoch', 'mse_loss', 'gt_mse_loss', 'gt_ssim_loss', 'None', 'None', 'None'])

        if val_loss.avg < best:
            print('best improved. Epoch:',epoch)
            best=val_loss.avg
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_prec1': best,
            # 'best_prec1': best_prec1,
            }, filename=os.path.join(model_storage, 'best.tar'))

     # TEST
#         with torch.no_grad():
#             test_loss= AverageMeter()
#             ssim_loss= AverageMeter()
#             for iter, (input, upscaled, target) in enumerate(test_loader):
#                 input=input.to(device)
#                 upscaled=upscaled.to(device)
#                 target=target.to(device)
#                 pred = model(upscaled[None, ...])
#                 mse_val = criterion(pred, target)
#                 ssim_loss.update(ssim3D(pred.double(), target.double()).item(), 1)
#                 val_loss.update(mse_val.item(), 1)
#             save_log(out_path+'TEST'+log_name,epoch, val_loss.avg, ssim_loss.avg, 0, 0)

    print('Run completed')



   

if __name__ == "__main__":
    main()






