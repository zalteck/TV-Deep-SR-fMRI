#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from utils import MRIdataset_NoDownsize, ssim3D
from options import set_opts

def evaluate_memory(step):
    t = torch.cuda.get_device_properties(0).total_memory/1024/1024/1024
    r = torch.cuda.memory_reserved(0)/1024/1024/1024
    a = torch.cuda.memory_allocated(0)/1024/1024/1024
    f = r-a  # free inside reserved
    print(step,'- Total:', t, 'GB Reserv:', r, 'GB Alloc:', a, 'GB Free:',f)

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
    
    
def main():
    
    
    
    args = set_opts()

    print('Arguments:')
    for arg in vars(args):
        print('{:<25s}: {:s}'.format(arg, str(getattr(args, arg))))    
    
    evaluate_memory('Begin')

    # In[2]:


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
    save_path= args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path) 
        
    factor=args.factor
    down_ratios= (1/factor)*np.ones(3)
    up_ratios = factor*np.ones(3)
    
    pixel_down_ratio,pixel_up_ratio=factor.as_integer_ratio() #1.25 -> 5,4
    
#     n_slices=12*8 # multiples of 12 work fine  
    n_slices = args.n_slices
    n_slices = n_slices - n_slices%pixel_up_ratio
    
    print('Expect a maximum usage of', (n_slices**3*4 + 2*factor**3*n_slices**3*4)/1024/1024, 'GB for the data')
    
    train_stimuli = args.training_stimuli
    validation_stimuli = args.validation_stimuli
#     train_stimuli = ['resting', 'stereo']
#     validation_stimuli=['motion','color']
   
    
  


    resume= False
    checkpoint_file = args.checkpoint_path

    model_name= 'SRTV_NoResize'

    resume=False
    checkpoint_file = ''


    # In[3]: LOADING DATA
#     OpNeuro_path= '/scratch/fperezbueno/Boston/OpenNeuro'

    #Build training

    training_list=[]
    print('training with stimuli:')
    for stimuli in train_stimuli[:]:
        print(stimuli, end=',')
        training_list = training_list + glob.glob(train_data_path +'/*/'+stimuli+'/*/f.nii*') #subject/stimuli/sesion/f.nii or f.nii.gz
    print('')
    print('training_list len:',len(training_list))

    
    validation_list=[]
    print('training with stimuli:')
    for stimuli in validation_stimuli[:]:
        print(stimuli, end=',')
        validation_list = validation_list + glob.glob(test_data_path +'/*/'+stimuli+'/*/f.nii*') #subject/stimuli/sesion/f.nii or f.nii.gz
    print('')
    print('validation_list len:',len(validation_list))
    

    device2='cpu'
    train_dataset = MRIdataset_NoDownsize(training_list, device=device2, time_choice='random', slices=n_slices, 
                                          scalefactor=factor )
    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=1, shuffle=False,
            num_workers=1, pin_memory=False)

    val_dataset = MRIdataset_NoDownsize(validation_list, device=device2, slices=n_slices, scalefactor=factor)
    val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=1, shuffle=True,
            num_workers=1, pin_memory=False)

#     test_dataset = MRIdataset_NoDownsize(test_list, device=device2, slices=40)
#     test_loader = torch.utils.data.DataLoader(test_dataset,
#             batch_size=1, shuffle=True,
#             num_workers=1, pin_memory=False)


    model = SRmodel(num_filters, num_residual_blocks, kernel_size, use_global_residual).to(device)
    evaluate_memory('Model_in')
    criterion = torch.nn.MSELoss()
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
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
          .format(42, checkpoint['epoch']))
        epoch= checkpoint['epoch']
      else:
        print("=> no checkpoint found at '{}'".format(check_file))





    model_name= model_name+str(lr)+'_'+str(TV_weight)#+'_'+str(lr_steps)
    epoch=0

    # base_path= '/home/fperezbueno/Boston'
    out_path= save_path + '/Output_'+model_name+'/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    model_storage = out_path + '/Checkpoints/' #'/content/drive/MyDrive/BCBL/SRcode/Checkpoints/'
    if not os.path.exists(model_storage):
        os.mkdir(model_storage)
    log_name= 'log'+model_name+datetime.datetime.now().strftime("%d%m_%H%M")+'.csv'
    print('results in:',out_path+log_name)
    # empty_folder(out_path)
    n_epochs = 200


    # resume= False
    if resume:
        resume_checkpoint(checkpoint_file)


    best = 100
    aff2=np.ones([4,4])
    save_freq = 50

    for epoch in range(n_epochs+1):
    #     epoch+=1
        adjust_learning_rate(optimizer, epoch)
        losses = AverageMeter()
        val_train_loss = AverageMeter()
        # print('Iter:',iter)
        for iter, (input, upscaled) in enumerate(train_loader):
    #         evaluate_memory('Iter')
            input=input.to(device)
    #         evaluate_memory('input_in')
            upscaled=upscaled.to(device)
    #         print('upscaled_shape',upscaled.shape)
    #         evaluate_memory('upscaled_in')
    #         target=target.to(device)
            optimizer.zero_grad()
    #         input, upscaled, target = next(gen)

            pred = model(upscaled[None, ...])
#             evaluate_memory('Predicted_in')

            downgraded = myzoom_torch(pred[0,0,:,:,:], down_ratios, device=device)

            mse_loss = criterion(downgraded[None,...], input)
            reg_loss = TVloss_3d(pred)
            loss = mse_loss + TV_weight*reg_loss
            losses.update(loss.item(),1)
            val_train_loss.update(mse_loss.item(),1)
            
            if epoch==0 and iter==0: evaluate_memory('Everything_in')
            
            loss.backward()
            optimizer.step()
    #         running_loss +=loss.item()
    #         mse_val = criterion(pred, target)
    #         val_train_loss.update(mse_val.item(),1)
            print(datetime.datetime.now().strftime("[%d/%m %H:%M]") +
                  'Iter:',iter,'-- loss:',loss.item(), 'mse:',mse_loss.item(),'TV:', TV_weight*reg_loss.item(), 0)

            # writer.add_scalar('Loss/loss', loss.item(), iter)
            # writer.add_scalar('Loss/mse', mse_loss.item(), iter)
            # writer.add_scalar('Loss/TV', TV_weight*reg_loss.item(), iter)

            save_log(out_path+log_name,iter, loss.item(), mse_loss.item(), TV_weight*reg_loss.item(), 0,
                    name_list=['iter', 'loss', 'mse_loss', 'weighted_TV_loss', 'None', 'None', 'None'])

        save_log(out_path+'EPOCH'+log_name,epoch, losses.avg, val_train_loss.avg, 0, 0, 0,
                name_list=['epoch', 'loss', 'mse_loss', 'None', 'None', 'None', 'None'])

        if epoch > 5 and epoch % save_freq == 0:
    #       save_volume(torch.unsqueeze(torch.squeeze(pred), dim=3).detach().cpu().numpy(), aff2, out_path+'epoch_'+str(epoch)+'.nii.gz')
          save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            # 'best_prec1': best_prec1,
            }, filename=os.path.join(model_storage, 'checkpoint_{}.tar'.format(epoch)))

        # VALIDATE
        with torch.no_grad():
            val_loss= AverageMeter()
            ssim_loss= AverageMeter()
            for iter, (input, upscaled) in enumerate(val_loader):
                input=input.to(device)
                upscaled=upscaled.to(device)
    #             target=target.to(device)
                pred = model(upscaled[None, ...])
                downgraded = myzoom_torch(pred[0,0,:,:,:], down_ratios, device=device)
                mse_val = criterion(downgraded[None,...], input)
#                 if epoch<2:
#                     print('downg',downgraded[None,None,...].shape, 'input', input[None,...].shape)
                ssim_loss.update(ssim3D(downgraded[None,None,...].double(), input[None,...].double()).item(), 1)
                val_loss.update(mse_val.item(), 1)
            save_log(out_path+'VALIDATION'+log_name,epoch, val_loss.avg, ssim_loss.avg, 0, 0,
                    name_list=['epoch', 'mse_loss', 'ssim_loss', 'None', 'None', 'None', 'None'])

        if val_loss.avg < best:
            print('best improved. Epoch:',epoch)
            best=val_loss.avg
            save_checkpoint({
            'epoch': iter + 1,
            'state_dict': model.state_dict(),
            # 'best_prec1': best_prec1,
            }, filename=os.path.join(model_storage, 'best.tar'))

     # TEST
#         with torch.no_grad():
#             test_loss= AverageMeter()
#             ssim_loss= AverageMeter()
#             for iter, (input, upscaled) in enumerate(test_loader):
#                 input=input.to(device)
#                 upscaled=upscaled.to(device)
#     #             target=target.to(device)
#                 pred = model(upscaled[None, ...])
#                 downgraded = myzoom_torch(pred[0,0,:,:,:], 0.5*np.ones(3), device=device)
#                 mse_val = criterion(downgraded, input)
#                 ssim_loss.update(ssim3D(downgraded[None,None,...].double(), input[None,...].double()).item(), 1)
#                 val_loss.update(mse_val.item(), 1)
#             save_log(out_path+'TEST'+log_name,epoch, val_loss.avg, ssim_loss.avg, 0, 0)

    print('Run completed')


if __name__ == "__main__":
    main()








