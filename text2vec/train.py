import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from multiprocessing import cpu_count
import numpy as np
import argparse
import os
import time
import math

from model import Text2Vec
from loss import DNNLoss
from log_utils import plot_alignment_to_numpy
from dataset import BufferDataset, DataLoader
from dataset import get_data_to_buffer, collate_fn_tensor
from optimizer import ScheduledOptim
import hparams as hp
import utils
from loss import  AttentionBinarizationLoss
from torch_optimizer import Lamb
from torch.utils.tensorboard import SummaryWriter

val_error_batch_num=0

def prepare_output_folders_and_logger(output_directory):
    # Get shared output_directory ready
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    output_hparams_path = os.path.join(output_directory, hp.log_seed,'hparams.py') # run/{seed}/hparams.py
    print("saving current configuration in output dir")

    cmd='cp text2vec/hparams.py {}'.format(output_hparams_path)
    print(cmd)
    os.system(cmd) # copy the current config

    tboard_out_path = hp.tensorboard_logs_path# run/{seed}/tb_logs
    print("setting up tboard log in %s" % (tboard_out_path))
    logger = SummaryWriter(tboard_out_path)
    return logger

def parse_data_from_batch(batch):
 
    feat_target = batch['feat_target']
    text = batch['text']

    in_lens = batch['input_lengths']
    out_lens = batch['output_lengths']

    feat_pos=batch['feat_pos']
    text_pos = batch["src_pos"]
    max_feat_len=batch["feat_max_len"]

    attn_prior = batch['attn_prior']
    audiopaths = batch['audiopaths']


    if attn_prior is not None:
        attn_prior = attn_prior.cuda()

    feat_target = feat_target.cuda()
    text = text.cuda()
    in_lens, out_lens = in_lens.cuda(), out_lens.cuda()

    feat_pos=feat_pos.cuda()
    text_pos=text_pos.cuda()


    return (feat_target, text,
            in_lens, out_lens,
            feat_pos,text_pos,max_feat_len,
            attn_prior, audiopaths)

# todo : this validation is not efficient, may opt in the future
def compute_validation_loss(iteration, model, criterion, valset,
                            attention_kl_loss=None,
                            logger=None, train_config=None):

    global val_error_batch_num
    model.eval()
    
    with torch.no_grad():
        # val_sampler = DistributedSampler(valset) if n_gpus > 1 else None
        val_loader = DataLoader(valset, sampler=None, num_workers=8,
                                shuffle=False, batch_size=hp.batch_expand_size * hp.batch_size,
                                pin_memory=False, collate_fn=collate_fn_tensor,drop_last = True)
        print(len(val_loader))
        loss_outputs_full = {}
        val_WVF_loss,val_WVF_postnet_loss=[],[] # WVF-loss and WVF-postnet-loss,not duration-loss

        for i, batchs in enumerate(val_loader):
            for j, batch in enumerate(batchs):
                (feat_target, text, in_lens,
                 out_lens,feat_pos,text_pos,
                 max_feat_len,attn_prior, audiopaths) = parse_data_from_batch(batch)


                if len(in_lens)<hp.batch_size:
                    continue
                for si in range(hp.batch_size):
                    text_len=in_lens[si]

                    feat_len=out_lens[si]
                    b=attn_prior[si,:feat_len,:text_len].unsqueeze(0)
                    # try:
                    if True:
                        outputs =  model(
                            text[si][:text_len].unsqueeze(0), #[1,n-t]
                            text_pos[si][:text_len].unsqueeze(0),##[1,n-t]
                            feat_target[si][:feat_len].unsqueeze(0),#[1,n-f,768]
                            in_lens=text_len.unsqueeze(0), #[1]
                            out_lens=feat_len.unsqueeze(0),# [1]
                            WVF_pos=feat_pos[si][:feat_len].unsqueeze(0),#[1,n-f]
                            WVF_max_length=feat_len.item(),
                            attn_prior=b,
                        )
                        
                        
                        if feat_target.shape[1]<outputs['feat_output'].shape[1]:
                            val_error_batch_num+=1
                            
                            WVF_loss,WVF_postnet_loss = criterion(
                                            outputs['feat_output'][0,:feat_target.shape[1],:].unsqueeze(0),# 1,shape,1024
                                            outputs['feat_postnet_output'][0,:feat_target.shape[1],:].unsqueeze(0),
                                            feat_target[si].unsqueeze(0),
                                            # outputs['duration_predictor_output'],
                                            # outputs['duration']
                                            )
                        else:
                            WVF_loss,WVF_postnet_loss = criterion(outputs['feat_output'],
                                            outputs['feat_postnet_output'],
                                            feat_target[si,:outputs['feat_output'].shape[1],:].unsqueeze(0),
                                            # outputs['duration_predictor_output'],
                                            # outputs['duration']
                                            )
                    # except:
                    #     val_error_batch_num+=1
                    #     continue
                    val_WVF_loss.append(WVF_loss.item())
                    val_WVF_postnet_loss.append(WVF_postnet_loss.item())
        # pdb.set_trace()
        mean_WVF_loss = np.mean(val_WVF_loss)
        mean_WVF_postnet_loss = np.mean(val_WVF_postnet_loss)
        sum_WVF_loss = np.sum(val_WVF_loss)
        sum_WVF_postnet_loss = np.sum(val_WVF_postnet_loss)
        
        mean_total_loss = mean_WVF_loss+mean_WVF_postnet_loss
        sum_total_loss = sum_WVF_loss+sum_WVF_postnet_loss
    # pdb.set_trace()

    loss_outputs_full['mean_total_loss']= mean_total_loss
    loss_outputs_full['mean_WVF_loss'] = mean_WVF_loss
    loss_outputs_full['mean_WVF_postnet_loss'] = mean_WVF_postnet_loss
    loss_outputs_full['sum_WVF_loss']= sum_WVF_loss
    loss_outputs_full['sum_WVF_postnet_loss'] = sum_WVF_postnet_loss
    loss_outputs_full['sum_total_loss'] = sum_total_loss
    # loss_outputs_full['duration_loss'] = loss_tuple[2].item()
    # loss_outputs_full['attn_binarization_loss'] = binarization_loss.item()

    if logger is not None:
        logger.add_scalar('val/validation-data-num', len(val_WVF_loss), iteration)
        
        
        logger.add_scalar('val/total_loss(meam)', loss_outputs_full['mean_total_loss'], iteration)
        logger.add_scalar('val/WVF_loss(meam)', loss_outputs_full['mean_WVF_loss'], iteration)
        logger.add_scalar('val/WVF_postnet_loss(meam)', loss_outputs_full['mean_WVF_postnet_loss'], iteration)
        logger.add_scalar('val/total_loss(sum)', loss_outputs_full['sum_WVF_loss'], iteration)
        logger.add_scalar('val/WVF_loss(sum)', loss_outputs_full['sum_WVF_postnet_loss'], iteration)
        logger.add_scalar('val/WVF_postnet_loss(sum)', loss_outputs_full['sum_total_loss'], iteration)
        
        # logger.add_scalar('val/duration_loss', loss_outputs_full['duration_loss'], iteration)
        # logger.add_scalar('val/attn_binarization_loss', loss_outputs_full['attn_binarization_loss'], iteration)

        if outputs is not None:
            attn_used = outputs['attn']
            attn_soft = outputs['attn_soft'] # [bz,1,len_feat,len_text][0,0]->[len_feat,len_text]
            audioname = os.path.basename(audiopaths[0])
            if attn_used is not None:
                logger.add_image(
                    'val/attention_weights (align_soft)',
                    plot_alignment_to_numpy(
                        attn_soft[0, 0].data.cpu().numpy().T, title=audioname),
                    iteration, dataformats='HWC')
                logger.add_image(
                    'val/attention_weights_mas (align_hard)',
                    plot_alignment_to_numpy(
                        attn_used[0, 0].data.cpu().numpy().T, title=audioname),
                    iteration, dataformats='HWC')

    model.train()
    return loss_outputs_full


def main(args):

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    print("Use Text2Vec")
    # model = nn.DataParallel(Text2Vec()).to(device) # multi-gpu
    model=Text2Vec().to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of TTS Parameters:', num_param)

    # Get buffer
    print("Load data to buffer")
    buffer = get_data_to_buffer(hp.train_list)
    # val_buffer=get_data_to_buffer(hp.val_list)
    # Get dataset
    dataset = BufferDataset(buffer)
    # valset = BufferDataset(val_buffer)

    # Get Training Loader
    training_loader = DataLoader(dataset,
                                 batch_size=hp.batch_expand_size * hp.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn_tensor,
                                 drop_last=True,
                                 num_workers=8)

    

    text2vec_loss = DNNLoss().to(device)
    attention_kl_loss = AttentionBinarizationLoss()

    
    learning_rate=hp.learning_rate
    restart_epoch=0
    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        learning_rate=checkpoint['learning_rate']
        restart_epoch=checkpoint['epoch']
        print("\n---Model Restored at Step %d---\n" % args.restore_step)
    except:
        print("\n---Start New Training---\n")

        os.makedirs(hp.checkpoint_path,exist_ok=True)

    # Optimizer and loss
    # Note:Change Adam(fastspeech) to Lamb
    optimizer = Lamb(model.parameters(),
                    lr=learning_rate,
                    betas = (hp.beta1, hp.beta2),
                    eps = hp.epsilon,
                    weight_decay = hp.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(),
    #                 lr=learning_rate,
    #                 betas = (hp.beta1, hp.beta2),
    #                 eps = hp.epsilon,
    #                 weight_decay = hp.weight_decay)
    scheduled_optim = ScheduledOptim(optimizer,
                           learning_rate,
                           hp.n_warm_up_step,
                           args.restore_step)
    print("Defined Optimizer and Loss Function.")
    
    # Init logger
    os.makedirs(hp.logger_path,exist_ok=True)
    tensorboard_logger=prepare_output_folders_and_logger(hp.run_path)

    total_step = hp.epochs * len(training_loader) * hp.batch_expand_size
    print('\ntotal steps:',total_step,'len(training_loader)',len(training_loader),'\n')
    
    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()

    # Training
    model = model.train()
    iteration = args.restore_step+1
    
    error_batch_num=0
 
    for epoch in range(restart_epoch,hp.epochs):
        for i, batchs in enumerate(training_loader):
            # real batch start here
            for j, batch in enumerate(batchs):
                start_time=time.perf_counter()
                
                # Init
                scheduled_optim.zero_grad()

                # Get Data
  
                (wv_feat_target, text,
                 in_lens, out_lens,
                 wv_feat_pos, text_pos, max_wv_feat_len,
                 attn_prior, audiopaths) = parse_data_from_batch(batch)
 

                # in rad-tts: binarization_start_iter=6000,kl_loss_start_iter=18000
                # • [0, 6k): Use A_soft for the alignment matrix.
                # • [6k, 18k): start using Viterbi A_hard instead of A_soft,
                # • [18k, end): add binarization term λ2L_bin to the loss.
                #
                # but this model,A_soft must be binarize to A_hard, and fed to the LengthRegulator
                # so I set binarization_start_iter=0,so that 'binarize' always be True
 

                # try:
                if True:
                    outputs = model(
                        text,
                        text_pos,
                        wv_feat_target,
                        in_lens,
                        out_lens,
                        WVF_pos=wv_feat_pos,
                        WVF_max_length=max_wv_feat_len,
                        binarize_attention=True,
                        attn_prior=attn_prior,
                    )
 
                
                # compute the MSE between: 1. gt-feat and predicated feat
                #                          2. duration from hard-attn and duration from length_regulator
 
                    WVF_loss, WVF_postnet_loss, duration_loss = text2vec_loss(outputs['feat_output'],
                                                                          outputs['feat_postnet_output'],
                                                                          wv_feat_target,
                                                                          outputs['duration_predictor_output'],
                                                                          outputs['duration'])
                
                # except:
                    # error_batch_num+=1
                    # continue
                    
                total_loss = WVF_loss + WVF_postnet_loss +duration_loss

                w_bin = hp.binarization_loss_weight
               
  
                binarization_loss = attention_kl_loss(
                    outputs['attn'], outputs['attn_soft'])
                total_loss += binarization_loss * w_bin
   


                # Logger

                t_l = total_loss.item()
                m_l = WVF_loss.item()
                m_p_l = WVF_postnet_loss.item()
                d_l = duration_loss.item()
                attn_kl_l=binarization_loss.item()
 


                # log loss


                tensorboard_logger.add_scalar('train/total_loss' , t_l, iteration)
                tensorboard_logger.add_scalar('train/WVF_loss' , m_l, iteration)
                tensorboard_logger.add_scalar('train/WVF_postnet_loss' , m_p_l, iteration)
                tensorboard_logger.add_scalar('train/duration_loss' , d_l, iteration)
                tensorboard_logger.add_scalar('train/attn_binarization_loss' , attn_kl_l, iteration)
                # Backward
  
                total_loss.backward()
  
                # Clipping gradients to avoid gradient explosion
                if iteration%10==0:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), hp.grad_clip_thresh)
 
                # Update weights
                if args.frozen_learning_rate:
                    scheduled_optim.step_and_update_lr_frozen(
                        args.learning_rate_frozen)
                else:
                    scheduled_optim.step_and_update_lr()
 

                # Print and save
                if iteration % hp.log_step == 0:
                    Now = time.perf_counter()

                    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                        epoch + 1, hp.epochs, iteration, total_step)
                    str2 = "W2V Feat Loss: {:.4f}, W2V Feat PostNet Loss: {:.4f},attn_binarization_loss:{:.4f};".format(
                        m_l, m_p_l,attn_kl_l)
                    str3 = "Current Learning Rate is {:.6f}.".format(
                        scheduled_optim.get_learning_rate())
                    str4 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                        (Now - Start), (total_step - iteration) * np.mean(Time))

                    print("\n" + str1)
                    print(str2)
                    print(str3)
                    print(str4)

                    with open(os.path.join(hp.logger_path, "logger.txt"), "a") as f_logger:
                        f_logger.write(str1 + "\n")
                        f_logger.write(str2 + "\n")
                        f_logger.write(str3 + "\n")
                        f_logger.write(str4 + "\n")
                        f_logger.write("\n")

                    # log attn fig
                    audioname = os.path.basename(audiopaths[0])

                    tensorboard_logger.add_image(
                        'train/attention_weights(align_soft)',
                        plot_alignment_to_numpy(
                            outputs['attn_soft'][0, 0].data.cpu().numpy().T, title=audioname),
                        iteration, dataformats='HWC')
                    tensorboard_logger.add_image(
                        'train/attention_weights_mas(align_hard)',
                        plot_alignment_to_numpy(
                            outputs['attn'][0, 0].data.cpu().numpy().T, title=audioname),
                        iteration, dataformats='HWC')
 


                if iteration % hp.save_step == 0:
                    torch.save({'model': model.state_dict(), 
                                'optimizer': optimizer.state_dict(),
                                'learning_rate': scheduled_optim.get_learning_rate(),
                                'epoch': epoch
                                }, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % iteration))
                    print("save model at step %d ..." % iteration)

                # val
                # if iteration % hp.val_step == 0:
                #     val_loss_outputs = compute_validation_loss(
                #             iteration, model, text2vec_loss, valset,
                #             attention_kl_loss=attention_kl_loss,
                #             logger=tensorboard_logger)
                #     print('Validation loss:', val_loss_outputs)

                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)
                
                iteration+=1
    
    global val_error_batch_num
    with open(os.path.join(hp.logger_path, "error_num.txt"), "a") as f_logger:
        print(error_batch_num,file=f_logger)
        print(val_error_batch_num,file=f_logger)

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--frozen_learning_rate', type=bool, default=False)
    parser.add_argument("--learning_rate_frozen", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
