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
from dataset import BufferDataset, DataLoader
from dataset import get_data_to_buffer, collate_fn_tensor
from optimizer import ScheduledOptim
import hparams as hp
import utils
from loss import  AttentionBinarizationLoss
from torch_optimizer import Lamb
from torch.utils.tensorboard import SummaryWriter

def prepare_output_folders_and_logger(output_directory):
    # Get shared output_directory ready
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    output_hparams_path = os.path.join(output_directory, 'hparams.py') # run/
    print("saving current configuration in output dir")

    os.system('cp ./hparams.py {}' % (output_hparams_path))

    tboard_out_path = os.path.join(output_directory, 'tb_logs')
    print("setting up tboard log in %s" % (tboard_out_path))
    logger = SummaryWriter(tboard_out_path)
    return logger

def parse_data_from_batch(batch):
 
    feat_target = batch['feat_target']
    text = batch['text']

    in_lens = batch['input_lengths']
    out_lens = batch['output_lengths']

    feat_pos=batch['feat_pos']
    src_pos = batch["src_pos"]
    max_feat_len=batch["feat_max_len"]

    attn_prior = batch['attn_prior']
    audiopaths = batch['audiopaths']


    if attn_prior is not None:
        attn_prior = attn_prior.cuda()

    feat_target = feat_target.cuda()
    text = text.cuda()
    in_lens, out_lens = in_lens.cuda(), out_lens.cuda()

    feat_pos=feat_pos.cuda()
    src_pos=src_pos.cuda()


    return (feat_target, text,
            in_lens, out_lens,
            feat_pos,src_pos,max_feat_len,
            attn_prior, audiopaths)


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

    # Get dataset
    dataset = BufferDataset(buffer)

    # Get Training Loader

    training_loader = DataLoader(dataset,
                                 batch_size=hp.batch_expand_size * hp.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn_tensor,
                                 drop_last=True,
                                 num_workers=0)

    # Optimizer and loss
    # Note:Change Adam(fastspeech) to Lamb
    optimizer = Lamb(model.parameters(),
                                 lr=hp.learning_rate,
                                 betas = (hp.beta1, hp.beta2),
                                 eps = hp.epsilon,
                                 weight_decay = hp.weight_decay)
    scheduled_optim = ScheduledOptim(optimizer,
                           hp.decoder_dim,
                           hp.n_warm_up_step,
                           args.restore_step)

    loss_weights={
        "binarization_loss_weight": hp.binarization_loss_weight,
    }

    text2vec_loss = DNNLoss().to(device)
    attention_kl_loss = AttentionBinarizationLoss()

    print("Defined Optimizer and Loss Function.")

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step %d---\n" % args.restore_step)
    except:
        print("\n---Start New Training---\n")

        os.makedirs(hp.checkpoint_path,exist_ok=True)

    # Init logger
    os.makedirs(hp.logger_path,exist_ok=True)


    total_step = hp.epochs * len(training_loader) * hp.batch_expand_size

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()

    # Training
    model = model.train()
    iteration = 0
    for epoch in range(hp.epochs):
        for i, batchs in enumerate(training_loader):
            # real batch start here
            for j, batch in enumerate(batchs):
                start_time = time.perf_counter()

                # Init
                scheduled_optim.zero_grad()

                # Get Data，这里要把duration替换掉
                (wv_feat_target, text,
                 in_lens, out_lens,
                 wv_feat_pos, text_pos, max_wv_feat_len,
                 attn_prior, audiopaths) = parse_data_from_batch(batch)

                # • [0, 6k): Use A_soft for the alignment matrix.
                # • [6k, 18k): start using Viterbi A_hard instead of A_soft,
                # • [18k, end): add binarization term λ2L_bin to the loss.

                if iteration >= hp.binarization_start_iter:
                    binarize = True  # binarization training phase
                else:
                    binarize = False  # no binarization, soft alignments only

                outputs = model(
                    wv_feat_target,
                    text,
                    text_pos,
                    in_lens,
                    out_lens,
                    mel_pos=wv_feat_pos,
                    mel_max_length=max_wv_feat_len,
                    binarize_attention=binarize,
                    attn_prior=attn_prior,
                   )

                loss_outputs={}



                #compute the MSE between: 1. gt-feat and predicated feat
                #                         2. duration from hard-attn and duration from length_regulator
                mel_loss, mel_postnet_loss, duration_loss = text2vec_loss(outputs['feat_output'],
                                                                          outputs['feat_postnet_output'],
                                                                          outputs['duration_predictor_output'],
                                                                          wv_feat_target,
                                                                          outputs['duration'])

                total_loss = mel_loss + mel_postnet_loss + duration_loss
                w_bin = loss_weights.get('binarization_loss_weight', 1.0)
                if binarize and iteration >= hp.kl_loss_start_iter:
                    binarization_loss = attention_kl_loss(
                        outputs['attn'], outputs['attn_soft'])
                    total_loss += binarization_loss * w_bin
                else:
                    binarization_loss = torch.zeros_like(total_loss)
                loss_outputs['binarization_loss'] = (binarization_loss, w_bin)

                # Logger
                t_l = total_loss.item()
                m_l = mel_loss.item()
                m_p_l = mel_postnet_loss.item()
                d_l = duration_loss.item()
                attn_kl_l=binarization_loss.item()

                with open(os.path.join(hp.logger_path, "total_loss.txt"), "a") as f_total_loss:
                    f_total_loss.write(str(t_l) + "\n")

                with open(os.path.join(hp.logger_path, "mel_loss.txt"), "a") as f_mel_loss:
                    f_mel_loss.write(str(m_l) + "\n")

                with open(os.path.join(hp.logger_path, "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
                    f_mel_postnet_loss.write(str(m_p_l) + "\n")

                with open(os.path.join(hp.logger_path, "duration_loss.txt"), "a") as f_d_loss:
                    f_d_loss.write(str(d_l) + "\n")
                with open(os.path.join(hp.logger_path, "attn_kl_loss.txt"), "a") as f_attn_kl_loss:
                    f_attn_kl_loss.write(str(attn_kl_l) + "\n")

                # Backward
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), hp.grad_clip_thresh)

                # Update weights
                if args.frozen_learning_rate:
                    scheduled_optim.step_and_update_lr_frozen(
                        args.learning_rate_frozen)
                else:
                    scheduled_optim.step_and_update_lr()

                # Print
                if iteration % hp.log_step == 0:
                    Now = time.perf_counter()

                    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                        epoch + 1, hp.epochs, iteration, total_step)
                    str2 = "Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f};".format(
                        m_l, m_p_l, d_l)
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

                if iteration % hp.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % iteration))
                    print("save model at step %d ..." % iteration)

                # todo
                # val
                # if iteration > -1 and iteration % hp.val_step == 0:
                #     val_loss_outputs = compute_validation_loss(
                #             iteration, model, criterion, valset, collate_fn,
                #             batch_size, n_gpus, logger=logger,
                #             train_config=train_config)
                #     print('Validation loss:', val_loss_outputs)
                #     with open(os.path.join(hp.logger_path, "attn_kl_loss.txt"), "a") as f_attn_kl_loss:
                #         f_attn_kl_loss.write(str(val_loss_outputs.item()) + "\n")


                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)

                iteration+=1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--frozen_learning_rate', type=bool, default=False)
    parser.add_argument("--learning_rate_frozen", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
