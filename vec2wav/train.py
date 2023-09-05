import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from dataset import MelDataset, mel_spectrogram, get_dataset_filelist,collate_fn_tensor
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, \
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

import hparams as hp

torch.backends.cudnn.benchmark = True


# def parse_data_from_batch(batch):
#     wv_feat,spk_emb,mel,audio,filename,mel_loss=batch
#     wv_feat=wv_feat.cuda()
#     spk_emb=spk_emb.cuda()
#     mel=mel.cuda()
#     audio=audio.cuda()
#     mel_loss=mel_loss.cuda()
#     return (wv_feat,spk_emb,mel,audio,filename,mel_loss)

def prepare_output_folders_and_logger(output_directory):
    # Get shared output_directory ready
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    output_hparams_path = os.path.join(output_directory, hp.log_seed,'hparams.py') # run/{seed}/hparams.py
    print("saving current configuration in output dir")

    cmd='cp vec2wav/hparams.py {}'.format(output_hparams_path)
    print(cmd)
    os.system(cmd) # copy the current config

    tboard_out_path = hp.tensorboard_logs_path# run/{seed}/tb_logs
    print("setting up tboard log in %s" % (tboard_out_path))

    logger = SummaryWriter(tboard_out_path)
    return logger


def train(rank, a):
    if hp.num_gpus > 1:
        init_process_group(backend=hp.dist_config['dist_backend'], init_method=hp.dist_config['dist_url'],
                           world_size=hp.dist_config['world_size'] * hp.num_gpus, rank=rank)

    torch.cuda.manual_seed(hp.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(hp).to(device)
    mpd = MultiPeriodDiscriminator(hp).to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        # print(generator)
        os.makedirs(hp.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", hp.checkpoint_path)

    if os.path.isdir(hp.checkpoint_path):
        cp_g = scan_checkpoint(hp.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(hp.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if hp.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), hp.learning_rate, betas=[hp.adam_b1, hp.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                hp.learning_rate, betas=[hp.adam_b1, hp.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hp.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hp.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(hp.input_training_file,hp.input_validation_file)

    trainset = MelDataset(training_filelist, hp.segment_size, hp.n_fft, hp.num_mels,
                          hp.hop_size, hp.win_size, hp.sampling_rate, hp.fmin, hp.fmax, n_cache_reuse=0,
                          shuffle=False if hp.num_gpus > 1 else True, fmax_loss=hp.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    train_sampler = DistributedSampler(trainset) if hp.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=hp.num_workers, shuffle=True,
                              sampler=train_sampler,
                              batch_size=hp.batch_size,
                              collate_fn=collate_fn_tensor,
                              pin_memory=True,
                              drop_last=True
                              )

    if rank == 0:
        validset = MelDataset(validation_filelist, hp.segment_size, hp.n_fft, hp.num_mels,
                              hp.hop_size, hp.win_size, hp.sampling_rate, hp.fmin, hp.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=hp.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir)
        validation_loader = DataLoader(validset, num_workers=hp.num_workers, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True,
                                       collate_fn=collate_fn_tensor)
        os.makedirs(hp.logger_path,exist_ok=True)
        sw = prepare_output_folders_and_logger(hp.run_path)

    generator.train()
    mpd.train()
    msd.train()
    import pdb
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if hp.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            
            wv_feat,spk_emb,x,y,_,y_mel = batch

            # 16, 80, 32 bz chan,frame
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            wv_feat=wv_feat.to(device)
            spk_emb=spk_emb.to(device)
        
            noise= torch.randn(hp.batch_size,hp.noise_dim)
            noise= noise.to(device)
            y_g_hat = generator(wv_feat,spk_emb,noise)

            y_=y
            y=y[:,:,:y_g_hat.shape[2]]
            
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), hp.n_fft, hp.num_mels, hp.sampling_rate, 
                                          hp.hop_size,hp.win_size,hp.fmin, hp.fmax_for_loss)
            y_g_hat_mel = y_g_hat_mel.permute(0,2,1)
            
            y_mel=y_mel[:,:y_g_hat_mel.shape[1],:]
            optim_d.zero_grad()

            # if y_g_hat_mel.shape[1]<y_mel.shape[1]:
            #     continue        
            # else:
            # print(y_g_hat_mel.shape[1],y_mel.shape[1])
            # print(y_g_hat.shape[2]-y.shape[2])
            # continue

            
            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, mel_error, time.time() - start_b))

                # checkpointing
                if steps % hp.save_step == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(hp.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if hp.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(hp.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'mpd': (mpd.module if hp.num_gpus > 1
                                             else mpd).state_dict(),
                                     'msd': (msd.module if hp.num_gpus > 1
                                             else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % hp.log_step == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                if steps % hp.val_step == 0 and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():

                        for j, batch in enumerate(validation_loader):

                            wv_feat,spk_emb,x,y,_,y_mel = batch

                            wv_feat=wv_feat.to(device)
                            spk_emb=spk_emb.to(device)
                        
                            noise= torch.randn(1,hp.noise_dim)
                            noise= noise.to(device)
                            pdb.set_trace()
                            y_g_hat = generator(wv_feat,spk_emb,noise)


                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), hp.n_fft, hp.num_mels, hp.sampling_rate,
                                                          hp.hop_size, hp.win_size,
                                                          hp.fmin, hp.fmax_for_loss)
                            y_g_hat_mel = y_g_hat_mel.permute(0,2,1)
                            
                            print(y_mel.shape)
                            print(y_g_hat_mel.shape)
                            y_mel=y_mel[:,:y_g_hat_mel.shape[1],:]
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, hp.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, hp.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), hp.n_fft, hp.num_mels,
                                                             hp.sampling_rate, hp.hop_size, hp.win_size,
                                                             hp.fmin, hp.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--config', default='config_v1.json')
    parser.add_argument('--training_epochs', default=100, type=int)
    parser.add_argument('--stdout_interval', default=50, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()


    torch.manual_seed(hp.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.seed)
        hp.num_gpus = torch.cuda.device_count()
        hp.batch_size = int(hp.batch_size / hp.num_gpus)
        print('Batch size per GPU :', hp.batch_size)
    else:
        pass

    # if h.num_gpus > 1:
    #     mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    # else:
    train(0, a)


if __name__ == '__main__':
    main()
