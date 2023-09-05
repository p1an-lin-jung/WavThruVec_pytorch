import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import librosa
import hparams as hp
from utils import pad_2D_tensor,pad_1D_tensor
MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    # print(full_path)
    data, sampling_rate = librosa.load(full_path, sr=16000)
    # sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True,return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(input_training_file,input_validation_file):
    with open(input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    with open(input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, split=False, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

    def __getitem__(self, index):
        
        filename = self.audio_files[index]
        #line: train/SSB0544/SSB05440252.npy 
        # print(line)
        
        
        dir_tuple=filename.split('/')
        #wav_file: /data_mnt/aishell3/train/wav/SSB0544/SSB05440252.wav 
        wav_file=os.path.join(hp.train_wav_path,dir_tuple[0],'wav',dir_tuple[1],dir_tuple[2][:-4]+'.wav')
        
        feat_file=os.path.join(hp.feat_ground_truth,filename)
        spk= dir_tuple[1]

        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(wav_file)
            # audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start + self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        wav2vec_ft=torch.from_numpy(np.load(feat_file))
        # wav2vec_ft=wav2vec_ft.permute(0,2,1)# 1,chan,fram
        spk_emb=torch.load(os.path.join(hp.spk_emb_path,spk+'.pth'))

        return (wav2vec_ft.squeeze(), spk_emb.squeeze().squeeze(),
                mel.squeeze(), audio.squeeze(0), filename, 
                mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)



def collate_fn_tensor(batch):
    wav2vec_fts=list()
    spk_embs=list()
    mels=list()
    audios=list()
    filenames=list()
    mel_losses=list()
    for i in range(len(batch)):
        (wav2vec_ft, spk_emb, mel, audio, filename, mel_loss) = batch[i]
        wav2vec_fts.append(wav2vec_ft)
        spk_embs.append(spk_emb)
        mel=mel.transpose(0,1)
        mels.append(mel)
        audios.append(audio)
        filenames.append(filename)
        mel_loss=mel_loss.transpose(0,1)
        mel_losses.append(mel_loss)
        
    wav2vec_fts=pad_2D_tensor(wav2vec_fts)
    wav2vec_fts=wav2vec_fts.permute(0,2,1)
    mels=pad_2D_tensor(mels)

    spk_embs=torch.stack(spk_embs)
    audios=pad_1D_tensor(audios)
    mel_losses=pad_2D_tensor(mel_losses)
    

    return (
        wav2vec_fts, 
        spk_embs,
        mels, 
        audios, 
        filenames, 
        mel_losses,
    )
