import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import time
import os


from utils import process_text, pad_1D, pad_2D
from utils import pad_1D_tensor, pad_2D_tensor
from text import text_to_sequence
from tqdm import tqdm
from scipy.stats import betabinom
import hparams as hp
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_text(self, text):
    text = self.tp.encode_text(text)
    text = torch.LongTensor(text)
    return text

def beta_binomial_prior_distribution(phoneme_count, mel_count,
                                     scaling_factor=0.05):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling_factor*i, scaling_factor*(M+1-i)
        rv = betabinom(P - 1, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))#[len_text,n_frame]


def get_attention_prior(n_tokens, n_frames):
    # cache the entire attn_prior by filename
    os.makedirs(hp.betabinom_cache_path,exist_ok=True)
    if hp.use_attn_prior_masking:
        filename = "{}_{}".format(n_tokens, n_frames)
        prior_path = os.path.join(hp.betabinom_cache_path, filename)
        prior_path += "_prior.pth"

        if os.path.exists(prior_path):
            attn_prior = torch.load(prior_path)
        else:
            attn_prior = beta_binomial_prior_distribution(
                n_tokens, n_frames, hp.betabinom_scaling_factor)
            torch.save(attn_prior, prior_path)
    else:
        attn_prior = torch.ones(n_frames, n_tokens)  # all ones baseline

    return attn_prior



def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)
        return txt

def get_data_to_buffer(file_path):
    """
        file_path format:
            example.npy|the text.|spk_id
    """
    if file_path=="":
        return None

    buffer = list()
    text = process_text(file_path)


    start = time.perf_counter()
    for line in tqdm(text):
        npy_file,character,spk=line.strip().split('|')
        feat_gt_name = os.path.join(
            hp.feat_ground_truth,npy_file)
        feat_gt_target = np.load(feat_gt_name)

        text_enc = np.array(
            text_to_sequence(character))

        text_enc = torch.from_numpy(text_enc)

        feat_gt_target = torch.from_numpy(feat_gt_target).squeeze() # [1,n_frame,n_channel]->[n_frame,n_channel]

        attn_prior = get_attention_prior(
            text_enc.shape[0], feat_gt_target.shape[0])

        if not hp.use_attn_prior_masking:
            attn_prior = None

        buffer.append({"text_enc": text_enc,
                       "feat_gt_target": feat_gt_target,
                       "audiopath":feat_gt_name,
                       "attn_prior":attn_prior
        })



    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end - start))

    return buffer


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]


def reprocess_tensor(batch, cut_list):
    import pdb
    # pdb.set_trace()

    texts = [batch[ind]["text_enc"] for ind in cut_list]
    feat_gt_targets = [batch[ind]["feat_gt_target"] for ind in cut_list]


    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i + 1 for i in range(int(length_src_row))],
                              (0, max_len - int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))



    length_feat = np.array(list())
    for feat in feat_gt_targets:
        length_feat = np.append(length_feat, feat.size(0))# [n_frame,n_channel]

    feat_pos = list()
    max_feat_len = int(max(length_feat))
    for length_feat_row in length_feat:
        feat_pos.append(np.pad([i + 1 for i in range(int(length_feat_row))],
                              (0, max_feat_len - int(length_feat_row)), 'constant'))
    feat_pos = torch.from_numpy(np.array(feat_pos))

    # pdb.set_trace()

    input_lengths, _ = torch.sort(
        torch.LongTensor([len(batch[ind]['text_enc']) for ind in cut_list]),
        dim=0, descending=True)

    max_input_len = input_lengths[0]

    attn_prior_padded = torch.FloatTensor(len(cut_list), max_feat_len, max_input_len)# [bz,n_feat,n_txt]
    attn_prior_padded.zero_()
    output_lengths = torch.LongTensor(len(cut_list))
    audiopaths = []

    for i in range(len(input_lengths)):
        ind=cut_list[i]
        W2V = batch[ind]['feat_gt_target']

        output_lengths[i] = W2V.size(0)#
        audiopath = batch[ind]['audiopath']
        audiopaths.append(audiopath)

        cur_attn_prior = batch[ind]['attn_prior']
        if cur_attn_prior is None:
            attn_prior_padded = None
        else:
            try:
                attn_prior_padded[i, :cur_attn_prior.size(0), :cur_attn_prior.size(1)] = cur_attn_prior
            except:
                print(cur_attn_prior.shape)
                print(attn_prior_padded[i].shape)

    texts = pad_1D_tensor(texts)
    feat_gt_targets = pad_2D_tensor(feat_gt_targets)  # []

    # p texts:
    # p text_padded:
    out = {"text": texts, # [bsz,n_text]
           "feat_target": feat_gt_targets,# [bsz,n_frame,n_feat(768)]
           "input_lengths":input_lengths, # [bsz],the text-enc length
           "output_lengths":output_lengths,# [bsz],the w2v_feat length
           "feat_pos": feat_pos,# [bsz,max_feat_len]
           "src_pos": src_pos,# [bsz,max_input_len]
           "feat_max_len": max_feat_len,# int
           "attn_prior":attn_prior_padded,#[bsz,max_feat_len,max_input_len]
           'audiopaths': audiopaths,#[16]
           }

    return out


def collate_fn_tensor(batch):
    len_arr = np.array([d["text_enc"].size(0) for d in batch])
    index_arr = np.argsort(-len_arr) # 已经按text 长度排序
    batchsize = len(batch)
    real_batchsize = batchsize // hp.batch_expand_size

    cut_list = list()
    for i in range(hp.batch_expand_size):
        cut_list.append(index_arr[i * real_batchsize:(i + 1) * real_batchsize])

    output = list()
    for i in range(hp.batch_expand_size):
        output.append(reprocess_tensor(batch, cut_list[i]))

    return output


if __name__ == "__main__":
    # TEST
    get_data_to_buffer()
