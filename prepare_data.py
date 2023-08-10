# generate wav2vec2.0 feat and filelist, vocab_txt

import os
import torch
# import torch.nn.functional as F
# import soundfile as sf
import numpy as np
import librosa
import re
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    Wav2Vec2Model,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wavs_path='/data_mnt/aishell3/train/wav' # this dir place the aishell3 speaker dirs, and speaker dirs contain the corresponding wav files

feat_output_path="/data_mnt/aishell3/w2v_feat/train/" #this dir place the output feature file
# align_path="/data_mnt/aishell3/enc_align/train/"

enc_train_list_path='./data/enc_train.txt' # this file contains the train list, every line follow this format: {spk/audio.npy}|{text}|{spk}
enc_val_list_path='./data/enc_val.txt'# this file contains the val list, every line follow this format: {spk/audio.npy}|{text}|{spk}
dec_filelist_path='./data/dec_train.txt'

model_path = "/data_mnt/wav2vec" #wav2vec_base.pt must in model_path

os.makedirs(feat_output_path,exist_ok=True)
# os.makedirs(align_path,exist_ok=True)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = Wav2Vec2Model.from_pretrained(model_path)

# for pretrain: Wav2Vec2ForPreTraining
# model = Wav2Vec2ForPreTraining.from_pretrained(model_path)

model = model.to(device)
# model = model.half()
model.eval()


label_dict={}
label_file_path='/data_mnt/aishell3/train/content.txt'
vocab = "PE abcdefghijklmnopqrstuvwxyz0123456789.?"
with open(label_file_path,'r',encoding='utf-8')as f:
    # fw=open('zh_label.txt','w',encoding='utf-8')
    for line in f.readlines():
        _path,text=line.strip().split('\t')
        dir=_path[:7]
        file_path=os.path.join(dir,_path)
        text = text.lower()
        text = re.sub("[{}]".format(vocab), " ", text)
        text = re.sub("[ ]+", "", text)
        text=text.strip()
        # print("{}|{}|{}".format(file_path,text,dir),file=fw)
        label_dict[_path]=text
    # fw.close()


fw_train =open(enc_train_list_path,'w',encoding='utf-8')
fw_val = open(enc_val_list_path,'w',encoding='utf-8')

for spk in os.listdir(wavs_path)[:15]:# only take 15 spk, as an example
    spk_path=os.path.join(wavs_path,spk)  #/data_mnt/aishell3/train/wav
    out_spk_path=os.path.join(feat_output_path,spk) # /data_mnt/aishell3/w2v_feat/train/SSB0005

    os.makedirs(out_spk_path,exist_ok=True)

    for ind,_wav in enumerate(os.listdir(spk_path)[:40]): # only take 40 file, half for train and half for val

        file_path=os.path.join(spk_path,_wav) #/data_mnt/aishell3/train/wav/SSB0005/file.wav

        # wav, sr = sf.read(wav_path)
        wav,sr=librosa.load(file_path,sr=16000)
        input_values = feature_extractor(wav, return_tensors="pt",sampling_rate = sr).input_values
        # input_values = input_values.half()
        input_values = input_values.to(device)

        with torch.no_grad():
            outputs = model(input_values)
            last_hidden_state = outputs.last_hidden_state
            npy_file_name=_wav[:-4]+'.npy'

            same_suffix=os.path.join(spk,npy_file_name)
            np_path=os.path.join(feat_output_path,same_suffix) #/data_mnt/aishell3/w2v_feat/train/SSB0005/file.wav
            np.save(np_path ,last_hidden_state.cpu().numpy())
            # print("shape:",last_hidden_state.shape)

            if ind%2==0:
                print("{}|{}|{}".format(same_suffix,label_dict[_wav],spk),file=fw_train)
            else:
                print("{}|{}|{}".format(same_suffix,label_dict[_wav],spk),file=fw_val)

print(last_hidden_state)

fw_val.close()
fw_train.close()

vocab_path='./data/vocab.txt'
def build_vocab(vocab_path,label_dict):
    simple_vocab = "PE "
    char_dict=set()

    for ch in simple_vocab:
        char_dict.add(ch)

    for k,v in label_dict.items():
        for ch in v:
            char_dict.add(ch)
    with open(vocab_path,'w',encoding='utf-8')as fw:
        for ch in char_dict:
            fw.write(ch)

build_vocab(vocab_path,label_dict)