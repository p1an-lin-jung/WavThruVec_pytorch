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

wavs_path='/data_mnt/aishell3/train/wav'

feat_output_path="/data_mnt/aishell3/w2v_feat/train/"
align_path="/data_mnt/aishell3/enc_align/train/"
enc_filelist_path='./data/enc_train.txt'
dec_filelist_path='./data/dec_train.txt'

model_path = "/data_mnt/wav2vec" #wav2vec_base.pt must in model_path

os.makedirs(feat_output_path,exist_ok=True)
os.makedirs(align_path,exist_ok=True)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = Wav2Vec2Model.from_pretrained(model_path)

# for pretrain: Wav2Vec2ForPreTraining
# model = Wav2Vec2ForPreTraining.from_pretrained(model_path)

model = model.to(device)
model = model.half()
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


# with open(enc_filelist_path,'w',encoding='utf-8')as fw:
#
#     for spk in os.listdir(wavs_path)[:20]:
#         spk_path=os.path.join(wavs_path,spk)  #/data_mnt/aishell3/train/wav
#         out_spk_path=os.path.join(feat_output_path,spk) # /data_mnt/aishell3/w2v_feat/train/SSB0005
#
#         os.makedirs(out_spk_path,exist_ok=True)
#
#         for _wav in os.listdir(spk_path)[:20]:
#
#             file_path=os.path.join(spk_path,_wav) #/data_mnt/aishell3/train/wav/SSB0005/file.wav
#
#             # wav, sr = sf.read(wav_path)
#             wav,sr=librosa.load(file_path,sr=16000)
#             input_values = feature_extractor(wav, return_tensors="pt",sampling_rate = sr).input_values
#             input_values = input_values.half()
#             input_values = input_values.to(device)
#
#             with torch.no_grad():
#                 outputs = model(input_values)
#                 last_hidden_state = outputs.last_hidden_state
#                 npy_file_name=_wav[:-4]+'.npy'
#
#                 same_suffix=os.path.join(spk,npy_file_name)
#                 np_path=os.path.join(feat_output_path,same_suffix) #/data_mnt/aishell3/w2v_feat/train/SSB0005/file.wav
#                 np.save(np_path ,last_hidden_state.cpu().numpy())
#                 print("shape:",last_hidden_state.shape)
#                 print("{}|{}|{}".format(same_suffix,label_dict[_wav],spk),file=fw)
#
#     print(last_hidden_state)


vocab_path='./data/vocab.txt'
def build_vocab(vocab_path,label_dict):
    # vocab = "PE abcdefghijklmnopqrstuvwxyz0123456789.?"
    char_dict=set()
    char_dict.add('P')# pad
    char_dict.add('E')# end
    char_dict.add(' ')

    for k,v in label_dict.items():
        for ch in v:
            char_dict.add(ch)
    with open(vocab_path,'w',encoding='utf-8')as fw:
        for ch in char_dict:
            fw.write(ch)

build_vocab(vocab_path,label_dict)