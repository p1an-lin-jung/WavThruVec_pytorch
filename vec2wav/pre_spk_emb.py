from speechbrain.pretrained import EncoderClassifier
import os
import librosa
import torch
import numpy as np

def wav_cat(wav_ls):
    res=np.concatenate(wav_ls,axis=0)
    return res


classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
# print(embeddings.shape)#1,1,192


base_dir="/data_mnt/aishell3/train/wav"
spk_emb_dir="/data_mnt/aishell3/spk_emb"
os.makedirs(spk_emb_dir,exist_ok=True)

for spk in os.listdir(base_dir):
    
    spk_dir=os.path.join(base_dir,spk)
    wav_list=[]
    for wav_item in os.listdir(spk_dir)[:50]:
        file_path=os.path.join(spk_dir,wav_item)
        
        wav_data,_=librosa.load(file_path,sr=16000)
        wav_list.append(wav_data)
    res_wav=wav_cat(wav_list)   
    embeddings = classifier.encode_batch(torch.Tensor(res_wav))

    # torch.save()
    torch.save(embeddings,os.path.join(spk_emb_dir,spk+'.pth'))