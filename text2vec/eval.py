import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import time
import shutil
import os

import hparams as hp
from dataset import get_attention_prior

import text
import model as M
import pdb
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_DNN(args,num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    # model = nn.DataParallel(M.FastSpeech()).to(device)
    model = M.Text2Vec().to(device)
    
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_path,
                                                  checkpoint_path))['model'])
    model.eval()
    return model


def synthesis(model, text, wv_feat,alpha=1.0):

    # pdb.set_trace()
    
    text = np.array(phn)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).cuda().long()
    src_pos = torch.from_numpy(src_pos).cuda().long()
    
    in_len=np.stack([text.shape[1]])
    in_len=torch.from_numpy(in_len).cuda().long()

    out_len=np.stack([wv_feat.shape[1]])
    out_len=torch.from_numpy(out_len).cuda().long()
    attn_prior = get_attention_prior(
            text.shape[1], wv_feat.shape[1]).unsqueeze(0).float().cuda()
    
    wv_feat=wv_feat.cuda()
    wv_feat_pos=torch.Tensor([i + 1 for i in range(int(wv_feat.shape[1]))]).unsqueeze(0).long().cuda()
    with torch.no_grad():
        output = model.forward(sequence, src_pos,wav_feat=wv_feat, 
                               in_lens=in_len,out_lens=out_len,
                               WVF_pos=wv_feat_pos,
                               alpha=alpha,attn_prior=attn_prior)
    # return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)
    # print(output['duration_predictor_output'])
    return output['feat_output'].cpu().numpy(),output['feat_postnet_output'].cpu().numpy() 


def get_data():
    test1 = "铁血共和"
    test2 = "大埔的中学有什么"
    test3 = "他将体奥动力称为中国体育产业的标王"
    test4 = "黛比需要一个你这样的自然疗愈者"
    test5 = "四百九十零二千六百一十三"
    test6 = "影片原定于二零一五年二月六日在北美上映"
    data_list = list()
    data_list.append(text.text_to_sequence(test1))
    data_list.append(text.text_to_sequence(test2))
    data_list.append(text.text_to_sequence(test3))
    data_list.append(text.text_to_sequence(test4))
    data_list.append(text.text_to_sequence(test5))
    data_list.append(text.text_to_sequence(test6))
    return data_list

def get_npy():
    data_list=list()
    data_list.append("train/SSB0935/SSB09350169.npy")# 铁血共和
    data_list.append("train/SSB0784/SSB07840032.npy")# 大埔的中学有什么
    data_list.append("train/SSB0016/SSB00160481.npy")
    data_list.append("train/SSB0623/SSB06230243.npy")
    data_list.append("train/SSB0380/SSB03800100.npy")
    data_list.append("train/SSB0316/SSB03160060.npy")
    return data_list

if __name__ == "__main__":
    # Test
    # duration_predictor_output=torch.Tensor([[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,10]])
    # tm=torch.sum(duration_predictor_output, -1)
    # expand_max_len = torch.max(
    #             torch.sum(duration_predictor_output, -1), -1)[0]
    # print(tm)
    # print(expand_max_len)
    # exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=30000)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--log_seed", type=str, default="30_30_nospk_7000")
    parser.add_argument("--checkpoint_path", type=str, default="./run/30_30_nospk_7000/model_new")
    parser.add_argument("--feat_ground_truth", type=str, default="/data_mnt/aishell3/w2v_feat/")

    
    args = parser.parse_args()

    model = get_DNN(args,args.step)
    data_list = get_data()
    npy_list=get_npy()

    if not os.path.exists("results"):
            os.mkdir("results")
            
    os.makedirs(os.path.join("results/",str(args.log_seed)),exist_ok=True)
    for i, phn in enumerate(data_list):
        gt=np.load(os.path.join(args.feat_ground_truth,npy_list[i]))
        gt=torch.from_numpy(gt)
        feat,feat_postnet = synthesis(model, phn,gt, args.alpha)
        gt=gt.numpy()
        

        np.save('results/'+str(args.log_seed)+"/"+str(args.step)+"_"+str(i)+"_feat",feat)
        np.save('results/'+str(args.log_seed)+"/"+str(args.step)+"_"+str(i)+"_feat_postnet",feat_postnet)
        # print(feat)
        # print(feat_postnet)
        # print("gt-shape:",gt.shape)
        # print("predict-shape:",feat.shape)
        # pdb.set_trace()

        # print("Done", i + 1)

    exit()
    s_t = time.perf_counter()
    for i in range(100):
        for _, phn in enumerate(data_list):
            _, _, = synthesis(model, phn, args.alpha)
        print(i)
    e_t = time.perf_counter()
    print("average time:",(e_t - s_t) / 100.)
