import os
# This is from hifi-gan config_v1, I change the JSON file to .py file because JSON doesn’t support comment.

# path:
run_path='./run_dec'
log_seed='30_30'
tensorboard_logs_path=os.path.join(run_path,log_seed,"tb_logs")
checkpoint_path = os.path.join(run_path,log_seed,"model_new")
logger_path = os.path.join(run_path,log_seed,"logger")
feat_ground_truth = "/data_mnt/aishell3/w2v_feat/"
train_wav_path = "/data_mnt/aishell3/"
spk_emb_path = "/data_mnt/aishell3/spk_emb/"


input_training_file='./data/enc_train_full.txt' # ,'./data/enc_train_extd  .txt'modify
input_validation_file='./data/enc_val_full.txt'

save_step = 5000
log_step = 1000
val_step = 100000
clear_Time = 20


# vec2wav:
n_feat_dim = 1024 # wav2vec 2.0 feature's dim
spk_dim=192
noise_dim=192

# hifi-gan
resblock= 1
num_gpus= 1
batch_size= 2
learning_rate= 0.0002
adam_b1= 0.8
adam_b2= 0.99
lr_decay= 0.999
seed= 1234

# generator
upsample_rates= [5,4,4,2,2] #[5,4,4,2,2,2] # 5*4*4*2*2*2=640
upsample_kernel_sizes= [11,8,8,4,4]#[11,8,8,4,4,4]
upsample_initial_channel= 512
resblock_kernel_sizes= [3,7,11]
resblock_dilation_sizes= [[1,3,5], [1,3,5], [1,3,5]]

# MultiPeriodDiscriminator
periods=[13,17,19]

segment_size= 8192
num_mels= 80
num_wv_feat=1024
num_freq= 1025
n_fft= 1024
hop_size= 256
win_size= 1024

sampling_rate= 16000

fmin= 0
fmax= 8000
fmax_for_loss= None

num_workers= 0

dist_config= {
    "dist_backend": "nccl",
    "dist_url": "tcp://localhost:54321",
    "world_size": 1
}

