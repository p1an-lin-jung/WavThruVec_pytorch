
import os
def get_vocab(vocab_path):
    with open(vocab_path,'r',encoding='utf-8')as fr:
        symbols = fr.readline()
    return symbols

# wav2vec feat
n_feat_dim = 1024 # wav2vec 2.0 feature's dim
text_cleaners = ['english_cleaners']
betabinom_cache_path='./data/align_prior'
betabinom_scaling_factor=1.0
use_attn_prior_masking=True


# ecapa_tdnn config
spk_channel = 1024 #Channel size for the speaker encoder
n_speaker_dim = 192
n_speakers = 200
input_wav= False # Text2vec use [wav2vec 2.0 feature] as ECAPA_TDNN input, while Vec2wav use [raw wav].


# Text2vec config
max_seq_len = 3000

encoder_dim = 256
encoder_n_layer = 1 #4
encoder_head = 2
encoder_conv1d_filter_size = 1024

decoder_dim = 256
decoder_n_layer = 1 #4
decoder_head = 2
decoder_conv1d_filter_size = 1024

fft_conv1d_kernel = (9, 1)
fft_conv1d_padding = (4, 0)

duration_predictor_filter_size = 256
duration_predictor_kernel_size = 3
dropout = 0.1

# Train data
run_path='./run'
log_seed='30_30_spk'
tensorboard_logs_path=os.path.join(run_path,log_seed,"tb_logs")
checkpoint_path = os.path.join(run_path,log_seed,"model_new")
logger_path = os.path.join(run_path,log_seed,"logger")
feat_ground_truth = "/data_mnt/aishell3/w2v_feat/"
# alignment_path = "/data_mnt/aishell3/enc_align/train"

train_list=['./data/enc_train_full.txt'] # ,'./data/enc_train_extd  .txt'modify
val_list=['./data/enc_val_full.txt']
vocab_path='./data/vocab.txt'
symbols=get_vocab(vocab_path)
vocab_size=len(symbols)


batch_size = 16
epochs = 200 # about 800k iters
n_warm_up_step = 4000
batch_expand_size = 16


save_step = 5000
log_step = 1000
val_step = 50000
clear_Time = 20


# LAMB optimizer
learning_rate = 1e-4
beta1=0.9
beta2=0.98
epsilon=1e-9
weight_decay = 1e-6
grad_clip_thresh = 1.0
decay_step = [200000, 400000, 600000]


# attn training , according to rad-tts
binarization_start_iter=0 # must be 0
kl_loss_start_iter= 0
learn_alignments=True
binarization_loss_weight=1.0
use_multi_speaker_condition= True
use_speaker_emb_for_alignment=True


