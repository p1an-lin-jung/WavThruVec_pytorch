

def get_vocab(vocab_path):
    with open(vocab_path,'r',encoding='utf-8')as fr:
        symbols = fr.readline()
    return symbols

# wav2vec feat
n_feat_dim = 768 # wav2vec 2.0 feature's dim
text_cleaners = ['english_cleaners']
betabinom_cache_path='./data/align_prior'
betabinom_scaling_factor=1.0
use_attn_prior_masking=True


# ecapa_tdnn config
spk_channel= 1024 #Channel size for the speaker encoder
n_speaker_dim=192
input_wav= False # Text2vec use [wav2vec 2.0 feature] as ECAPA_TDNN input, while Vec2wav use [raw wav].
use_multi_speaker_condition= True


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
checkpoint_path = "./run/model_new"
logger_path = "./run/logger"
feat_ground_truth = "/data_mnt/aishell3/w2v_feat/train/"
alignment_path = "/data_mnt/aishell3/enc_align/train"

train_list='./data/enc_train.txt' # modify
val_list=''
vocab_path='./data/vocab.txt'
symbols=get_vocab(vocab_path)
vocab_size=len(symbols)


batch_size = 16
epochs = 2000
n_warm_up_step = 4000



# LAMB optimizer
learning_rate = 0.1
beta1=0.9
beta2=0.98
epsilon=1e-9
weight_decay = 1e-6
grad_clip_thresh = 1.0
decay_step = [500000, 1000000, 2000000]

save_step = 3000
log_step = 5
clear_Time = 20

batch_expand_size = 16

# attn training , according to rad-tts
binarization_start_iter=0 # must be 0
kl_loss_start_iter= 18000
learn_alignments=True
binarization_loss_weight=1.0
use_speaker_emb_for_alignment=True


