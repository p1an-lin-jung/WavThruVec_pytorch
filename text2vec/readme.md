


fastspeech的编码器，作为主体框架

glow-tts的单调对齐搜索

rad-tts的硬对齐-软对齐训练（使用KL散度优化）


hifi-gan作为解码器（只修改部分参数）

ecapa-tdnn作为说话人embedding的编码器（）



编码器原文：

The first-stage component of our pipeline mostly follows the
FastSpeech [5] architecture with two blocks of Feed-Forward
Transformers (FFT) consisting of a self-attention and 1D con-
volutional network (Figure 2A). 
**编码器主要遵循fastspeech，使用FFT模块**

Instead of using a teacher-based length regulator between the FFT blocks as in the origi-
nal work, we utilize unsupervised Monotonic Alignment Search
introduced by [7]. 

We specifically train soft and hard alignments with additional diagonal prior as in [24]. The soft align-
ment matrix Asof t ∈ RN×T is based on the learned pair-
wise affinity between all text tokens φ ∈ Φ and WAV2VEC
2.0 activations x ∈ X of lengths N and T respectively.


The forward-backward algorithm is used to maximize the likelihood
P (st = φ | xt), where st is a random variable for a text to-
ken aligned at timestep t with target xt. To obtain a binary
alignment map Ahard, the Viterbi algorithm is used, and to fur-
ther close the gap between soft and hard distributions, their KL-
divergence is minimized: Lbin = Ahard （曼哈德积） log Asof t. 
**rad-tts使用随机变量生成先验对齐，然后使用一个attn层学习soft对齐，并将soft转为(MAS算法)hard,增加了一个KL散度*w到loss当中**


Hard alignment serves as a target for the duration predictor that is
trained via Mean Squared Error loss (MSE) to be used at infer-
ence time. 
**硬对齐作为持续时间预测器的目标（即fastspeech中使用tacotron2（MFA）的对齐作为duration p的输入，而这里使用MAS的对齐作为输入）**


Similarly, the model optimizes MSE between predicted and target speech representation. 
**类似的，模型使用MSE优化标签与目标的representation**

For the multi-speaker setup, we condition the first FFT block on the speaker embed-
ding that is obtained through feeding the target sequence into a series of convolution layers followed by channel-dependent
frame attention pooling [25]. 
**使用ecapa-tdnn作为spk emb的提取器，按照multispeaker，使用concat的方法连接，一般是连接在time维度上：**
```angular2html
def cat_speaker_emb(speaker_emb: Tensor, x: Tensor) -> Tensor:
    """Concat the speaker embedding to the prenet/encoder results.

    Args:
        speaker_emb (Tensor): The speaker embedding of dimension [B, 1, E]
        x (Tensor): The results to be concatenated with of shape [B, M, E]

    Returns:
        Tensor: The concatenated speaker embedding with the input x of of shape
        [B, M + 1, E]
    """
    return torch.cat([speaker_emb, x], dim=1)

```

Such an encoder is supposed to capture the style of a particular speaker with regards to some
prosody features that are represented in WAV2VEC 2.0 latent
variables. It can be used at inference time to produce speaker
embedding in a zero-shot manner.

** 既然使用了spk emb，那么就要修改fastspeech的输入维度 **



解码器原文：

The role of the second-stage component is to generate an audio waveform conditioned on hidden activations of WAV2VEC
2.0 (Figure 2B). 
**第二个组件使用wav2vec作为训练条件**

Vec2wav is a Generative Adversarial Network,based on the HiFi-GAN [13], consisting of a fully convolutional
generator and several sub-discriminators.
**vec2wav 是生成对抗网络，基于hifi-gan，包括了一系列的全卷积生成器核几个子判别器**


The generator upsamples input features through the sequence of transposed convolutions followed by residual blocks of dilated convolutions. Similarly to [21], we introduce Conditional Batch Normalization to condition the network on the speaker embedding between the residual blocks at different temporal resolutions. 
**生成器对输入的feat上采样，上采样模块使用【21】，即gan-tts里的CBN，整合spk embedding**

Each Conditional Batch Normalization is preceded by a linear network that
takes the speaker embedding concatenated with a vector of random numbers from a normal distribution. 

**每个cbn之前有一个线性网络，将spk emb和正态分布的随机向量（即噪声）拼接**

We synthesize speech at a sampling rate of 32 kHz while our input features have temporal resolution of 50 Hz, resulting in 640x upsampling factor,
compared to 256x of original HiFi-GAN. Therefore the configuration of the generator was changed for upsample rates to a
sequence of (5, 4, 4, 2, 2, 2) with corresponding kernel sizes(11, 8, 8, 4, 4, 4), while the hyper-parameters of residual blocks
are the same as in HiFi-GAN V1. 
**生成的语音，采样率在32k，用640*的上采样因子，将50hz分辨率的input feat采样到32k。**


Additional multi-period sub-discriminators are added with periods of [13, 17, 19] to obtain
the receptive field of similar length. To enable multi-speaker
capabilities, we do not use learnable embeddings through a
look-up table, but rather train a speaker encoder that takes mel-
spectrogram of a particular sample as an input and produces
fixed-length embedding. 
**额外的多周期判别器使用 13 17 19 作为周期，并使用一个spk ecd，使用mel特征，提取样本里的说话人嵌入**
Specifically ECAPA-TDNN [25] architecture is used as a speaker encoder.



训练：

Text2vec is trained using the LAMB optimizer with learning
rate of 0.1, β1 = 0.9, β2 = 0.98, ? = 10−9, similarly to [6].
We follow the training schedule of [24], to add the binarization
term and hard alignments to the loss function.
The discriminators and the generator of the GAN-based
vec2wav are trained as in [13], using the AdamW optimizer
with β1 = 0.8, β2 = 0.99, weight decay λ = 0.01 and learn-
ing rate decaying by a 0.999 factor in every epoch with an ini-
tial value of 2·10−4. Our intermediate representation is already
aligned, so we do not have to incorporate dynamic time warping
to relax alignment constraint in the spectrogram prediction loss
as in [17]. However, we linearly decay its weight coefficient to
make the loss function increasingly dependent on the GAN ob-
jective. Similarly to [16, 17], we adopt the windowed generator
training with a training window of 0,64 s.
Both text2vec and vec2wav were trained on 4 NVIDIA
V100 GPUs with batch sizes of 32 and 24, respectively. Af-
ter 800k iterations of pretraining, both models are finetuned for
80k iterations on VCTK dataset with a 10-fold lower value of
initial learning rate.