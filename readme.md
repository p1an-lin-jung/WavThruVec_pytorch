

# WavThruVec Pytorch (unfinished)
An Unofficial Implementation of WavThruVec Based on Pytorch.

The original paper is [WavThruVec: Latent speech representation as intermediate features for
neural speech synthesis](https://arxiv.org/abs/2203.16930)

### Note:
Use wav2vec2.0's output as the wav's feature(instead of mel spectrogram), with a dtype of 'float16' and a shape of (batch_size, n_frame, n_channel).

note: n_channel=768


## environment
* CUDA 10.1
* python                    3.9.7
* torch                     1.8.1+cu101
* torch-optimizer           0.3.0      
* torchaudio                0.8.1
* tensorboard               2.12.0 
* librosa                   0.8.0 
* numba                     0.56.4
* numpy                     1.22.4  
* llvmlite                  0.39.1  

## wav2vec 2.0 pretrained

From this repository [wav2vec2.0 (chinese speech pretrain)](https://github.com/TencentGameMate/chinese_speech_pretrain), and it can also be found at [huggingface](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base)


## dataset and prepare
[aishell3](https://www.aishelltech.com/aishell_3)

The prepare_data.py:
* 1.read the wav files and wav2vec2 pretrained model, resample the wavs to 16khz, and convert to .npy files, which contrain the corresponding wav2vec 2.0 feature.
* 2.read the aishell3 transcription(content.txt), and filter the Chinese phoneme and blank. Take the transcription and file path to build the train list(./data/enc_train.txt).
* 3.build the vocab, which will be used to convert the characters to torch Variable. 

As an example, prepare_data.py only take a few speakers and a few wav files. 


## training
WavThruVec contrains 2 components: Text2Vec(encoder) and Vec2Wav(decoder), and they train independently

Thus, I placed them in two separate dirs and used different training configurations for each.

## Todo
* Vec2Wav  (Major task)
* experiment & Performace 
* More details for implementation 


## Reference
### Repository
- [fastspeech (xcmyz's)](https://github.com/xcmyz/FastSpeech)
- [wav2vec2.0 (chinese speech pretrain)](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [rad-tts (nvidia's)](https://github.com/NVIDIA/radtts)
- [gan-tts (yanggeng1995's)](https://github.com/yanggeng1995/GAN-TTS)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [Fastpitch (dan-wells')](https://github.com/dan-wells/fastpitch)
- [ecapa_tdnn (Tao Ruijie's)](https://github.com/TaoRuijie/ECAPA-TDNN/tree/main)
- [ecapa_tdnn (lawlict's)](https://github.com/lawlict/ECAPA-TDNN/tree/master)
- [glow-tts (jaywalnut310's)](https://github.com/jaywalnut310/glow-tts)

### Paper
- [FastSpeech](https://arxiv.org/abs/1905.09263)
- [FastSpeech2](https://arxiv.org/abs/2006.04558)
- [hifi-gan](https://arxiv.org/pdf/2010.05646.pdf)
- [wav2vec](https://arxiv.org/pdf/2006.11477.pdf)
- [rad-tts](https://openreview.net/pdf?id=0NQwnnwAORi)
- [monotonic alignment search](https://arxiv.org/pdf/2108.10447.pdf)
