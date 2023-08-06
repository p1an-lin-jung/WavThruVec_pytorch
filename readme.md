

# WavThruVec Pytorch
An Unofficial Implementation of WavThruVec Based on Pytorch.

The original paper is [WavThruVec: Latent speech representation as intermediate features for
neural speech synthesis](https://arxiv.org/abs/2203.16930)

### Note:
Use wav2vec2.0's output as the wav's feature(instead of mel spectrogram), with a dtype of 'float16' and a shape of (batch_size, n_frame, n_channel).
I transpose the feature's shape to (batch_size, n_channel,n_frame) to make it similar to mel spectrogram.

note: n_channel=768


## dataset
[aishell3](https://www.aishelltech.com/aishell_3)

## Todo
text2vec- validation
Vec2Wav - (decoder)
Performace 



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
- wav2vec
- [rad-tts](https://openreview.net/pdf?id=0NQwnnwAORi)
- [monotonic alignment search](https://arxiv.org/pdf/2108.10447.pdf)
