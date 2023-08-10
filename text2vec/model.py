import torch
import torch.nn as nn
import hparams as hp

import numpy as np

from subLayer import FFTBlock, PreNet, PostNet, Linear
from module import LengthRegulator, CBHG,ConvAttention
import Constants
import utils
from ecapa_tdnn_TaoRuijie import ECAPA_TDNN
from alignment import mas_width1 as mas
import pdb



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_mask_from_lengths(lengths):
    """Constructs binary mask from a 1D torch tensor of input lengths

    Args:
        lengths (torch.tensor): 1D tensor
    Returns:
        mask (torch.tensor): num_sequences x max_length x 1 binary tensor
    """
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class Encoder(nn.Module):
    ''' Encoder '''

    def __init__(self,
                 n_src_vocab=hp.vocab_size,
                 len_max_seq=hp.vocab_size,
                 d_word_vec=hp.encoder_dim,  # 256
                 n_layers=hp.encoder_n_layer,
                 n_head=hp.encoder_head,
                 d_model=hp.encoder_dim,
                 d_inner=hp.encoder_conv1d_filter_size,#1024
                 dropout=hp.dropout):

        super(Encoder, self).__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(n_src_vocab,
                                         d_word_vec,
                                         padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        if hp.use_multi_speaker_condition:
            self.speaker_encoder=ECAPA_TDNN(hp.spk_channel,input_wav=hp.input_wav,n_feat_dim=hp.n_feat_dim)
            d_model+=hp.n_speaker_dim

        d_k=d_model // hp.encoder_head
        d_v=d_model // hp.encoder_head
        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, wav_feat,return_attns=False):


        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward

        text_emb=self.src_word_emb(src_seq) # [bsz,src_seq_len,text_dim(256)]
        enc_output = text_emb + self.position_enc(src_pos) #[batch_sz，src_seq_len，encoder_dim])

        if hp.use_multi_speaker_condition:
            spk_emb=self.speaker_encoder(wav_feat.transpose(1,2))# [batch_sz,192]
            spk_emb_repeat=spk_emb.unsqueeze(1)## [batch_sz,1,192]
            spk_emb_repeat = spk_emb_repeat.repeat(1, enc_output.size(1), 1)# [bsz,src_seq_len,192]
            enc_output=torch.cat((enc_output,spk_emb_repeat),dim=2)# [bsz,src_seq_len,text_dim+spk_dim(192+256=448)]


        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, non_pad_mask, text_emb, spk_emb


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self,
                 len_max_seq=hp.max_seq_len,
                 n_layers=hp.decoder_n_layer,
                 n_head=hp.decoder_head,

                 d_model=hp.decoder_dim,
                 d_inner=hp.decoder_conv1d_filter_size,
                 dropout=hp.dropout):

        super(Decoder, self).__init__()

        n_position = len_max_seq + 1

        if hp.use_multi_speaker_condition:
            d_model+=hp.n_speaker_dim

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        d_k = d_model // hp.encoder_head
        d_v = d_model // hp.encoder_head
        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):
        # encseq:# 16, 151, 448] enc_pos:[1（16）,151,]
        #p slf_attn_mask.shape ：torch.Size([1, 151, 151])

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos) #([16, 431, 448])

        for dec_layer in self.layer_stack:
            # pdb.set_trace()
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output


class Text2Vec(nn.Module):
    """ Text2Vec """

    def __init__(self):
        super(Text2Vec, self).__init__()

        self.encoder = Encoder() # return : enc_output, non_pad_mask, spk_emb
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder()

        if hp.use_multi_speaker_condition:
            self.WVF_linear = Linear(hp.decoder_dim+hp.n_speaker_dim, hp.n_feat_dim)
        else:
            self.WVF_linear = Linear(hp.decoder_dim, hp.n_feat_dim)


        self.postnet = CBHG(hp.n_feat_dim, K=8,
                            projections=[256, hp.n_feat_dim])
        self.last_linear = Linear(hp.n_feat_dim * 2, hp.n_feat_dim)


        self.learn_alignments=hp.learn_alignments # temp
        self.use_speaker_emb_for_alignment=hp.use_speaker_emb_for_alignment
        if self.learn_alignments:
            if self.use_speaker_emb_for_alignment:
                # n_feat_dim=768, that is the wav2vec's output shape
                self.attention = ConvAttention(
                    hp.n_feat_dim, hp.encoder_dim + hp.n_speaker_dim)
            else:
                self.attention = ConvAttention(hp.n_feat_dim, hp.encoder_dim)

    def mask_tensor(self, WVF_output, position, WVF_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~utils.get_mask_from_lengths(lengths, max_len=WVF_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, WVF_output.size(-1))
        return WVF_output.masked_fill(mask, 0.)

    def convert_hard_attn_to_duration(self,hard_attn):
        """
        return: duration(torch.Tensor): [b x len_txt]

        Args:
            hard_attn (torch.Tensor): [b x len_text x len_feat]
        """
        return torch.sum(hard_attn,dim=2).int()

    def binarize_attention(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS. These will
        no longer recieve a gradient
        Args:
            attn: B x 1 x max_WVF_len x max_text_len
        """
        b_size = attn.shape[0]
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = torch.zeros_like(attn)
            for ind in range(b_size):
                hard_attn = mas(attn_cpu[ind, 0, :out_lens[ind], :in_lens[ind]])
                attn_out[ind, 0, :out_lens[ind], :in_lens[ind]] = torch.tensor(
                    hard_attn, device=attn.get_device())  # bsz,1,len_text,len_WVF
        return attn_out

    def get_attn_and_duration(self,
                              wav_feat,
                              in_lens,out_lens,
                              text_embeddings,
                              speaker_vecs,attn_prior,
                              binarize_attention=True):
        # ====ref to rad-tts

        # text_embeddings: b x len_text x n_text_dim [512, 125]
        text_embeddings = text_embeddings.transpose(1, 2)
        attn = None
        attn_soft = None
        attn_hard = None
        # make sure to do the alignments before folding
        attn_mask = get_mask_from_lengths(in_lens)[..., None] == 0


        text_embeddings_for_attn = text_embeddings
        if self.use_speaker_emb_for_alignment:
            speaker_vecs_expd = speaker_vecs[:, :, None].expand(
                -1, -1, text_embeddings.shape[2])
            text_embeddings_for_attn = torch.cat(
                (text_embeddings_for_attn, speaker_vecs_expd.detach()), 1)

        attn_soft, attn_logprob = self.attention(
            wav_feat.transpose(1,2), # switch  frame-dim and channel-dim
            text_embeddings_for_attn,
            out_lens,
            attn_mask,
            key_lens=in_lens, attn_prior=attn_prior)

        if binarize_attention:
            attn = self.binarize_attention(attn_soft, in_lens, out_lens)
            attn_hard = attn
        else:
            attn = attn_soft

        duration = self.convert_hard_attn_to_duration(attn.squeeze(1).transpose(1, 2))  # [bsz, 1, len_txt])

        return attn,attn_soft,duration


    def forward(self, wav_feat,src_seq, src_pos,in_lens=None, out_lens=None
                ,WVF_pos=None, WVF_max_length=None, alpha=1.0,
                binarize_attention=True,attn_prior=None):

        encoder_output, _,text_embeddings,speaker_vecs = self.encoder(src_seq, src_pos,wav_feat)
        # pdb.set_trace()
        # train soft-alignment,and convert to hard-alignment and duration(as length_regulator's target)
        attn_hard,attn_soft,duration=self.get_attn_and_duration(wav_feat,
                                   in_lens,
                                   out_lens,
                                   text_embeddings,
                                   speaker_vecs,
                                   attn_prior,
                                   binarize_attention=binarize_attention)

        if self.training:
            length_regulator_output, duration_predictor_output = self.length_regulator(encoder_output,
                                                                                       attn=attn_hard,
                                                                                       WVF_max_length=WVF_max_length)

            decoder_output = self.decoder(length_regulator_output, WVF_pos) #torch.Size([16, 431, 448])
            WVF_output = self.WVF_linear(decoder_output)
            WVF_output = self.mask_tensor(WVF_output, WVF_pos, WVF_max_length)
            residual = self.postnet(WVF_output)
            residual = self.last_linear(residual)
            WVF_postnet_output = WVF_output + residual
            WVF_postnet_output = self.mask_tensor(WVF_postnet_output,
                                                  WVF_pos,
                                                  WVF_max_length)

            output={
                'feat_output':WVF_output,
                'feat_postnet_output':WVF_postnet_output,
                'duration_predictor_output':duration_predictor_output,
                'duration':duration,
                'attn':attn_hard,
                'attn_soft':attn_soft
            }
            return output
        else:
            length_regulator_output, decoder_pos = self.length_regulator(encoder_output,
                                                                         alpha=alpha)

            decoder_output = self.decoder(length_regulator_output, decoder_pos) #[16, 431, 448])

            WVF_output = self.WVF_linear(decoder_output)
            residual = self.postnet(WVF_output)
            residual = self.last_linear(residual)
            WVF_postnet_output = WVF_output + residual

            output = {
                'feat_output': WVF_output,
                'feat_postnet_output': WVF_postnet_output,
                # 'duration_predictor_output': duration_predictor_output,
                'duration': duration,
                'attn': attn_hard,
                'attn_soft': attn_soft
            }
            return output


if __name__ == "__main__":
    # Test
    model = Text2Vec()
    print(sum(param.numel() for param in model.parameters()))
