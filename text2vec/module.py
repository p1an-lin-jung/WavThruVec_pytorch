import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from numba import jit
import numpy as np
import copy
import math

import hparams as hp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# convert duration to alignment
def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = DurationPredictor()

    def LR(self, x, attn_hard=None,duration_predictor_output=None,WVF_max_length=None):
 

        # training：
        if attn_hard is None:
            expand_max_len = torch.max(
                torch.sum(duration_predictor_output, -1), -1)[0]#### todo check
            alignment = torch.zeros(duration_predictor_output.size(0),
                                    int(expand_max_len),
                                    duration_predictor_output.size(1)).numpy()###

            alignment = create_alignment(alignment,
                                         duration_predictor_output.cpu().detach().numpy())
            alignment = torch.from_numpy(alignment).to(device)
        # inference：
        else:
            alignment = attn_hard.squeeze()  # [batch_sz,1,len_feat,len_text]->[batch_sz, len_feat,len_text]

        output = alignment @ x  # alig：16,n_fr，28 x：16,28,256-> 16,n_fr,256
        if WVF_max_length:
            output = F.pad(
                output, (0, 0, 0, WVF_max_length-output.size(1), 0, 0))
        return output
    #
    def forward(self, x, alpha=1.0,target=None, attn=None,WVF_max_length=None):

        duration_predictor_output = self.duration_predictor(x) # infer:[n] val:[bz,n]

        # train stage
        if attn is not None:
            output = self.LR(x,attn_hard=attn,WVF_max_length=WVF_max_length)
            return output, duration_predictor_output
        else:
            #  infer and val stage
            duration_predictor_output = (
                (duration_predictor_output + 0.5) * alpha).int() # 做四舍五入
            print(duration_predictor_output)
            output = self.LR(x, duration_predictor_output=duration_predictor_output)

            # if len(output.shape)==2: # if single-infer , output.shape=[bz,n_text] todo
            WVF_pos = torch.stack(
                [torch.Tensor([i+1 for i in range(output.shape[1])])]).long().to(device) # [1,n_text]
            
            return output, WVF_pos



class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self):
        super(DurationPredictor, self).__init__()

        self.input_size = hp.encoder_dim #
        if hp.use_multi_speaker_condition:
            self.input_size += hp.n_speaker_dim
        self.filter_size = hp.duration_predictor_filter_size
        self.kernel = hp.duration_predictor_kernel_size
        self.conv_output_size = hp.duration_predictor_filter_size
        self.dropout = hp.dropout

        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(self.input_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("relu_1", nn.ReLU()),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("conv1d_2", Conv(self.filter_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("relu_2", nn.ReLU()),
            ("dropout_2", nn.Dropout(self.dropout))
        ]))

        self.linear_layer = Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        # pdb.set_trace()
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = self.relu(out)
        out = out.squeeze()

        #  val or infer
        if not self.training:
            # val:[bz,n], same as training
            # infer: [n]->[1,n];
            out = out.unsqueeze(0)
        return out


class BatchNormConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding,
                 activation=None, w_init_gain='linear'):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_dim, out_dim,
                                kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = activation

        torch.nn.init.xavier_uniform_(
            self.conv1d.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class Prenet(nn.Module):
    """
    Prenet before passing through the network
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(Prenet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
            ('fc1', Linear(self.input_size, self.hidden_size)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.5)),
            ('fc2', Linear(self.hidden_size, self.output_size)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.5)),
        ]))

    def forward(self, x):
        out = self.layer(x)
        return out


class CBHG(nn.Module):
    """CBHG module: a recurrent neural network composed of:
        - 1-d convolution banks
        - Highway networks + residual connections
        - Bidirectional gated recurrent units
    """

    def __init__(self, in_dim, K=16, projections=[128, 128]):
        super(CBHG, self).__init__()
        self.in_dim = in_dim
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList(
            [BatchNormConv1d(in_dim, in_dim, kernel_size=k, stride=1,
                             padding=k // 2, activation=self.relu)
             for k in range(1, K + 1)])
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        in_sizes = [K * in_dim] + projections[:-1]
        activations = [self.relu] * (len(projections) - 1) + [None]
        self.conv1d_projections = nn.ModuleList(
            [BatchNormConv1d(in_size, out_size, kernel_size=3, stride=1,
                             padding=1, activation=ac)
             for (in_size, out_size, ac) in zip(
                 in_sizes, projections, activations)])

        self.pre_highway = nn.Linear(projections[-1], in_dim, bias=False)
        self.highways = nn.ModuleList(
            [Highway(in_dim, in_dim) for _ in range(4)])

        self.gru = nn.GRU(
            in_dim, in_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        # (B, T_in, in_dim)
        x = inputs

        # Needed to perform conv1d on time-axis
        # (B, in_dim, T_in)
        if x.size(-1) == self.in_dim:
            x = x.transpose(1, 2)

        T = x.size(-1)

        # (B, in_dim*K, T_in)
        # Concat conv1d bank outputs
        x = torch.cat([conv1d(x)[:, :, :T]
                       for conv1d in self.conv1d_banks], dim=1)
        assert x.size(1) == self.in_dim * len(self.conv1d_banks)
        x = self.max_pool1d(x)[:, :, :T]

        for conv1d in self.conv1d_projections:
            x = conv1d(x)

        # (B, T_in, in_dim)
        # Back to the original shape
        x = x.transpose(1, 2)

        if x.size(-1) != self.in_dim:
            x = self.pre_highway(x)

        # Residual connection
        x += inputs
        for highway in self.highways:
            x = highway(x)

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True)

        # (B, T_in, in_dim*2)
        self.gru.flatten_parameters()
        outputs, _ = self.gru(x)

        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)

        return outputs

class PartialConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):

        self.multi_channel = False
        self.return_mask = False
        super(PartialConv1d, self).__init__(*args, **kwargs)

        self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0])
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2]

        self.last_size = (None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    @torch.jit.ignore
    def forward(self, input: torch.Tensor, mask_in : torch.Tensor = None):
        """
        input: standard input to a 1D conv
        mask_in: binary mask for valid values, same shape as input
        """
        assert len(input.shape) == 3
        # if a mask is input, or tensor shape changed, update mask ratio
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)
            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)
                if mask_in is None:
                    mask = torch.ones(1, 1, input.data.shape[2]).to(input)
                else:
                    mask = mask_in
                self.update_mask = F.conv1d(mask, self.weight_maskUpdater,
                                            bias=None, stride=self.stride,
                                            padding=self.padding,
                                            dilation=self.dilation, groups=1)
                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-6)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)
        raw_out = super(PartialConv1d, self).forward(
            torch.mul(input, mask) if mask_in is not None else input)
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear',
                 use_partial_padding=False, use_weight_norm=False):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_partial_padding = use_partial_padding
        self.use_weight_norm = use_weight_norm
        conv_fn = torch.nn.Conv1d
        if self.use_partial_padding:
            conv_fn = PartialConv1d
        self.conv = conv_fn(in_channels, out_channels,
                            kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation,
                            bias=bias)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
        if self.use_weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, signal, mask=None):
        if self.use_partial_padding:
            conv_signal = self.conv(signal, mask)
        else:
            conv_signal = self.conv(signal)
        if mask is not None:
            # always re-zero output if mask is
            # available to match zero-padding
            conv_signal = conv_signal * mask
        return conv_signal

class ConvAttention(torch.nn.Module):
    def __init__(self, n_WVF_channels=80, n_text_channels=512,
                 n_att_channels=80, temperature=1.0):
        super(ConvAttention, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            ConvNorm(n_text_channels, n_text_channels*2, kernel_size=3,
                     bias=True, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_text_channels*2, n_att_channels, kernel_size=1,
                     bias=True))

        self.query_proj = nn.Sequential(
            ConvNorm(n_WVF_channels, n_WVF_channels*2, kernel_size=3,
                     bias=True, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_WVF_channels*2, n_WVF_channels, kernel_size=1,
                     bias=True),
            torch.nn.ReLU(),
            ConvNorm(n_WVF_channels, n_att_channels, kernel_size=1, bias=True)
        )

    def run_padded_sequence(self, sorted_idx, unsort_idx, lens, padded_data,
                            recurrent_model):
        """Sorts input data by previded ordering (and un-ordering) and runs the
        packed data through the recurrent model

        Args:
            sorted_idx (torch.tensor): 1D sorting index
            unsort_idx (torch.tensor): 1D unsorting index (inverse of sorted_idx)
            lens: lengths of input data (sorted in descending order)
            padded_data (torch.tensor): input sequences (padded)
            recurrent_model (nn.Module): recurrent model to run data through
        Returns:
            hidden_vectors (torch.tensor): outputs of the RNN, in the original,
            unsorted, ordering
        """

        # sort the data by decreasing length using provided index
        # we assume batch index is in dim=1
        padded_data = padded_data[:, sorted_idx]
        padded_data = nn.utils.rnn.pack_padded_sequence(padded_data, lens)
        hidden_vectors = recurrent_model(padded_data)[0]
        hidden_vectors, _ = nn.utils.rnn.pad_packed_sequence(hidden_vectors)
        # unsort the results at dim=1 and return
        hidden_vectors = hidden_vectors[:, unsort_idx]
        return hidden_vectors

    def forward(self, queries, keys, query_lens, mask=None, key_lens=None,
                attn_prior=None):
        """Attention mechanism for radtts. Unlike in Flowtron, we have no
        restrictions such as causality etc, since we only need this during
        training.

        Args:
            queries (torch.tensor): B x C x T1 tensor (likely mel data)
            keys (torch.tensor): B x C2 x T2 tensor (text data)
            query_lens: lengths for sorting the queries in descending order
            mask (torch.tensor): uint8 binary mask for variable length entries
                                 (should be in the T2 domain)
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask.
                                 Final dim T2 should sum to 1
        """
        temp = 0.0005
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        # Beware can only do this since query_dim = attn_dim = n_mel_channels
        queries_enc = self.query_proj(queries)

        # Gaussian Isotopic Attention
        # B x n_attn_dims x T1 x T2
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None])**2

        # compute log-likelihood from gaussian
        eps = 1e-8
        attn = -temp * attn.sum(1, keepdim=True)
        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + eps)

        attn_logprob = attn.clone()

        if mask is not None:
             
            attn.data.masked_fill_(
                    mask.permute(0, 2, 1).unsqueeze(2), -float("inf"))
             
        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob



if __name__ == "__main__":
    # TEST
    a = torch.Tensor([[2, 3, 4], [1, 2, 3]])
    b = torch.Tensor([[5, 6, 7], [7, 8, 9]])
    c = torch.stack([a, b])

    d = torch.Tensor([[1, 4], [6, 3]]).int()
    expand_max_len = torch.max(torch.sum(d, -1), -1)[0]
    base = torch.zeros(c.size(0), expand_max_len, c.size(1))

    alignment = create_alignment(base.numpy(), d.numpy())
    print(alignment)
    print(torch.from_numpy(alignment) @ c)
