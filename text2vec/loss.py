import pdb

import torch
import torch.nn as nn


class AttentionBinarizationLoss(torch.nn.Module):
    def __init__(self):
        super(AttentionBinarizationLoss, self).__init__()

    def forward(self, hard_attention, soft_attention):
        # pdb.set_trace()

        log_sum = torch.log(soft_attention[hard_attention == 1]).sum()
        return -log_sum / hard_attention.sum()


class DNNLoss(nn.Module):
    def __init__(self):
        super(DNNLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    # def forward(self, mel, mel_postnet, duration_predicted, mel_target, duration_predictor_target):
    #     mel_target.requires_grad = False
    #     mel_loss = self.mse_loss(mel, mel_target)
    #     mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)
    #
    #     duration_predictor_target.requires_grad = False
    #     duration_predictor_loss = self.l1_loss(duration_predicted,
    #                                            duration_predictor_target.float())
    #
    #     return mel_loss, mel_postnet_loss, duration_predictor_loss

    def forward(self, feat_output, feat_postnet, feat_target,duration_predicted=None,
                duration_predictor_target=None):

        feat_target.requires_grad = False

        # pdb.set_trace()
        WVF_loss = self.mse_loss(feat_output, feat_target)
        WVF_postnet_loss = self.mse_loss(feat_postnet, feat_target)

        # training stage
        if duration_predicted is not None:
            duration_predictor_target.requires_grad = False
            duration_predictor_loss = self.mse_loss(duration_predicted,
                                                   duration_predictor_target.float())


            return WVF_loss, WVF_postnet_loss, duration_predictor_loss

        # inference stage: not pass duration_predicted
        else:
            return WVF_loss,WVF_postnet_loss
