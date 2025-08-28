import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Generator_FITS(nn.Module):
    def __init__(self, configs):
        super(Generator_FITS, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in  # 原始代码中通常为1，现在应为4（OHLC）
        self.num_classes = configs.num_classes  # 新增：分类类别的数量

        self.dominance_freq = configs.cut_freq
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        # 频率上采样器，现在处理多个通道
        if self.individual:
            # 独立处理每个通道
            self.freq_upsampler_real = nn.ModuleList()
            self.freq_upsampler_imag = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler_real.append(
                    nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)))
                self.freq_upsampler_imag.append(
                    nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)))
        else:
            # 为实部和虚部创建独立的线性层
            self.freq_upsampler_real = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio))
            self.freq_upsampler_imag = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio))

        self.output_dim = 4
        pred_len_upsampled = int(self.dominance_freq * self.length_ratio)

        # ✅ 线性头输入 = 未来片段长度 * 通道数（与实际喂入严格一致）
        in_feats = self.pred_len * configs.enc_in
        self.regression_head = nn.Linear(in_features=in_feats, out_features=self.output_dim)
        self.classification_head = nn.Linear(in_features=in_feats, out_features=configs.num_classes)

    def forward(self, x):
        # RIN（Relative Input Normalization）
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:, self.dominance_freq:] = 0
        low_specx = low_specx[:, 0:self.dominance_freq, :]

        # 分别提取实部和虚部
        low_specx_real = low_specx.real.permute(0, 2, 1)
        low_specx_imag = low_specx.imag.permute(0, 2, 1)

        # 频率上采样
        if self.individual:
            low_specxy_real = torch.zeros(
                [low_specx_real.size(0), int(self.dominance_freq * self.length_ratio), low_specx_real.size(2)],
                dtype=torch.float32).to(low_specx.device)
            low_specxy_imag = torch.zeros(
                [low_specx_imag.size(0), int(self.dominance_freq * self.length_ratio), low_specx_imag.size(2)],
                dtype=torch.float32).to(low_specx.device)

            for i in range(self.channels):
                low_specxy_real[:, :, i] = self.freq_upsampler_real[i](low_specx_real[:, :, i])
                low_specxy_imag[:, :, i] = self.freq_upsampler_imag[i](low_specx_imag[:, :, i])

            low_specxy_ = torch.complex(low_specxy_real, low_specxy_imag).permute(0, 2, 1)
        else:
            # 分别对实部和虚部进行上采样
            low_specxy_real = self.freq_upsampler_real(low_specx_real)
            low_specxy_imag = self.freq_upsampler_imag(low_specx_imag)
            # 重新组合成复数
            low_specxy_ = torch.complex(low_specxy_real, low_specxy_imag).permute(0, 2, 1)

        # 零填充并进行逆FFT
        low_specxy = torch.zeros(
            [low_specxy_.size(0), int((self.seq_len + self.pred_len) / 2 + 1), low_specxy_.size(2)],
            dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.length_ratio  # 能量补偿

        # 反向 RIN
        pred_ohlc_full = low_xy * torch.sqrt(x_var) + x_mean  # [B, seq_len+pred_len, C]

        # ✅ 只取未来片段，再展平
        pred_future = pred_ohlc_full[:, -self.pred_len:, :]  # [B, pred_len, C]
        flattened = pred_future.reshape(pred_future.size(0), -1)  # [B, pred_len*C]

        # 线性头
        new_pred_ohlc = self.regression_head(flattened)  # [B, 4]
        classification_output = self.classification_head(flattened)  # [B, num_classes]
        return new_pred_ohlc, classification_output