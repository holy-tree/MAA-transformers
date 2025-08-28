import torch
import torch.nn as nn
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding_inverted,PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Generator_ptransformer(nn.Module):
    """
    修改后的PatchTST模型，显式参数传入，仅保留预测和分类任务
    参考论文: https://arxiv.org/pdf/2211.14730.pdf
    """
    def __init__(self,
                 input_dim,  # 输入特征维度（对应原enc_in）
                 seq_len,  # 输入序列长度
                 output_len=1,  # 预测序列长度（输出长度）
                 feature_size=512,  # 模型内部特征维度（对应原d_model）
                 num_layers=2,  # 编码器层数（对应原e_layers）
                 num_heads=8,  # 注意力头数（对应原n_heads）
                 d_ff=2048,  # 前馈网络维度（通常为feature_size的4倍）
                 dropout=0.1,  # dropout概率
                 activation='gelu',  # 激活函数
                 factor=5,  # 注意力因子（原FullAttention的factor）
                 patch_len=96,  # patch长度
                 stride=96,  # patch滑动步长
                 num_cls=3  # 分类任务类别数（固定为3类）
                 ):
        super().__init__()
        self.input_dim = input_dim  # 输入特征维度
        self.seq_len = seq_len  # 输入序列长度
        self.output_len = output_len  # 预测长度
        self.feature_size = feature_size  # 模型内部特征维度
        self.num_cls = num_cls  # 分类类别数

        # 1. Patch嵌入层（核心组件，用于序列分块）
        padding = stride  # 保持原逻辑的padding设置
        self.patch_embedding = PatchEmbedding(
            d_model=feature_size,
            patch_len=patch_len,
            stride=stride,
            padding=padding,
            dropout=dropout
        )

        # 2. 编码器（堆叠多层EncoderLayer）
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,  # 不使用掩码
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=True
                        ),
                        d_model=feature_size,
                        n_heads=num_heads
                    ),
                    d_model=feature_size,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(num_layers)  # 堆叠num_layers层
            ],
            # 归一化层（保留原转置+BN逻辑）
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(feature_size), Transpose(1, 2))
        )

        # 3. 计算预测头的输入维度（head_nf）
        self.patch_num = int((self.seq_len - patch_len) / stride + 2)  # 分块数量
        self.head_nf = feature_size * self.patch_num  # 预测头的输入特征数

        # 4. 预测任务头（时间序列预测）
        self.pred_head = FlattenHead(
            n_vars=input_dim,
            nf=self.head_nf,
            target_window=output_len,
            head_dropout=dropout
        )

        # 5. 分类任务头（3类分类）
        self.cls_flatten = nn.Flatten(start_dim=-2)  # 展平特征
        self.cls_dropout = nn.Dropout(dropout)
        self.cls_projection = nn.Linear(
            self.head_nf * input_dim,  # 输入维度=head_nf * 特征数
            num_cls  # 输出3类
        )
        self.output_projection = nn.Linear(input_dim, 4)

    def _normalize(self, x):
        """序列归一化（保留原Non-stationary Transformer逻辑）"""
        means = x.mean(1, keepdim=True).detach()  # 按时间步求均值
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)  # 标准差
        x /= stdev
        return x, means, stdev

    def _denormalize(self, x, means, stdev):
        """反归一化"""
        stdev = stdev[:, 0, :].unsqueeze(1).repeat(1, self.output_len, 1)
        means = means[:, 0, :].unsqueeze(1).repeat(1, self.output_len, 1)
        return x * stdev + means

    def forecast(self, x_enc):
        """时间序列预测任务"""
        # 归一化
        x_enc, means, stdev = self._normalize(x_enc)  # x_enc: [B, seq_len, input_dim]

        # Patch嵌入：调整维度并分块
        x_enc = x_enc.permute(0, 2, 1)  # 转为 [B, input_dim, seq_len]（适配分块逻辑）
        enc_out, n_vars = self.patch_embedding(x_enc)  # enc_out: [B*input_dim, patch_num, feature_size]

        # 编码器特征提取
        enc_out, _ = self.encoder(enc_out)  # [B*input_dim, patch_num, feature_size]

        # 维度调整：恢复批次和特征维度
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))  # [B, input_dim, patch_num, feature_size]
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, input_dim, feature_size, patch_num]

        # 预测头输出
        gen = self.pred_head(enc_out)  # [B, input_dim, pred_len]
        gen = gen.permute(0, 2, 1)  # 转为 [B, pred_len, input_dim]

        # # 反归一化
        gen = self._denormalize(gen, means, stdev)
        gen = gen.squeeze(1)  # [B, 12]
        final_gen = self.output_projection(gen)  # [B, 4]

        return final_gen

    def classification(self, x_enc):
        """分类任务（预测3类）"""
        # # 归一化
        # x_enc, _, _ = self._normalize(x_enc)  # 仅归一化，不保留均值标准差
        # print(x_enc.shape)
        # Patch嵌入
        x_enc = x_enc.permute(0, 2, 1)  # [B, input_dim, seq_len]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [B*input_dim, patch_num, feature_size]

        # 编码器特征提取
        enc_out, _ = self.encoder(enc_out)  # [B*input_dim, patch_num, feature_size]

        # 维度调整
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))  # [B, input_dim, patch_num, feature_size]
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, input_dim, feature_size, patch_num]

        # 分类头输出
        cls_feat = self.cls_flatten(enc_out)  # 展平为 [B, input_dim, feature_size*patch_num]
        cls_feat = self.cls_dropout(cls_feat)
        cls_feat = cls_feat.reshape(cls_feat.shape[0], -1)  # [B, input_dim*feature_size*patch_num]
        cls = self.cls_projection(cls_feat)  # [B, num_cls]
        return cls

    def forward(self, x_enc):
        """前向传播：同时输出预测和分类结果"""
        # 预测结果（gen）
        gen = self.forecast(x_enc)  # [B, pred_len, input_dim]
        # 分类结果（cls）
        cls = self.classification(x_enc)  # [B, num_cls]
        return gen, cls