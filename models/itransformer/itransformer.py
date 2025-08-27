import torch
import torch.nn as nn
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding_inverted


class Generator_itransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Generator_itransformer, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # 添加新的分类头
        self.classifier = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model // 2, configs.num_classes)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out,enc_out, attns

    # def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
    #     dec_out,enc_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
    #
    #     # 从 enc_out 中计算分类结果
    #     # 这里使用对所有 tokens 求均值，得到一个全局表示
    #     cls_input = enc_out.mean(1)
    #     cls_out = self.classifier(cls_input)
    #
    #     # 假设你的目标是预测 pred_len 个时间步，并且每个时间步有 N 个变量
    #     # dec_out 的形状是 [B, pred_len, N]
    #     # 但你的 val_y 形状是 [B, N]，这暗示你的目标只有 1 个时间步
    #     # 因此，我们只取 dec_out 的最后一个时间步作为预测结果
    #     final_predictions = dec_out[:, -1, :]  # 形状变为 [B, N]
    #
    #     if self.output_attention:
    #         return final_predictions, cls_out, attns
    #     else:
    #         return final_predictions, cls_out  # [B, N], [B, num_classes]
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # 打印输入 x_enc 的形状
        # print(f"输入 x_enc 形状: {x_enc.shape}")

        dec_out, enc_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # 打印 forecast 的输出 dec_out 的形状
        # print(f"forecast 输出 dec_out 形状: {dec_out.shape}")

        # 从 enc_out 中计算分类结果
        # 这里使用对所有 tokens 求均值，得到一个全局表示
        cls_input = enc_out.mean(1)
        cls_out = self.classifier(cls_input)

        # 打印分类头的输出 cls_out 的形状
        # print(f"分类头输出 cls_out 形状: {cls_out.shape}")

        # --- 这是关键的修改部分 ---

        # 1. 先从 dec_out (形状 [B, pred_len, 12]) 中选择最后一个时间步的预测
        final_predictions_all_features = dec_out[:, -1, :]  # 形状变为 [B, 12]

        # 2. 【新增】从这12个特征中，只选择前4个作为最终输出
        #    因为你的目标变量 (target_columns) 有4个
        num_target_features = 4  # 根据你的需求，这里是4
        final_predictions = final_predictions_all_features[:, :num_target_features]  # 形状变为 [B, 4]

        # --- 修改结束 ---

        # 打印最终预测的形状
        # print(f"最终预测 final_predictions 形状: {final_predictions.shape}")

        if self.output_attention:
            return final_predictions, cls_out, attns
        else:
            return final_predictions, cls_out  # [B, N], [B, num_classes]