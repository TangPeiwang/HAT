import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.encoder import EncoderLayer
import math
from einops import rearrange


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.res_attention = True
        self.layers = configs.e_layers

        self.small_len = 96
        self.emb_patch_len = 8
        self.mask_patch_len = 4
        self.pt_len = configs.pt_len

        self.layer_norm = nn.LayerNorm(configs.d_model)
        # Residual dropout
        self.dropout = nn.Dropout(configs.dropout)

        if self.task_name == 'per_training':
            self.keep_ratio = configs.keep_ratio
            self.mask_token = nn.Parameter(torch.zeros(1, 1))
            self.enc_embedding = nn.Linear(self.emb_patch_len, configs.d_model, bias=True)
            self.encoder_pos_embed = nn.Parameter(torch.zeros(1, self.pt_len // self.emb_patch_len, configs.d_model),
                                                  requires_grad=True)
            ###########
            attn_dropout = 0.0
            self.encoder = nn.ModuleList(
                [EncoderLayer(configs.d_model, n_heads=configs.n_heads, d_ff=configs.d_ff, others_args=configs,
                              dropout=configs.dropout, res_attention=self.res_attention,
                              activation=configs.activation) for i in range(self.layers)])

            self.decoder = nn.Linear(configs.d_model, self.emb_patch_len, bias=True)  # decoder to patch
        else:
            self.enc_embedding = nn.Linear(self.emb_patch_len, configs.d_model, bias=True)
            self.encoder_pos_embed = nn.Parameter(torch.zeros(1, self.pt_len // self.emb_patch_len, configs.d_model),
                                                  requires_grad=False)
            self.encoder = nn.ModuleList(
                [EncoderLayer(configs.d_model, n_heads=configs.n_heads, d_ff=configs.d_ff, others_args=configs,
                              dropout=configs.dropout, res_attention=self.res_attention,
                              activation=configs.activation) for i in range(self.layers)])
            self.head_nf = configs.d_model * \
                           int((configs.seq_len - self.emb_patch_len) / self.emb_patch_len + 1)
            if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                self.head = FlattenHead(self.head_nf, configs.pred_len,
                                        head_dropout=configs.dropout)
            if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
                self.head = FlattenHead(self.head_nf, configs.seq_len,
                                        head_dropout=configs.dropout)
            if self.task_name == 'classification':
                self.emb_patch_len = 8
                self.enc_embedding = nn.Linear(self.emb_patch_len, configs.d_model, bias=True)
                self.flatten = nn.Flatten(start_dim=-2)
                self.dropout = nn.Dropout(configs.dropout)
                self.projection = nn.Linear(
                    self.head_nf * configs.enc_in, configs.num_class)

    def mask_patch(self, xb):
        mask_ratio = 1 - self.keep_ratio
        B, L = xb.shape
        P = self.mask_patch_len
        x = xb.view(B, -1, P)
        bs, p_l, p = x.shape

        noise = torch.rand(bs, p_l, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=-1).to(x.device)
        ids_restore = torch.argsort(ids_shuffle, dim=-1).to(x.device)
        seg_l = math.floor(p_l * mask_ratio / 3)
        point_mask_ratio = (mask_ratio * p_l - seg_l * 2) / (p_l - seg_l * 2)
        change_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        ###########
        shuffled_indices = ids_shuffle[:, :seg_l]
        x_shuffled = torch.gather(x, dim=-2, index=shuffled_indices.unsqueeze(-1).repeat(1, 1, p))
        random_indices = torch.rand(bs, seg_l, p, device=x.device).argsort(dim=-1)
        x_shuffled = torch.gather(x_shuffled, dim=-1, index=random_indices)
        ##########
        mean_indices = ids_shuffle[:, seg_l:seg_l * 2]
        mean_indices = mean_indices.unsqueeze(-1).repeat(1, 1, p)
        x_mean = torch.gather(x, dim=-2, index=mean_indices)
        x_mean = x_mean.mean(dim=-1, keepdim=True).repeat(1, 1, p)
        change_mask[:, :seg_l * 2, :] = True
        ##########
        ids_keep = ids_shuffle[:, seg_l * 2:]
        x_kept = torch.gather(x, dim=-2, index=ids_keep.unsqueeze(-1).repeat(1, 1, p))
        x_ = torch.cat([x_shuffled, x_mean, x_kept], dim=-2)
        x_masked = torch.gather(x_, dim=-2, index=ids_restore.unsqueeze(-1).repeat(1, 1, p))
        change_mask = torch.gather(change_mask, dim=-2, index=ids_restore.unsqueeze(-1).repeat(1, 1, p))
        ##########
        indices_to_mask = torch.rand(x.shape, device=x.device) < point_mask_ratio
        indices_to_mask = indices_to_mask * (~change_mask)
        x_masked[indices_to_mask] = 0
        change_mask[indices_to_mask] = True

        x = x_masked.view(B, L)
        change_mask = change_mask.view(B, L)
        return x, change_mask

    def input_process(self, x):
        small_len = self.small_len
        x1 = x[:, :small_len]
        x2 = x[:, small_len:]
        input_x1, mask_x1 = self.mask_patch(x1)
        input_x2, mask_x2 = self.mask_patch(x2)
        x_concat = torch.cat((input_x1, input_x2), dim=1)
        mask_concat = torch.cat((mask_x1, mask_x2), dim=1)

        return x_concat, mask_concat

    def patch_embedding(self, x, patch_len):
        x = x.unfold(dimension=1, size=patch_len, step=patch_len)
        x = self.enc_embedding(x)
        x = self.dropout(x + self.encoder_pos_embed)
        return x

    def per_training(self, x_enc, x_mark_enc):

        x_mark = torch.tensor(x_mark_enc, dtype=torch.bool)
        selected = torch.masked_select(x_enc, x_mark)
        means = selected.mean()

        x_enc = x_enc - means
        selected = selected - means
        stdev = selected.std()
        x_enc /= stdev
        x = x_enc

        x_enc, mask = self.input_process(x_enc)

        x_enc = x_enc * x_mark_enc
        x_enc = self.patch_embedding(x_enc, self.emb_patch_len)

        x_mark = x_mark.unfold(dimension=1, size=self.emb_patch_len, step=self.emb_patch_len)
        x_mark = x_mark[:, :, -1]

        for mod in self.encoder:
            x_enc = mod(x_enc, masked=x_mark)

        dec_out = x_enc * x_mark.unsqueeze(-1)
        dec_out = self.decoder(dec_out)
        B, L, D = dec_out.shape
        dec_out = dec_out.view(B, self.pt_len)

        dec_out = dec_out * mask * x_mark_enc

        loss = (dec_out - x * mask * x_mark_enc) ** 2
        loss = loss.sum() / (mask * x_mark_enc).sum()

        return loss

    def padding_input(self, x):
        b, l, v = x.shape
        # x = x.permute(0, 2, 1)
        x = x.reshape(b * v, l)
        padding_length = self.pt_len - l
        padded_tensor = F.pad(x, (0, padding_length), 'constant', 0)
        x_mark = torch.zeros(b * v, self.pt_len, dtype=torch.bool, device=x.device)
        x_mark[:, l:] = True

        return padded_tensor, x_mark

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        b, l, v = x_enc.shape
        x_enc, x_mark = self.padding_input(x_enc)
        x_enc = self.patch_embedding(x_enc, self.emb_patch_len)

        x_mark = x_mark.unfold(dimension=1, size=self.emb_patch_len, step=self.emb_patch_len)
        x_mark = x_mark[:, :, -1]

        for mod in self.encoder:
            x_enc = mod(x_enc, masked=x_mark)

        enc_out = x_enc * x_mark.unsqueeze(-1)

        enc_out = torch.reshape(
            enc_out, (b, v, enc_out.shape[-2], enc_out.shape[-1]))
        # Decoder

        dec_out = self.head(enc_out)

        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        b, l, v = x_enc.shape
        x_enc, x_mark = self.padding_input(x_enc)
        x_enc = self.patch_embedding(x_enc, self.emb_patch_len)

        x_mark = x_mark.unfold(dimension=1, size=self.emb_patch_len, step=self.emb_patch_len)
        x_mark = x_mark[:, :, -1]
        # Encoder [b,l,p]
        for mod in self.encoder:
            x_enc = mod(x_enc, masked=x_mark)

        enc_out = x_enc * x_mark.unsqueeze(-1)
        enc_out = torch.reshape(
            enc_out, (-1, v, enc_out.shape[-2], enc_out.shape[-1]))

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        b, l, v = x_enc.shape
        x_enc, x_mark = self.padding_input(x_enc)
        x_enc = self.patch_embedding(x_enc, self.emb_patch_len)

        x_mark = x_mark.unfold(dimension=1, size=self.emb_patch_len, step=self.emb_patch_len)
        x_mark = x_mark[:, :, -1]
        # Encoder [b,l,p]
        for mod in self.encoder:
            x_enc = mod(x_enc, masked=x_mark)

        enc_out = x_enc * x_mark.unsqueeze(-1)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        b, l, v = x_enc.shape
        x_enc, x_mark = self.padding_input(x_enc)
        x_enc = self.patch_embedding(x_enc, self.emb_patch_len)

        x_mark = x_mark.unfold(dimension=1, size=self.emb_patch_len, step=self.emb_patch_len)
        x_mark = x_mark[:, :, -1]
        # Encoder [b,l,p]
        for mod in self.encoder:
            x_enc = mod(x_enc, masked=x_mark)

        enc_out = x_enc * x_mark.unsqueeze(-1)
        enc_out = torch.reshape(
            enc_out, (-1, v, enc_out.shape[-2], enc_out.shape[-1]))
        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        if self.task_name == 'per_training':
            dec_out = self.per_training(x_enc, x_mark_enc)
            return dec_out  # [B, L, D]
        else:
            if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
            if self.task_name == 'imputation':
                dec_out = self.imputation(
                    x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
                return dec_out  # [B, L, D]
            if self.task_name == 'anomaly_detection':
                dec_out = self.anomaly_detection(x_enc)
                return dec_out  # [B, L, D]
            if self.task_name == 'classification':
                dec_out = self.classification(x_enc, x_mark_enc)
                return dec_out  # [B, N]
        return None
