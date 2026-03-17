import torch.nn as nn
import torch
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x



class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None,seq_len=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.norm1 = nn.LayerNorm(d_model)
        self.seq_len = seq_len

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.trand_function = nn.Linear(seq_len, seq_len)
        # self.trand_function_2 = nn.Linear(d_ff, 96)


    def forward(self, x, attn_mask=None, tau=None, delta=None):


        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )

        x = x + new_x
        x = self.norm1(x)
        y = x


        return y


class MyEncoderLayerForloss(nn.Module):
    def __init__(self, attention, d_model, d_ff=None,seq_len=None, dropout=0.1, activation="relu"):
        super(MyEncoderLayerForloss, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.norm1 = nn.LayerNorm(d_model)
        self.seq_len = seq_len

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.trand_function = nn.Linear(seq_len, seq_len)
        # self.trand_function_2 = nn.Linear(d_ff, 96)


    def forward(self, x, attn_mask=None, tau=None, delta=None):


        new_x, loss = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )

        x = x + new_x
        x = self.norm1(x)
        y = x


        return y,loss


class MyEncoderLayerForVision(nn.Module):
    def __init__(self, attention, d_model, d_ff=None,seq_len=None, dropout=0.1, activation="relu"):
        super(MyEncoderLayerForVision, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.norm1 = nn.LayerNorm(d_model)
        self.seq_len = seq_len

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.trand_function = nn.Linear(seq_len, seq_len)
        # self.trand_function_2 = nn.Linear(d_ff, 96)


    def forward(self, x, attn_mask=None, tau=None, delta=None):


        new_x, loss, v = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )

        x = x + new_x
        x = self.norm1(x)
        y = x


        return y,loss, v


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, L, -1)
        return self.out_projection(out)

class EncoderLayer1(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", **kwargs):
        super(EncoderLayer1, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        new_x, _ = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta,
            **kwargs
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)

class MyV2EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None,seq_len=None, dropout=0.1, activation="relu"):
        super(MyV2EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.norm1 = nn.LayerNorm(d_model)
        self.seq_len = seq_len

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "relu" else F.gelu
        # self.trand_function = nn.Linear(seq_len, seq_len)
        # self.trand_function_2 = nn.Linear(d_ff, 96)


    def forward(self, x,core, attn_mask=None, tau=None, delta=None):
        k = torch.matmul(x, core)
        new_x, attn = self.attention(
            x, k, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )

        x = x + new_x
        x = self.norm1(x)
        y = x


        return y


# 更改 局部注意力
class MyV6EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None,seq_len=None, dropout=0.1, activation="relu"):
        super(MyV6EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.norm1 = nn.LayerNorm(d_model)
        self.seq_len = seq_len

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "relu" else F.gelu
        # self.trand_function = nn.Linear(seq_len, seq_len)
        # self.trand_function_2 = nn.Linear(d_ff, 96)


    def forward(self, x,core, attn_mask=None, tau=None, delta=None):
        k = torch.matmul(x, core)
        new_x, attn = self.attention(
            x, k, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )

        x = x + new_x
        x = self.norm1(x)
        y = x


        return y

class MyV6GlobalEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None,seq_len=None, dropout=0.1, activation="relu"):
        super(MyV6GlobalEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.norm1 = nn.LayerNorm(d_model)
        self.seq_len = seq_len

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "relu" else F.gelu
        # self.trand_function = nn.Linear(seq_len, seq_len)
        # self.trand_function_2 = nn.Linear(d_ff, 96)


    def forward(self, q,k,v, attn_mask=None, tau=None, delta=None):
        old = v
        new_x, attn = self.attention(
            q, k, v,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )

        old =  old + new_x
        x = self.norm1(old)


        return x


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x= attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, **kwargs)
                # attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class MyEncoderForLoss(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(MyEncoderForLoss, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, loss= attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, **kwargs)
                if loss is not None:
                    attns.append(loss)

        if self.norm is not None:
            x = self.norm(x)
        if len(attns) > 0:
            loss = torch.stack(attns).mean()
        else:
            loss = torch.tensor(0.0, device=x.device)
        return x, loss

class MyEncoderForVision(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(MyEncoderForVision, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        # x [B, L, D]
        attns = []
        vl = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, loss,v= attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, **kwargs)
                if loss is not None:
                    attns.append(loss)
                if v is not None:
                    vl.append(v)

        if self.norm is not None:
            x = self.norm(x)
        if len(attns) > 0:
            loss = torch.stack(attns).mean()
        else:
            loss = torch.tensor(0.0, device=x.device)
        return x, loss, vl

class MyV2Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(MyV2Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, core, attn_mask=None, tau=None, delta=None, **kwargs):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x= attn_layer(x, core, attn_mask=attn_mask, tau=tau, delta=delta, **kwargs)
                # attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class MyV6GlobalEncoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(MyV6GlobalEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, q, k, v, attn_mask=None, tau=None, delta=None, **kwargs):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(q, k, v, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x= attn_layer(q, k, v, attn_mask=attn_mask, tau=tau, delta=delta, **kwargs)
                # attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
