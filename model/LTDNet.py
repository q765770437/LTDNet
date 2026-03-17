import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# Assuming these are your custom layers.
# Ideally, for a full open source release, these dependencies should be clearly organized.
from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import MyEncoderForLoss, MyEncoderLayerForloss

class MovingAverage(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    """
    def __init__(self, channels, kernel_size=3):
        super(MovingAverage, self).__init__()
        self.avg = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
            groups=channels
        )
        # Initialize with uniform weights for true moving average
        nn.init.constant_(self.avg.weight, 1.0 / kernel_size)

    def forward(self, x):
        # x shape: [Batch, Channels, Seq_Len]
        return self.avg(x)


class LatentTemporalDecomposition(nn.Module):
    """
    Latent Temporal Decomposition (LTD) Module.
    Decomposes the latent representation into Trend, Seasonal, and Noise components.
    """
    def __init__(self, channels, trend_kernel=3, top_k=10, use_extra_channels=True):
        super(LatentTemporalDecomposition, self).__init__()

        # Determine input channels based on marking (time features)
        self.input_channels = channels + 4 if use_extra_channels else channels

        # Trend extraction layers (Multi-scale moving average)
        self.trend_extractor_small = MovingAverage(self.input_channels, kernel_size=trend_kernel)
        self.trend_extractor_large = MovingAverage(self.input_channels, kernel_size=trend_kernel * 2 - 1)

        self.top_k = top_k

    def forward(self, x):
        # x shape: [Batch, Channels, Seq_Len]

        # 1. Extract Trend Component
        trend_initial = self.trend_extractor_small(x)

        # 2. Calculate Residual
        residual = x - trend_initial

        # 3. Extract Seasonal Component via Frequency Domain
        # Perform FFT
        xf = torch.fft.rfft(residual)
        freq = torch.abs(xf)
        freq[0] = 0 # Remove DC component
        freq = freq.clone()

        # Adaptive Thresholding logic
        freq_mean = torch.mean(freq, dim=-1, keepdim=True)
        freq_std = torch.std(freq, dim=-1, keepdim=True)
        freq_threshold = freq_mean - freq_std

        # Filter noise in frequency domain
        xf[freq >= freq_threshold] = 0

        # Inverse FFT to get smoothed seasonal component
        Trend_in_residual = torch.fft.irfft(xf)

        # Refine seasonal component
        x_seasonal_raw = residual - Trend_in_residual

        # 4. Re-smooth Trend
        trend_refined = x - x_seasonal_raw
        trend_final = self.trend_extractor_large(trend_refined)

        # 5. Extract Noise
        # Typically: Original = Trend + Seasonal + Noise
        # Here: Seasonal is refined from residual
        seasonal_final = x_seasonal_raw
        noise = residual - seasonal_final

        return trend_final, seasonal_final, noise


class ConvolutionalVariableInteraction(nn.Module):
    """
    Lightweight Convolution-based Interaction Mechanism with Long-term Memory.
    Captures cross-variable dependencies using a memory matrix of size 'mem_size'.
    """
    def __init__(self, seq_len, num_channels, kernel_size, mem_size, k_factor):
        super(ConvolutionalVariableInteraction, self).__init__()

        # Dynamic kernel sizing based on k_factor
        kernel_h = int(kernel_size * k_factor)
        kernel_w = kernel_size

        self.conv_inter_1 = nn.Conv2d(1, 1, kernel_size=(kernel_h, kernel_w))
        self.conv_inter_2 = nn.Conv2d(1, 1, kernel_size=(kernel_h, kernel_w))

        # Calculate dimensions after convolution (valid padding)
        self.feat_h = num_channels - (2 * kernel_w) + 2
        self.feat_w = seq_len - int((2 * k_factor * kernel_w)) + 2

        # Linear projections
        # Projecting the convolved time dimension to the Memory Size
        self.projector_memory = nn.Linear(self.feat_w, mem_size)

        # Fusion layers
        self.mlp_fusion_1 = nn.Linear(seq_len + mem_size, seq_len + mem_size)
        self.mlp_fusion_2 = nn.Linear(seq_len + mem_size, seq_len)

    def forward(self, x):
        # x shape: [Batch, Channels, Seq_Len]
        batch_size, channels, seq_len = x.shape

        x_trans = x.permute(0, 2, 1).unsqueeze(1) # [B, 1, L, C]

        # Convolutional Interaction
        x_inter = F.relu(self.conv_inter_1(x_trans))
        x_inter = F.relu(self.conv_inter_2(x_inter))

        # Reshape back: [B, 1, L', C'] -> squeeze -> [B, L', C'] -> permute -> [B, C', L']
        # Here C' is the processed channel dim, L' is the processed time dim
        x_inter = x_inter.squeeze(1).permute(0, 2, 1)

        # Project to Memory Matrix Size
        # Compressing/Mapping the temporal information into a compact memory representation
        x_memory = self.projector_memory(x_inter) # [Batch, Channels, mem_size]

        # Stochastic Pooling / Selection Mechanism
        # Using softmax to select salient features from the memory matrix
        batch_size, current_channels, mem_dim = x_memory.shape
        ratio = F.softmax(x_memory, dim=1) # Softmax over channels
        ratio = ratio.permute(0, 2, 1).reshape(-1, current_channels)

        indices = torch.multinomial(ratio, 1)
        indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)

        combined_features = torch.gather(x_memory, 1, indices)
        combined_features = combined_features.repeat(1, channels, 1)

        # MLP Fusion: Concatenate original input with extracted memory features
        # x: [B, C, L], combined_features: [B, C, mem_size]
        combined_cat = torch.cat([x, combined_features], dim=-1)

        combined_cat = F.gelu(self.mlp_fusion_1(combined_cat))
        output = self.mlp_fusion_2(combined_cat)

        return output


class LTDBlock(nn.Module):
    """
    LTDBlock: The core building block of the Encoder.
    Integrates Latent Temporal Decomposition and Variable Interaction (with Memory).
    """
    def __init__(self, configs):
        super(LTDBlock, self).__init__()

        # Decomposition Module
        use_extra = (configs.mark != 0)
        self.decomposition = LatentTemporalDecomposition(
            configs.enc_in,
            trend_kernel=3,
            top_k=10,
            use_extra_channels=use_extra
        )

        # Interaction Module
        input_dim_for_inter = configs.enc_in + 4 if use_extra else configs.enc_in
        self.interaction = ConvolutionalVariableInteraction(
            configs.d_model,
            input_dim_for_inter,
            configs.cross_kernel_size,
            configs.mem_size,  # Changed from d_core
            configs.k_factor
        )

        # Fusion Layer
        self.fusion_mlp = nn.Linear(configs.d_model * 3, configs.d_model)

    def forward(self, x, *args, **kwargs):
        # x shape: [Batch, d_model, Enc_In] or similar depending on embedding

        # 1. Decompose into components
        trend, seasonal, noise = self.decomposition(x)

        # 2. Capture cross-variable dependencies using Memory Matrix
        interaction_out = self.interaction(x)

        # 3. Fusion
        # Concatenate decomposed components and interaction results
        time_features = torch.cat([trend, seasonal], dim=-1)
        combined = torch.cat([interaction_out, time_features], dim=-1)

        # Final projection
        enc_out = self.fusion_mlp(combined)

        return enc_out, noise


class Model(nn.Module):
    """
    LTDNet: Latent Temporal Decomposition Network.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        # Embedding Layer
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.dropout
        )

        # Encoder
        # Using the custom Encoder wrapper that handles the loss return
        self.encoder = MyEncoderForLoss(
            [
                MyEncoderLayerForloss(
                    LTDBlock(configs), # The core LTD mechanism
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
        )

        # Output Projection
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization (RevIN logic usually fits here, simplifying to standard norm)
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoding with Decomposition and Interaction
        enc_out, loss = self.encoder(enc_out, attn_mask=None)

        # Projection to prediction horizon
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, loss

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        dec_out, noise_loss = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :], noise_loss


