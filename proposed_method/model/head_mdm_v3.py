import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import math
import numpy as np
import copy

from loguru import logger
from tqdm import tqdm

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, T, S):
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, i] = 0
    return (mask==1).to(device=device)

def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))

def masked_l2(a, b, mask):
    """
    Calculates the l2 (MSE) loss between a and b given mask
    -> calculates the l2 difference between a nd b
    -> calculates the loss sum of diff * mask (removes masked elements)
    -> calculates average with respect to non zero mask elements (number of elements "visible")
    """

    l2_loss = lambda a, b: (a - b) ** 2

    loss = l2_loss(a, b)
    loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
    n_entries = a.shape[1]
    non_zero_elements = sum_flat(mask) * n_entries
    mse_loss_val = loss / non_zero_elements
    return mse_loss_val

def kl_div_loss(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

class Head_MDM(nn.Module):
    def __init__(self, args):
        super(Head_MDM, self).__init__()
        self.device = args.device

        self.input_features = args.input_features
        self.latent_dim = args.latent_dim

        self.max_seq_len = args.max_seq_len

        self.clip_version = args.clip_version
        self.clip_dim = args.clip_dim
        self.clip_model = self._load_and_freeze_clip(self.clip_version)

        self.input_process = nn.Linear(self.input_features, self.latent_dim).to(self.device)
        self.output_process = nn.Linear(self.latent_dim, self.input_features).to(self.device)

        self.encoder = Encoder_TRANSFORMER(args)
        self.decoder = Decoder_TRANSFORMER(args)


        """
        self.vqvae_num_embeddings = args.num_embeddings

        self.cvae = CVAE(self.max_seq_len * self.d_action, self.clip_dim,  self.latent_dim).to(self.device)
        self.emb_macro = nn.Linear(self.d_action, self.latent_dim - self.d_random).to(self.device)
        #self.emb_macro = nn.Linear(self.d_action, self.latent_dim).to(self.device)
        self.input_process = nn.Linear(self.input_features, self.latent_dim).to(self.device)
        self.output_process = nn.Linear(self.latent_dim, self.input_features).to(self.device)
        self.emb_obj = nn.Linear(self.d_random, self.latent_dim, bias = False).to(self.device)

        self.pe = PositionalEncoding(self.latent_dim, self.dropout)
        self.PPE = PeriodicPositionalEncoding(self.latent_dim, period = self.period)
        self.biased_mask = init_biased_mask(n_head = self.n_head, max_seq_len = self.max_seq_len, period = self.period)

        encoder_layer = nn.TransformerEncoderLayer( d_model = self.latent_dim,
                                                    nhead = self.n_head,
                                                    dim_feedforward = 2 * self.latent_dim,
                                                    batch_first = True,
                                                    activation = self.activation,
                                                    dropout = self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = self.num_layers)

        decoder_layer = nn.TransformerDecoderLayer( d_model = self.latent_dim,
                                                    nhead = self.n_head, 
                                                    dim_feedforward = 2 * self.latent_dim, 
                                                    batch_first = True,
                                                    activation = self.activation,
                                                    dropout = self.dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers = self.num_layers)
        


        nn.init.constant_(self.output_process.weight, 0)
        nn.init.constant_(self.output_process.bias, 0)
        """

    def parameters_wo_clip(self):
        """
            Returns the list of named parameters without the CLIP params
        """
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def _load_and_freeze_clip(self, clip_version):
        """
            Loads the specified CLIP version and freezes its weights to prevent training
        """
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def _encode_text(self, raw_text):
        """
            encodes the raw_text using clip
        """

        device = next(self.parameters()).device

        # max_text_len = 50       # Test
        max_text_len = None

        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2
            assert context_length < default_context_length

            texts = clip.tokenize(raw_text, context_length = context_length, truncate = True).to(device)
            zero_pad = torch.zeros([texts.shape[0], default_context_length - context_length], dtype = texts.dtype, device = texts.device)
            texts = torch.cat([texts, zero_pad], dim = 1)
        else:
            texts = clip.tokenize(raw_text, truncate = True).to(device)
        return self.clip_model.encode_text(texts).float()

    def reparametrize(self, mu, logvar, seed = None):
        std = torch.exp(logvar / 2)

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device = self.device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z

    def forward(self, text, action, vertice, random, temporal_mask):
        """
            text:       list of sentences                       [bs]
            vertice:    Ground Truth output                     [bs, seqlen, V*3] = [bs, seqlen, 15069]     # Currently [bs, V*3, seqlen]
            random:    unique identifier for each sequence      [bs, d_random]
            teacher_forcing: (bool)
        """

        losses = {}
        vertice = vertice.permute(0, 2, 1).to(self.device)
        temporal_mask = temporal_mask.permute(0, 2, 1).to(self.device)

        bs = len(text)
        frame_num = vertice.shape[1]

        enc_text = self._encode_text(text).type(torch.get_default_dtype())

        vertice_input = self.input_process(vertice)

        mu, logvar = self.encoder(vertice_input, enc_text, temporal_mask)
        z = self.reparametrize(mu, logvar)
        logger.critical(f"z Shape: {z.shape}")
        out = self.decoder(z, enc_text, temporal_mask)

        out = self.output_process(out)

        losses["verts"] = masked_l2(out, vertice, temporal_mask).mean()
        losses["kl_div"] = kl_div_loss(mu, logvar)

        losses["full"] = losses["verts"] + losses["kl_div"]

        return losses


        
    def predict(self, text, random):
        bs = len(text)
        enc_text = self._encode_text(text).type(torch.get_default_dtype())

        z = torch.randn((bs, self.max_seq_len, self.latent_dim), device = self.device)
        temporal_mask = torch.ones((bs, self.max_seq_len), dtype=bool, device = self.device)
        out = self.decoder(z, enc_text, temporal_mask)
        #logger.critical(f"Out shape: {out.shape}")
        out = self.output_process(out)

        return out.permute(0, 2, 1)


class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, args):
        super(Encoder_TRANSFORMER, self).__init__()
        self.device = args.device
        self.input_features = args.input_features
        self.latent_dim = args.latent_dim
        self.max_seq_len = args.max_seq_len
        self.d_clip = args.clip_dim

        self.n_heads = args.num_heads
        self.num_layers = args.num_layers
        self.activation = args.activation
        self.dropout = args.dropout

        #self.muQuery = nn.Parameter(torch.randn(self.d_clip, self.latent_dim))
        #self.sigmaQuery = nn.Parameter(torch.randn(self.d_clip, self.latent_dim))
        
        self.mu_layer = nn.Linear(self.d_clip, self.latent_dim)
        self.sigma_layer = nn.Linear(self.d_clip, self.latent_dim)

        encoder_layer = nn.TransformerEncoderLayer( d_model = self.latent_dim,
                                                    nhead = self.n_heads,
                                                    dim_feedforward = 2 * self.latent_dim,
                                                    batch_first = True,
                                                    activation = self.activation,
                                                    dropout = self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = self.num_layers)

        self.pe = PositionalEncoding(self.latent_dim, self.dropout)

    def forward(self, x, y, mask):
        """
        bs, nframes, latent_dim = x.shape

        mu = self.mu_layer(y).unsqueeze(dim = 1)
        sigma = self.sigma_layer(y).unsqueeze(dim = 1)

        #xseq = torch.cat((self.muQuery[y][None], self.sigmaQuery[y][None], x), axis = 0)
        #logger.critical(f"Shapes: {mu.shape} - {sigma.shape} - {x.shape}")
        xseq = torch.cat((mu, sigma, x), axis = 1)

        xseq = self.pe(xseq)
        if len(mask.shape) == 3:
            mask = mask.squeeze(dim = -1)
        musandsigmaMask = torch.ones((bs, 2), dtype = bool, device = self.device)
        #logger.critical(f"Shapes: {musandsigmaMask.shape} - {mask.shape}")
        maskseq = torch.cat((musandsigmaMask, mask), axis = 1)

        final = self.transformer_encoder(xseq, src_key_padding_mask = ~maskseq)
        #final = self.transformer_encoder(xseq)
        #logger.critical(final.shape)
        mu = final[:, 0]
        logvar = final[:, 1]

        return mu, logvar
        """
        bs, nframes, latent_dim = x.shape
        if len(mask.shape) == 3:
            mask = mask.squeeze(dim = -1)
        
        x = self.pe(x)
        final = self.transformer_encoder(x, src_key_padding_mask = ~mask)
        #z = final.mean(axis = 1)
        z = final
        mu = self.mu_layer(z)
        logvar = self.sigma_layer(z)

        logger.critical(f"Mu shape: {mu.shape}, logvar: {logvar.shape}")

        return mu, logvar

class Decoder_TRANSFORMER(nn.Module):
    def __init__(self, args):
        super(Decoder_TRANSFORMER, self).__init__()
        self.device = args.device
        self.input_features = args.input_features
        self.latent_dim = args.latent_dim
        self.max_seq_len = args.max_seq_len
        self.d_clip = args.clip_dim

        self.n_heads = args.num_heads
        self.num_layers = args.num_layers
        self.activation = args.activation
        self.dropout = args.dropout

        #self.actionBiases = nn.Parameter(torch.randn(self.d_clip, self.latent_dim))
        self.actionBiases = nn.Linear(self.d_clip, self.latent_dim)
        self.pe = PositionalEncoding(self.latent_dim, self.dropout)

        decoder_layer = nn.TransformerDecoderLayer( d_model = self.latent_dim,
                                                    nhead = self.n_heads,
                                                    dim_feedforward = 2 * self.latent_dim,
                                                    batch_first = True,
                                                    activation = self.activation,
                                                    dropout = self.dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers = self.num_layers)

    def forward(self, z, y, mask):
        if len(mask.shape) == 3:
            mask = mask.squeeze(dim = -1)
        bs, nframes = mask.shape

        #logger.critical(f"Decoder: {mask.shape}")

        z = z + self.actionBiases(y)
        #z = z.unsqueeze(1)

        timequeries = torch.zeros(bs, nframes, self.latent_dim, device = self.device)
        timequeries = self.pe(timequeries)

        output = self.transformer_decoder(tgt = timequeries, memory = z, tgt_key_padding_mask = ~mask)
        
        return output



# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    """
        Performs a positional encoding
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
            d_model: latent dimension
            
            Initializes the (max_len, d_model) large positional encoding tensor with sin and cos
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)                  #Shape: (5000, 1, 512) -> (max_len, 1, d_model)


    def forward(self, x):
        """
            adds the positional encoding to the input
        """

        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class CVAE(nn.Module):
    def __init__(self, input_dim, text_dim, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + text_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def encode(self, x, text_embedding):
        x = torch.cat((x, text_embedding), dim = -1)
        params = self.encoder(x)
        mu, log_var = torch.chunk(params, 2, dim = -1)
        return mu, log_var

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, text_embedding):
        z = torch.cat((z, text_embedding), dim = -1)
        return self.decoder(z)

    def forward(self, x, text_embedding):
        mu, log_var = self.encode(x, text_embedding)
        z = self.reparametrize(mu, log_var)
        recon_x = self.decode(z, text_embedding)
        return recon_x, mu, log_var

    def sample(self, text_embedding):
        with torch.no_grad():
            z = torch.randn(len(text_embedding), self.latent_dim).to(text_embedding.device)
            samples = self.decode(z, text_embedding)
        return samples