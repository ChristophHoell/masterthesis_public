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

class Head_MDM(nn.Module):
    def __init__(self, args):
        super(Head_MDM, self).__init__()
        self.device = args.device

        self.input_features = args.input_features
        self.latent_dim = args.latent_dim
        self.d_emb = args.d_emb
        self.d_random = args.d_random
        self.d_action = args.d_action

        self.max_seq_len = args.max_seq_len
        self.period = args.period

        self.clip_version = args.clip_version
        self.clip_dim = args.clip_dim

        self.n_head = args.num_heads
        self.num_layers = args.num_layers
        self.activation = args.activation
        self.dropout = args.dropout

        self.vqvae_num_embeddings = args.num_embeddings


        """
        # V1:


        """
        """
        # V2:
        self.embed_clip = nn.Linear(self.clip_dim, self.d_emb * self.max_seq_len).to(self.device)
        self.embed_action = nn.Linear(self.d_action, self.d_emb).to(self.device)
        self.embedding_process = nn.Linear(self.d_emb, self.latent_dim).to(self.device)
        """
        """
        # V3:
        self.embed_clip = nn.Linear(self.clip_dim + self.d_random, self.d_emb * self.max_seq_len).to(self.device)
        self.embed_action = nn.Linear(self.d_action, self.d_emb).to(self.device)
        self.embedding_process = nn.Linear(self.d_emb, self.latent_dim).to(self.device)
        """
        """
        # V4:
        #self.num_embeddings = 128
        self.num_embeddings = 64
        self.vqvae = ConditionalVQVAE(self.clip_dim, self.d_emb, self.num_embeddings, self.d_emb * self.max_seq_len, self.d_random).to(self.device)
        self.embedding_process = nn.Linear(self.d_emb, self.latent_dim).to(self.device)
        """
        
        # V5:
        self.text_embedder = nn.Sequential(
            nn.Linear(self.clip_dim + self.d_random, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.max_seq_len * self.d_emb),
        ).to(self.device)
        self.embedding_process = nn.Linear(self.d_emb, self.latent_dim).to(self.device)
        """
        # V6:
        self.vqvae = ConditionalVQVAE(self.clip_dim, self.d_emb, self.vqvae_num_embeddings, self.d_emb * self.max_seq_len, self.d_random).to(self.device)
        self.embedding_process = nn.Linear(self.d_emb, self.latent_dim).to(self.device)
        """
        #self.embed_action = nn.Linear(self.action_dim, self.d_emb)
        self.input_process = nn.Linear(self.input_features, self.latent_dim).to(self.device)
        self.output_process = nn.Linear(self.latent_dim, self.input_features).to(self.device)
        self.emb_obj = nn.Linear(self.d_random, self.latent_dim, bias = False).to(self.device)


        self.PPE = PeriodicPositionalEncoding(self.latent_dim, period = self.period)
        self.biased_mask = init_biased_mask(n_head = self.n_head, max_seq_len = self.max_seq_len, period = self.period)

        decoder_layer = nn.TransformerDecoderLayer( d_model = self.latent_dim,
                                                    nhead = self.n_head, 
                                                    dim_feedforward = 2 * self.latent_dim, 
                                                    batch_first = True,
                                                    activation = self.activation,
                                                    dropout = self.dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers = self.num_layers)
        self.clip_model = self._load_and_freeze_clip(self.clip_version)


        nn.init.constant_(self.output_process.weight, 0)
        nn.init.constant_(self.output_process.bias, 0)

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

    def forward(self, text, action, vertice, random, temporal_mask):
        """
            text:       list of sentences                       [bs]
            vertice:    Ground Truth output                     [bs, seqlen, V*3] = [bs, seqlen, 15069]     # Currently [bs, V*3, seqlen]
            random:    unique identifier for each sequence      [bs, d_random]
            teacher_forcing: (bool)
        """
        """
        # V1: 
            vertice = vertice.permute(0, 2, 1).to(self.device)
            random = random.to(self.device)
            temporal_mask = temporal_mask.permute(0, 2, 1).to(self.device)

            bs = len(text)
            obj_embedding = self.emb_obj(random)       # [bs, d_model]

            hidden_states = self._encode_text(text).type(torch.get_default_dtype()).unsqueeze(1).repeat(1, self.max_seq_len, 1)                      # [bs, clip_dim]

            for i in range(self.max_seq_len):
                if i == 0:
                    vertice_emb = obj_embedding.unsqueeze(1)        # [bs, 1, d_model]
                    style_emb = vertice_emb                         # [bs, 1, d_model]
                    vertice_input = self.PPE(style_emb)             # [bs, 1, d_model]
                else:
                    vertice_input = self.PPE(vertice_emb)           # [bs, i, d_model]

                #logger.critical(f"Shapes: {vertice_input.shape} - {hidden_states.shape}")
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().repeat(bs, 1, 1).detach().to(self.device)
                memory_mask = enc_dec_mask(self.device, vertice_input.shape[1], hidden_states.shape[1])

                #logger.critical(f"Forward Shapes: {vertice_input.shape} - {hidden_states.shape} - {tgt_mask.shape} - {memory_mask.shape}")
                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask = tgt_mask, memory_mask = memory_mask)
                vertice_out = self.output_process(vertice_out)
                new_output = self.input_process(vertice_out[:, -1, :]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

            #logger.critical(f"Shapes: {vertice_out.shape} - {vertice.shape} - {temporal_mask.shape}")
            losses = {}
            losses["full"] = masked_l2(vertice_out, vertice, temporal_mask).mean()

            return losses
        """
        """
        # V2:
        # Like faceformer Teacher forcing

            losses = {}

            # Prepare Input and move to CUDA
            action = action.to(self.device)
            vertice = vertice.permute(0, 2, 1).to(self.device)
            random = random.to(self.device)
            temporal_mask = temporal_mask.permute(0, 2, 1).to(self.device)

            bs = len(text)

            # Style Embedding
            style_emb = self.emb_obj(random).unsqueeze(1)                # [bs, d_model]

            enc_text = self._encode_text(text).type(torch.get_default_dtype())             # [bs, clip_dim]
            emb_text = self.embed_clip(enc_text).reshape((bs, self.max_seq_len, -1))

            # Action Embedding
            emb_action = self.embed_action(action)

            # Embedding Loss
            losses["emb"] = masked_l2(emb_text, emb_action, temporal_mask).mean()

            # Use Action Embedding as Hidden States during training
            emb = self.embedding_process(emb_action)
            hidden_states = emb
            
            # Emulate template through Zero-Tensor
            template = torch.zeros_like(vertice[:, 0]).unsqueeze(1)
            # Right shift target vector and prepend the zero vector template
            vertice_input = torch.cat((template, vertice[:, :-1]), 1)

            vertice_input = self.input_process(vertice_input)
            vertice_input = vertice_input + style_emb
            vertice_input = self.PPE(vertice_input)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().repeat(bs, 1, 1).detach().to(self.device)
            memory_mask = enc_dec_mask(self.device, vertice_input.shape[1], hidden_states.shape[1])

            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask = tgt_mask, memory_mask = memory_mask)
            vertice_out = self.output_process(vertice_out)

            losses["verts"] = masked_l2(vertice_out, vertice, temporal_mask).mean()

            losses["full"] = losses["verts"] + losses["emb"]
            return losses
        """
        """
        # V3
        # Like faceformer Teacher forcing
            losses = {}

            # Prepare Input and move to CUDA
            action = action.to(self.device)
            vertice = vertice.permute(0, 2, 1).to(self.device)
            random = random.to(self.device)
            temporal_mask = temporal_mask.permute(0, 2, 1).to(self.device)

            bs = len(text)

            # Style Embedding
            style_emb = self.emb_obj(random).unsqueeze(1)                # [bs, d_model]

            # Text Embedding
            enc_text = self._encode_text(text).type(torch.get_default_dtype())             # [bs, clip_dim]
            enc_text = torch.cat((enc_text, random), 1)
            emb_text = self.embed_clip(enc_text).reshape((bs, self.max_seq_len, -1))

            # Action Embedding
            emb_action = self.embed_action(action)
            # TODO: add randomness to action embedding
            #
            # Else: following happens
            #   FF(clip_out, random) tries to match FF(action) 
            #   as FF(action) only contains information on which action is present in the corresponding frame NOT which part of the action should occur
            #

            # Embedding Loss
            losses["emb"] = masked_l2(emb_text, emb_action, temporal_mask).mean()

            # Use Action Embedding as Hidden States during training
            emb = self.embedding_process(emb_action)
            hidden_states = emb
            
            # Emulate template through Zero-Tensor
            template = torch.zeros_like(vertice[:, 0]).unsqueeze(1)
            # Right shift target vector and prepend the zero vector template
            vertice_input = torch.cat((template, vertice[:, :-1]), 1)

            vertice_input = self.input_process(vertice_input)
            vertice_input = vertice_input + style_emb
            vertice_input = self.PPE(vertice_input)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().repeat(bs, 1, 1).detach().to(self.device)
            # TODO: hiddenstates has currently no sequence dimension
            memory_mask = enc_dec_mask(self.device, vertice_input.shape[1], hidden_states.shape[1])

            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask = tgt_mask, memory_mask = memory_mask)
            vertice_out = self.output_process(vertice_out)

            losses["verts"] = masked_l2(vertice_out, vertice, temporal_mask).mean()

            losses["full"] = losses["verts"] + losses["emb"]
            return losses
        """
        """
        # V4:
            losses = {}

            action = action.to(self.device)
            vertice = vertice.permute(0, 2, 1).to(self.device)
            random = random.to(self.device)
            temporal_mask = temporal_mask.permute(0, 2, 1).to(self.device)

            bs = len(text)

            style_emb = self.emb_obj(random).unsqueeze(1)

            # Text Embedding using VQVAE
            enc_text = self._encode_text(text).type(torch.get_default_dtype())
            emb_text, losses["vqvae"] = self.vqvae(enc_text, random)
            emb_text = emb_text.view(-1, self.max_seq_len, self.d_emb)

            # Don't use Action as we should be able to handle randomness form the VQVAE
            emb = self.embedding_process(emb_text)
            hidden_states = emb

            # Emulate template through Zero-Vector:
            template = torch.zeros_like(vertice[:, 0]).unsqueeze(1)
            # Right shift target vector and prepend the zero vector template
            vertice_input = torch.cat((template, vertice[:, :-1]), 1)

            vertice_input = self.input_process(vertice_input)
            vertice_input = vertice_input + style_emb
            vertice_input = self.PPE(vertice_input)

            # Obtain tgt_mask (Added repeat here to handle batched input correctly)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().repeat(bs, 1, 1).detach().to(self.device)
            memory_mask = enc_dec_mask(self.device, vertice_input.shape[1], hidden_states.shape[1])
            

            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask = tgt_mask, memory_mask = memory_mask)
            vertice_out = self.output_process(vertice_out)

            losses["verts"] = masked_l2(vertice_out, vertice, temporal_mask).mean()

            losses["full"] = losses["verts"] + losses["vqvae"]

            return losses
        """
        
        # V5:
        losses = {}

        action = action.to(self.device)
        vertice = vertice.permute(0, 2, 1).to(self.device)
        random = random.to(self.device)
        temporal_mask = temporal_mask.permute(0, 2, 1).to(self.device)

        bs = len(text)

        style_emb = self.emb_obj(random).unsqueeze(1)

        # Text Embedding using VQVAE
        enc_text = self._encode_text(text).type(torch.get_default_dtype())
        enc_text = torch.cat((enc_text, random), dim = 1)
        emb_text = self.text_embedder(enc_text)
        emb_text = emb_text.view(-1, self.max_seq_len, self.d_emb)
        #emb_text, losses["vqvae"] = self.vqvae(enc_text, random)
        #emb_text = emb_text.view(-1, self.max_seq_len, self.d_emb)

        # Don't use Action as we should be able to handle randomness form the VQVAE
        emb = self.embedding_process(emb_text)
        hidden_states = emb

        # Emulate template through Zero-Vector:
        template = torch.zeros_like(vertice[:, 0]).unsqueeze(1)
        # Right shift target vector and prepend the zero vector template
        vertice_input = torch.cat((template, vertice[:, :-1]), 1)

        vertice_input = self.input_process(vertice_input)
        vertice_input = vertice_input + style_emb
        vertice_input = self.PPE(vertice_input)

        # Obtain tgt_mask (Added repeat here to handle batched input correctly)
        tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().repeat(bs, 1, 1).detach().to(self.device)
        memory_mask = enc_dec_mask(self.device, vertice_input.shape[1], hidden_states.shape[1])
        

        vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask = tgt_mask, memory_mask = memory_mask)
        vertice_out = self.output_process(vertice_out)

        losses["verts"] = masked_l2(vertice_out, vertice, temporal_mask).mean()

        losses["full"] = losses["verts"]

        return losses
        
        """
        # V6:
            losses = {}
            vertice = vertice.permute(0, 2, 1).to(self.device)
            random = random.to(self.device)
            temporal_mask = temporal_mask.permute(0, 2, 1).to(self.device)

            bs = len(text)
            obj_embedding = self.emb_obj(random)       # [bs, d_model]

            #hidden_states = self._encode_text(text).type(torch.get_default_dtype()).unsqueeze(1).repeat(1, self.max_seq_len, 1)                      # [bs, clip_dim]

            enc_text = self._encode_text(text).type(torch.get_default_dtype())
            emb_text, losses["vqvae"] = self.vqvae(enc_text, random)
            emb_text = emb_text.view(-1, self.max_seq_len, self.d_emb)
            emb = self.embedding_process(emb_text)
            hidden_states = emb

            for i in range(self.max_seq_len):
                if i == 0:
                    vertice_emb = obj_embedding.unsqueeze(1)        # [bs, 1, d_model]
                    style_emb = vertice_emb                         # [bs, 1, d_model]
                    vertice_input = self.PPE(style_emb)             # [bs, 1, d_model]
                else:
                    vertice_input = self.PPE(vertice_emb)           # [bs, i, d_model]

                #logger.critical(f"Shapes: {vertice_input.shape} - {hidden_states.shape}")
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(self.device)
                memory_mask = enc_dec_mask(self.device, vertice_input.shape[1], hidden_states.shape[1])

                #logger.critical(f"Forward Shapes: {vertice_input.shape} - {hidden_states.shape} - {tgt_mask.shape} - {memory_mask.shape}")
                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask = tgt_mask, memory_mask = memory_mask)
                vertice_out = self.output_process(vertice_out)
                new_output = self.input_process(vertice_out[:, -1, :]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

            #logger.critical(f"Shapes: {vertice_out.shape} - {vertice.shape} - {temporal_mask.shape}")
            losses["verts"] = masked_l2(vertice_out, vertice, temporal_mask).mean()
            losses["full"] = losses["verts"] + losses["vqvae"]

            return losses
        """

    def predict(self, text, random):
        """
        # V1:
            random = random.to(self.device)
            bs = len(text)
            obj_embedding = self.emb_obj(random)
            #logger.critical(f"Obj Emb.shape: {obj_embedding.shape}")

            hidden_states = self._encode_text(text).type(torch.get_default_dtype())

            for i in tqdm(range(self.max_seq_len)):
                if i == 0:
                    vertice_emb = obj_embedding.unsqueeze(1)
                    style_emb = vertice_emb
                    vertice_input = self.PPE(style_emb)
                else:
                    vertice_input = self.PPE(vertice_emb)

                #logger.critical(f"Predict Shapes: {vertice_input.shape} - {hidden_states.shape}")
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().repeat(bs, 1, 1).detach().to(self.device)
                memory_mask = enc_dec_mask(self.device, vertice_input.shape[1], hidden_states.shape[1])
                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask = tgt_mask, memory_mask = memory_mask)
                vertice_out = self.output_process(vertice_out)
                new_output = self.input_process(vertice_out[:,-1,:]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

            return vertice_out.permute(0, 2, 1)
        """
        """
        # V2:
            random = random.to(self.device)
            bs = len(text)

            vertice_emb = self.emb_obj(random).unsqueeze(1)
            style_emb = vertice_emb

            enc_text = self._encode_text(text).type(torch.get_default_dtype())
            emb_text = self.embed_clip(enc_text).reshape((bs, self.max_seq_len, -1))
            
            emb = self.embedding_process(emb_text)
            hidden_states = emb

            for i in tqdm(range(self.max_seq_len)):
                if i == 0:
                    vertice_input = self.PPE(style_emb)
                else:
                    vertice_input = self.PPE(vertice_emb)

                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().repeat(bs, 1,1).detach().to(self.device)
                memory_mask = enc_dec_mask(self.device, vertice_input.shape[1], hidden_states.shape[1])

                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask = tgt_mask, memory_mask = memory_mask)
                vertice_out = self.output_process(vertice_out)
                new_output = self.input_process(vertice_out[:, -1, :]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

            return vertice_out.permute(0, 2, 1)
        """
        """
        # V3
            random = random.to(self.device)
            bs = len(text)

            vertice_emb = self.emb_obj(random).unsqueeze(1)
            style_emb = vertice_emb

            # Text Embedding
            enc_text = self._encode_text(text).type(torch.get_default_dtype())             # [bs, clip_dim]
            enc_text = torch.cat((enc_text, random), 1)
            emb_text = self.embed_clip(enc_text).reshape((bs, self.max_seq_len, -1))

            emb = self.embedding_process(emb_text)
            hidden_states = emb

            for i in tqdm(range(self.max_seq_len)):
                if i == 0:
                    vertice_input = self.PPE(style_emb)
                else:
                    vertice_input = self.PPE(vertice_emb)

                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().repeat(bs, 1,1).detach().to(self.device)
                memory_mask = enc_dec_mask(self.device, vertice_input.shape[1], hidden_states.shape[1])

                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask = tgt_mask, memory_mask = memory_mask)
                vertice_out = self.output_process(vertice_out)
                new_output = self.input_process(vertice_out[:, -1, :]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

            return vertice_out.permute(0, 2, 1)
        """
        """
        # V4:
            random = random.to(self.device)
            bs = len(text)

            vertice_emb = self.emb_obj(random).unsqueeze(1)
            style_emb = vertice_emb

            # Text Embedding:
            enc_text = self._encode_text(text).type(torch.get_default_dtype())
            emb_text, _ = self.vqvae(enc_text, random)
            emb_text = emb_text.view(-1, self.max_seq_len, self.d_emb)

            emb = self.embedding_process(emb_text)
            hidden_states = emb

            for i in tqdm(range(self.max_seq_len)):
                if i == 0:
                    vertice_input = self.PPE(style_emb)
                else:
                    vertice_input = self.PPE(vertice_emb)

                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().repeat(bs, 1, 1).to(self.device)
                memory_mask = enc_dec_mask(self.device, vertice_input.shape[1], hidden_states.shape[1])

                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask = tgt_mask, memory_mask = memory_mask)
                vertice_out = self.output_process(vertice_out)
                new_output = self.input_process(vertice_out[:, -1, :]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

            return vertice_out.permute(0, 2, 1)
        """
        
        # V5
        random = random.to(self.device)
        bs = len(text)

        vertice_emb = self.emb_obj(random).unsqueeze(1)
        style_emb = vertice_emb

        # Text Embedding:
        enc_text = self._encode_text(text).type(torch.get_default_dtype())
        enc_text = torch.cat((enc_text, random), dim = 1)
        emb_text = self.text_embedder(enc_text)
        emb_text = emb_text.view(-1, self.max_seq_len, self.d_emb)

        emb = self.embedding_process(emb_text)
        hidden_states = emb

        for i in tqdm(range(self.max_seq_len)):
            if i == 0:
                vertice_input = self.PPE(style_emb)
            else:
                vertice_input = self.PPE(vertice_emb)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().repeat(bs, 1, 1).to(self.device)
            memory_mask = enc_dec_mask(self.device, vertice_input.shape[1], hidden_states.shape[1])

            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask = tgt_mask, memory_mask = memory_mask)
            vertice_out = self.output_process(vertice_out)
            new_output = self.input_process(vertice_out[:, -1, :]).unsqueeze(1)
            new_output = new_output + style_emb
            vertice_emb = torch.cat((vertice_emb, new_output), 1)

        return vertice_out.permute(0, 2, 1)
        
        """
        # V6:
            random = random.to(self.device)
            bs = len(text)

            vertice_emb = self.emb_obj(random).unsqueeze(1)
            style_emb = vertice_emb

            # Text Embedding:
            enc_text = self._encode_text(text).type(torch.get_default_dtype())
            emb_text, _ = self.vqvae(enc_text, random)
            emb_text = emb_text.view(-1, self.max_seq_len, self.d_emb)
            emb = self.embedding_process(emb_text)
            hidden_states = emb

            for i in tqdm(range(self.max_seq_len)):
                if i == 0:
                    vertice_input = self.PPE(style_emb)
                else:
                    vertice_input = self.PPE(vertice_emb)

                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().repeat(bs, 1, 1).to(self.device)
                memory_mask = enc_dec_mask(self.device, vertice_input.shape[1], hidden_states.shape[1])

                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask = tgt_mask, memory_mask = memory_mask)
                vertice_out = self.output_process(vertice_out)
                new_output = self.input_process(vertice_out[:, -1, :]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

            return vertice_out.permute(0, 2, 1)
        """

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

class ConditionalVQVAE(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_embeddings, output_dim, noise_dim):
        super(ConditionalVQVAE, self).__init__()
        
        """
        # V4:
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )

        self.vq = VQEmbedding(num_embeddings, embedding_dim)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim + noise_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.noise_dim = noise_dim
        """
        # V6:
        self.vq = VQEmbedding(num_embeddings, input_dim)
        self.decoder = nn.Sequential(
            nn.Linear(input_dim + noise_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        )

    def forward(self, x, noise):
        """
        # V4
        z = self.encoder(x)

        z_q, vq_loss, _ = self.vq(z)
        z_q = torch.cat((z_q, noise), dim = -1)
        x_recon = self.decoder(z_q)

        return x_recon, vq_loss
        """
        # V6:
        z_q, vq_loss, _ = self.vq(x)
        z_q = torch.cat((z_q, noise), dim = -1)
        x_recon = self.decoder(z_q)

        return x_recon, vq_loss

class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VQEmbedding, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        # Flatten
        x = x.view(-1, self.embedding_dim)

        # Calculate distance from input to the elements in the Codebook
        distances = (x ** 2).sum(dim = 1, keepdim = True) + (self.embedding.weight ** 2).sum(dim = 1) - 2 * torch.matmul(x, self.embedding.weight.t())

        # Get nearest Codebook element index
        indices = distances.argmin(dim = 1).unsqueeze(1)
        #logger.critical(indices)

        # retrieve nearest codebook element
        x_q = self.embedding(indices).view(*x.size())

        # calculate embedding loss (split 0.75 / 0.25 to update the input and to update the embedding)
        loss = F.mse_loss(x_q.detach(), x) + 0.25 * F.mse_loss(x_q, x.detach())

        # Straight through estimator
        x_q = x + (x_q - x).detach()

        return x_q, loss, indices