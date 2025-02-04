"""
    Utility File

    Enables the loading of the model (without the CLIP-weights)
    Creates a Namespace Object for the necessary parameters of the model
"""
from loguru import logger
from model.Model import Model
from argparse import Namespace

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict = False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith("clip_model.") for k in missing_keys])

def get_model_args(args, data):
    opt = Namespace()
    
    opt.input_features = data.opt.num_vertices * 3
    opt.d_random = data.opt.d_random
    opt.max_seq_len = data.opt.max_seq_len

    opt.clip_version = "ViT-B/32"
    opt.clip_dim = 512

    opt.num_layers = args.num_layers
    opt.num_heads = args.num_heads
    opt.latent_dim = args.latent_dim
    opt.activation = args.activation
    opt.dropout = args.dropout
    opt.period = args.period
    opt.d_emb = args.d_emb
    opt.d_action = 37
    opt.d_temporal = args.d_temporal
    opt.num_embeddings = args.num_embeddings

    opt.teacher_forcing = args.teacher_forcing
    opt.ablation_use_style = args.ablation_use_style

    opt.lambda_weights = {
        "emb": args.lambda_emb,

        "velocity": args.lambda_velocity,

        "vel_face": args.lambda_vel_face,
        "vel_ear": args.lambda_vel_ear,
        "vel_eye": args.lambda_vel_eye,
        "vel_nose": args.lambda_vel_nose,
        "vel_mouth": args.lambda_vel_mouth,
        "vel_rest": args.lambda_vel_rest,

        "face": args.lambda_face,
        "ear": args.lambda_ear,
        "eye": args.lambda_eye,
        "nose": args.lambda_nose,
        "mouth": args.lambda_mouth,
        "rest": args.lambda_rest,
    }

    return opt