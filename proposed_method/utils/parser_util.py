"""
    Utility File
    Defines all the different possible arguments
"""

from argparse import ArgumentParser
import argparse
import os
import json
from loguru import logger

def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten

    add_data_options(parser)
    add_model_options(parser)
    #add_diffusion_options(parser)

    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ["dataset", "model", "diffusion"]:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    assert os.path.exists(args_path), "Arguments json file not found!"

    with open(args_path, "r") as f:
        model_args = json.load(f)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])
        else:
            logger.warning(f"[WARNING]: was not able to load [{a}], using default value [{args.__dict__[a]}] instead")

    return args

def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError("group_name was not found!")

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument("model_path")
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError("model_path argument must be specified!")

def add_base_options(parser):
    group = parser.add_argument_group("base")
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=8, type=int, help="Batch size during training.")

def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")

def add_model_options(parser):
    group = parser.add_argument_group("model")
    group.add_argument("--num_layers", default = 1, type = int, help = "Number of Attention Layers")
    group.add_argument("--num_heads", default = 4, type = int, help = "Number of Attention Heads")
    group.add_argument("--latent_dim", default = 512, type = int, help = "Transformer Latent Size")
    group.add_argument("--activation", default = "gelu", type = str, help = "String name of the Activation function")
    group.add_argument("--dropout", default = 0.1, type = float, help = "Dropout Probability")
    group.add_argument("--period", default = 30, type = int, help = "Period of the PPE")
    group.add_argument("--d_emb", default = 64, type = int, help = "Intermediate Dimension of the Text Embeddings")
    group.add_argument("--num_embeddings", default = 128, type = int, help = "Number of embeddings in the VQVAE Codebook")
    group.add_argument("--d_temporal", default = 64, type = int, help = "Size of the temporal output of the VQVAE")

    group.add_argument("--lambda_emb", default = 1.0, type = float, help = "Embedding Latent Space Loss")
    # New Lambda weights:
    group.add_argument("--lambda_velocity", default = 0.0, type = float, help = "Vertex Velocity loss factor")
    group.add_argument("--lambda_vel_face", default = 0.0, type = float, help = "Vertex Velocity loss factor for face region")
    group.add_argument("--lambda_vel_ear", default = 0.0, type = float, help = "Vertex Velocity loss factor for ear region")
    group.add_argument("--lambda_vel_eye", default = 0.0, type = float, help = "Vertex Velocity loss factor for eye region")
    group.add_argument("--lambda_vel_nose", default = 0.0, type = float, help = "Vertex Velocity loss factor for nose region")
    group.add_argument("--lambda_vel_mouth", default = 0.0, type = float, help = "Vertex Velocity loss factor for mouth region")
    group.add_argument("--lambda_vel_rest", default = 0.0, type = float, help = "Vertex Velocity loss factor for rest of the head")

    group.add_argument("--lambda_face", default = 0.0, type = float, help = "Vertex - Face loss factor")
    group.add_argument("--lambda_ear", default = 0.0, type = float, help = "Vertex - Ear loss factor")
    group.add_argument("--lambda_eye", default = 0.0, type = float, help = "Vertex - Eye loss factor")
    group.add_argument("--lambda_nose", default = 0.0, type = float, help = "Vertex - Nose loss factor")
    group.add_argument("--lambda_mouth", default = 0.0, type = float, help = "Vertex - Mouth loss factor")
    group.add_argument("--lambda_rest", default = 0.0, type = float, help = "Vertex - Rest of the head loss factor")

    group.add_argument("--teacher_forcing", default = True, type = lambda x: x.lower() in ["true", "1"], help = "Defines if the model should be trained in a teacher-forcing way or an autoregressive one")
    group.add_argument("--ablation_use_style", default = False, type = lambda x: x.lower() in ["true", "1"], help = "Defines if the randomness should be used for the style or not")


def add_data_options(parser):
    group = parser.add_argument_group("dataset")
    # omitted: dataset
    group.add_argument("--data_dir", default = "./data/custom_single", type = str, help = "If empty use default [./data/CelebV]")
    group.add_argument("--split", default = "train", type = str, help = "Defines which dataset split should be used")
    group.add_argument("--mode", default = "train", type = str, choices = ["train", "text_only", "uncond"], help = "Defines how the data should be prepared by the dataset")
    group.add_argument("--load_mode", default = "memory", choices = ["memory", "disk"], help = "Should the whole dataset be loaded into the memory or loaded from disk each time?")

def add_training_options(parser):
    group = parser.add_argument_group("training")
    group.add_argument("--save_dir", required = True, type = str, help = "Path to save checkpoints and results.")
    group.add_argument("--overwrite", action = "store_true", help = "If true will enable to use an already existing save_dir")
    group.add_argument("--train_platform_type", default = "TensorboardPlatform", choices = ["NoPlatform", "ClearmlPlatform", "TensorboardPlatform"], type = str, help = "Choose platform to log results. NoPlatform means no logging")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps. (lr decay)")
    group.add_argument("--lr_step_size", default = 500, type = int, help = "Number of steps with same learning rate")
    group.add_argument("--lr_step_reduction", default = 0.5, type = float, help = "Factor of learning rate reduction after each reduction step")

    """
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    """

    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=200, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=500, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=2000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")

def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=1, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--use_fixseed", default = True, type = lambda x: x.lower() in ["true", "1"], help = "Specifies if a fixed seed should be used for the sampling")
    group.add_argument("--use_batched_sampling", default = True, type = lambda x: x.lower() in ["true", "1"], help = "Specifies if the samples should be generated as part of a batch or individually")
    group.add_argument("--render_sample", default = True, type = lambda x: x.lower() in ["true", "1"], help = "Specifies if the samples should directly be rendered into the mp4 clips")

def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=10.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    # Omitted: action_file
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    # Omitted: action_name

# omitted catgory: edit -> edit_mode, text_condition, prefix_end, suffix_start

def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_mode", default='debug', choices=["debug", "debug_disk", "full", "full_disk"], type=str,
                       help="   debug: very small evaluation to check functionality \
                                debug_disk: very small evaluation with samples present on disk \
                                full: Full evaluation with newly generated samples \
                                full_disk: evaluation with samples on disk")
    group.add_argument("--filter_region_pickle", default = "./eval/vertex_id_to_face_region.pkl", type = str, help = "Path to the Pickle file containing a naming for the vertices")

def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    #add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()

def generate_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    # omitted: get_cond_mode(args)

    #return parser.parse_args()
    return args

# omitted: edit_args()

def evaluation_parser():
    parser = ArgumentParser()
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)