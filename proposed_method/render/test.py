import os
import torch
import clip


def load_and_freeze_clip(clip_version):
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

def encode_text(raw_text, clip_model):
    """
        Encodes the raw text using CLIP
    """
    max_text_len = 50           # MDM limits it here depending on the dataset -> TODO: check what works best

    default_context_length = 77
    context_length = max_text_len + 2
    assert context_length < default_context_length

    texts = clip.tokenize(raw_text, context_length = context_length, truncate = True)
    print(texts)
    zero_pad = torch.zeros([texts.shape[0], default_context_length - context_length], dtype = texts.dtype, device = texts.device)
    texts = torch.cat([texts, zero_pad], dim=1)

    return clip_model.encode_text(texts).float()

def main():
    clip_model = load_and_freeze_clip("ViT-B/32")

    raw_text = ["In the beginning the person is smiling for a short time"]

    enc_text = encode_text(raw_text)





if __name__ == "__main__":
    main()