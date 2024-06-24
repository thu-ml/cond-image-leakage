import os
import torch
from collections import OrderedDict
from base_utils import PrintContext

def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k, v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]] = state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model


def model_load(model_context, config, **kwargs):
    model = model_context["model"]
    assert os.path.exists(config.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, config.ckpt_path)
    return model_context


from lvdm.modules.attention import TemporalTransformer
from lvdm.modules.networks.openaimodel3d import TemporalConvBlock
# trainable modules for optimizer
def freeze_layers(model, layer_types_to_freeze=(TemporalTransformer, TemporalConvBlock)):
        for name, module in model.model.diffusion_model.named_modules():
            if isinstance(module, tuple(layer_types_to_freeze)):
                for param in module.parameters():
                    param.requires_grad = False

def get_trainable_params(model_context, config, **kwargs):
    with PrintContext(f"{'='*20} get trainable params {'='*20}"):
        model = model_context["model"]
        freeze_layers(model, (TemporalTransformer, TemporalConvBlock))
        trainable_params = list(model.image_proj_model.parameters()) + [param for param in model.model.diffusion_model.parameters() if param.requires_grad]
    
    return trainable_params