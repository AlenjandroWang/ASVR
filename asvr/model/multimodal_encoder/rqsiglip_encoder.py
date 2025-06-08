import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, AutoProcessor, PreTrainedModel, PretrainedConfig
from transformers.image_processing_utils import BaseImageProcessor
import torch.nn.functional as F
from .rqvaesigliptransformer import RQVAESIGLIPTransformerConfig, RQVAESIGLIPTransformer

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import logging

from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig





class RQVAESIGLIPTransformerVisionTower(nn.Module):
    def __init__(self, model_name_or_path, weights_path,delay_load=False):
        super().__init__()
        dtype = torch.bfloat16

        self.config = RQVAESIGLIPTransformerConfig.from_pretrained(model_name_or_path)
        self.vision_tower = RQVAESIGLIPTransformer._from_config(self.config, torch_dtype=dtype)
        self.is_loaded = False
        if not delay_load:
            self.load_pretrained(weights_path)
    

        encoder_path = self.config.rqvaesiglip["pretrained_model"]
        if "siglip-so400m-patch14-384" in encoder_path:  # SigLIP-SO400M-patch14-384
            self.image_processor = CLIPImageProcessor(
                size={"height": 384, "width": 384}, 
                crop_size={"height": 384, "width": 384}, 
                image_mean=[0.5, 0.5, 0.5], 
                image_std=[0.5, 0.5, 0.5]
            )
            self.image_tokens = 729

            # self.config.hidden_size == 1152
        elif "siglip-large-patch16-384" in encoder_path:  # SigLIP-Large-patch16-384
            self.image_processor = CLIPImageProcessor(
                size={"height": 384, "width": 384}, 
                crop_size={"height": 384, "width": 384}, 
                image_mean=[0.5, 0.5, 0.5], 
                image_std=[0.5, 0.5, 0.5]
            )
            self.image_tokens = 576
            # self.config.hidden_size == 1024
        elif "siglip-large-patch16-256" in encoder_path:  # SigLIP-Large-patch16-256
            self.image_processor = CLIPImageProcessor(
                size={"height": 256, "width": 256}, 
                crop_size={"height": 256, "width": 256}, 
                image_mean=[0.5, 0.5, 0.5], 
                image_std=[0.5, 0.5, 0.5]
            )
            self.image_tokens = 256
            # self.config.hidden_size == 1024
        else:
            raise NotImplementedError()
    
    def forward(self, images: torch.Tensor):
        if images.shape[2]!=384:
            images = F.interpolate(images, size=(384, 384), mode='bilinear', align_corners=False)
        vision_output = self.vision_tower.rqvaesiglip.encode_image(images)
        

        image_features_visual, tokens_visual = vision_output[0], vision_output[1]
        image_features_semantic, tokens_semantic = vision_output[2], vision_output[3]

        bs, patch_size, _, dim = image_features_visual.shape
        image_features_visual = torch.reshape(image_features_visual, [bs, patch_size**2, dim])
        # tokens_visual = torch.add(torch.reshape(tokens_visual, [bs, patch_size**2, -1]), text_vocab_size)
        tokens_visual = torch.reshape(tokens_visual, [bs, patch_size**2, -1])
        
        # visual_vocab_size = 32768

        bs, patch_size, _, dim = image_features_semantic.shape
        image_features_semantic = torch.reshape(image_features_semantic, [bs, patch_size**2, dim])
        # tokens_semantic = torch.add(torch.reshape(tokens_semantic, [bs, patch_size**2, -1]), text_vocab_size + visual_vocab_size)
        tokens_semantic = torch.reshape(tokens_semantic, [bs, patch_size**2, -1])

        return image_features_visual, image_features_semantic, tokens_visual, tokens_semantic

    
    def load_pretrained(self, path, ignore_keys=list()):
        if self.is_loaded:
            print('visual tokenizer is already loaded, `load_model` called again, skipping.')
            return
        sd = torch.load(path, map_location="cpu")["state_dict"]

        keys = list(sd.keys())
        for k in keys:
            # print(k)
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        
        params = list(self.vision_tower.rqvaesiglip.parameters())
        try:
            if hasattr(params[0], "ds_status"):
                if params[0].ds_status == ZeroParamStatus.NOT_AVAILABLE:
                    with zero.GatheredParameters(params, modifier_rank=0):
                        # self.vision_tower.rqvaesiglip.load_state_dict(sd, strict=True)
                        self.vision_tower.rqvaesiglip.load_state_dict(sd, strict=False)
                else:
                    # self.vision_tower.rqvaesiglip.load_state_dict(sd, strict=True)
                    self.vision_tower.rqvaesiglip.load_state_dict(sd, strict=False)
            else:
                # self.vision_tower.rqvaesiglip.load_state_dict(sd, strict=True)
                self.vision_tower.rqvaesiglip.load_state_dict(sd, strict=False)

            print("Successfully loaded rqsiglip's pretrained weights!")
        except Exception as e:
            print("Error loading pretrained weights", exc_info=e)
        self.is_loaded = True

