import os
from .rqsiglip_encoder import RQVAESIGLIPTransformerVisionTower
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .siglip.siglip_encoder import SiglipVisionTower, SiglipVisionTowerS2


def build_vision_tokenizer(vision_tokenizer_cfg,**kwargs):
    model_name_or_path = getattr(vision_tokenizer_cfg, 'mm_vision_tokenizer', getattr(vision_tokenizer_cfg, 'vision_tokenizer', None))
    if model_name_or_path is None:
        return None
    
    weight_path = getattr(vision_tokenizer_cfg, 'vision_tokenizer_weight', None)
  



    print(f"Loading vision tokenizer from: {model_name_or_path}")
    vision_tokenizer = RQVAESIGLIPTransformerVisionTower(model_name_or_path, weight_path,**kwargs)

    vision_tokenizer_cfg.mm_hidden_size = vision_tokenizer.config.hidden_size
    
    return vision_tokenizer


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 'use_s2', False)

    if 'sig' in vision_tower.lower():
        if use_s2:
            return SiglipVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    elif 'clip' in vision_tower.lower():
        if use_s2:
            raise ValueError(f'Currently not supporting S2 for CLIP')
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')


