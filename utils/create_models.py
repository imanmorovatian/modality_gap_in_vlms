from models.custom_clip import CustomCLIP
from models.custom_align import CustomALIGN
from models.custom_imagebind import CustomImageBind
from models.custom_cyclip import CustomCyCLIP
from models.custom_flava import CustomFLAVA
from models.custom_albef import CustomALBEF
from models.custom_vista import CustomVISTA


def create_model(name, local_path=None):
    if name == 'ALBEF':
        return CustomALBEF()
    
    elif name == 'FLAVA':
        return CustomFLAVA()
    
    elif name == 'zero_shot_ALIGN' or name == 'ALIGN_LL':
        return CustomALIGN()

    elif name == 'ALIGN_LU':
        return CustomALIGN(frozen_text_encoder=False)
    
    elif name == 'ALIGN_UL':
        return CustomALIGN(frozen_image_encoder=False)
    
    elif name == 'ALIGN_UU':
        return CustomALIGN(frozen_image_encoder=False, frozen_text_encoder=False)
    
    elif name == 'ImageBind':
        return CustomImageBind()
    
    elif name == 'CyCLIP':
        return CustomCyCLIP()
    
    elif name == 'zero_shot_CLIP_RN50':
        return CustomCLIP(vision_encoder='RN50')
    
    elif name == 'zero_shot_CLIP_ViT' or name == 'CLIP_ViT_LL':
        return CustomCLIP(vision_encoder='ViT')

    elif name == 'CLIP_RN50':
        return CustomCLIP(
            vision_encoder='RN50',
            text_encoder_from_local=True,
            image_encoder_from_local=True,
            frozen_projection_layers=True,
            projection_layers_from_local=True,
            local_pretrained_weights_path=local_path)
    
    elif name == 'CLIP_ViT':
        return CustomCLIP(
            vision_encoder='ViT',
            text_encoder_from_local=True,
            image_encoder_from_local=True,
            frozen_projection_layers=True,
            projection_layers_from_local=True,
            local_pretrained_weights_path=local_path)
    
    elif name == 'CLIP_ViT_LU':
        # Freeze the image encoder, while fine tune the text encoder
        return CustomCLIP(vision_encoder='ViT', frozen_text_encoder=False)
    
    elif name == 'CLIP_ViT_UL':
        # Freeze the text encoder, while fine tune the image encoder
        return CustomCLIP(vision_encoder='ViT', frozen_image_encoder=False,)
    
    elif name == 'CLIP_ViT_UU':
        # Fine tune the whole model
        return CustomCLIP(vision_encoder='ViT', frozen_text_encoder=False, frozen_image_encoder=False)
    
    elif name == 'CLIP_ViT_UL+LU':
        # Firtly, finetune the image encoder (using the local weights), and then finetune the text encoder
        return CustomCLIP(
            vision_encoder='ViT',
            frozen_text_encoder=False,
            image_encoder_from_local=True,
            local_pretrained_weights_path=local_path)
    
    elif name == 'CLIP_ViT_LU+UL':
        # Firtly, finetune the text encoder (using the local weights), and then finetune the image encoder
        return CustomCLIP(
            vision_encoder='ViT',
            text_encoder_from_local=True,
            frozen_image_encoder=False,
            local_pretrained_weights_path=local_path)
    
    elif name == 'zero_shot_VISTA':
        return CustomVISTA()
    
    elif name == 'VISTA':
        return CustomVISTA(
            text_encoder_from_local=True,
            image_encoder_from_local=True,
            local_pretrained_weights_path=local_path)
    
    if name == 'VISTA_LU':
        # Freeze the image encoder, while fine tune the text encoder
        return CustomVISTA(frozen_text_encoder=False)
    
    elif name == 'VISTA_UL':
        # Freeze the text encoder, while fine tune the image encoder
        return CustomVISTA(frozen_image_encoder=False)
    
    elif name == 'VISTA_UU':
        # Fine tune the whole model
        return CustomVISTA(frozen_text_encoder=False, frozen_image_encoder=False)
    
    elif name == 'VISTA_UL+LU':
        # Firtly, finetune the image encoder (using the local weights), and then finetune the text encoder
        return CustomVISTA(
            frozen_text_encoder=False,
            image_encoder_from_local=True,
            local_pretrained_weights_path=local_path)
    
    elif name == 'VISTA_LU+UL':
        # Firtly, finetune the text encoder (using the local weights), and then finetune the image encoder
        return CustomVISTA(
            text_encoder_from_local=True,
            frozen_image_encoder=False,
            local_pretrained_weights_path=local_path)

    else:
        raise ValueError('The selected model is not implemented yet')