import os
import torch

from clip import clip
from src.utils import clip4finetuning
from src.utils.custom_clip import CustomCLIP
from src.models.model import CLIPLinearProj

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_path_if_not_existant(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_state_dict(
        pretrained_on,
        finetuned_on,
        clip_v_encoder,
        init_strategy,
        lossname,
        epoch
        ):
    
    return torch.load(
        os.path.join(
            'models',
            pretrained_on,
            'finetuned',
            finetuned_on,
            clip_v_encoder,
            init_strategy,
            lossname,
            f'model_checkpoint_epoch_{epoch}.pt')
        )

def load(
        model_type: str,
        clip_v_encoder: str,
        init_strategy: str = None,
        lossname: str = None,
        epoch: str = None,
    ):

    assert model_type in ['pretrained', 'finetuned']
    assert clip_v_encoder in ['ViT-B32', 'ViT-L14@336px']

    pretrained_checkpoint  = f"models/openai/pretrained/{clip_v_encoder}.pt"
    state_dict = torch.load(pretrained_checkpoint,
                            map_location=torch.device('cpu'),
                            )["model_state_dict"]
    
    if model_type == 'pretrained':
        model = clip.build_model(state_dict)
        transform = clip._transform(model.visual.input_resolution)
        model = CustomCLIP(model, None)
        return model, transform
    
    assert init_strategy in ['random', 'identity', 'clip_crop', 'clip_pca']
    assert lossname in ["sym_xe", "nt_xent", "sym_xe+cmd"]

    if model_type == 'finetuned':
        base = clip4finetuning.build_model(state_dict)
        transform = clip._transform(base.visual.input_resolution)
        dict_ = load_state_dict(
                                'openai',
                                'coco',
                                clip_v_encoder,
                                init_strategy,
                                lossname,
                                epoch,
                                )
        config         = dict_['config']
        mlp            = CLIPLinearProj(**config)
        mlp.load_state_dict(dict_['model_state_dict'])
        model          = CustomCLIP(base, mlp)
        return model, transform


def load_state_dict(model_name, pretrained_on, finetuned_on, lossname, 
                    train_dim, visual_encoder, epochs):
    
    return torch.load(os.path.join('..', 'finetuning', 'finetuning', 'models', model_name,  
                        pretrained_on, finetuned_on, lossname, 
                        f'{train_dim}-2', f'{visual_encoder}-e{epochs}.pt'))
    

def get_model(model_name:str, model_type:str, pretrained_on:str, finetuned_on:str, 
              lossname:str, train_dim:int, visual_encoder:str, epochs:int):
    
    from finetuning.model import Contrastive

    if model_name == 'clip':
        from finetuning.clip import clip
        from my_clip import MyClip
        if model_type == 'pretrained':
            from finetuning.clip.model import build_model
            state_dict     = torch.load(os.path.join('..', 'finetuning', 'clip', 'models', 
                                        pretrained_on, f'{visual_encoder}.pt'))['model_state_dict']
            base           = build_model(state_dict)
            mlp            = None
            lossname       = 'cliploss'
        if model_type == 'finetuned':
            if 'laion' in pretrained_on:
                from finetuning.clip.model4finetuning import build_model
                state_dict         = torch.load(os.path.join('..', 'finetuning', 'clip', 'models', 
                                        pretrained_on, f'{visual_encoder}.pt'))['model_state_dict']
                base               = build_model(state_dict)
                dict_              = load_state_dict(model_name, pretrained_on, 
                                                     finetuned_on, lossname, 
                                                     train_dim, visual_encoder, 
                                                     epochs)
            if pretrained_on == 'openai':
                # from finetuning.clip.model4finetuning import build_model
                clip_v_encoder = {'ViT-B32':'ViT-B/32', 'RN50':'RN50', 'ViT-L14@336px':'ViT-L/14@336px'}
                # state_dict     = torch.load(os.path.join('..', 'finetuning', 'clip', 'models', 
                #                         pretrained_on, f'{visual_encoder}.pt'))['model_state_dict']
                # base, _        = build_model(state_dict)
                base, _ = clip.load(clip_v_encoder[visual_encoder], finetuning=True)
                dict_          = load_state_dict(model_name, pretrained_on, 
                                                finetuned_on, lossname, 
                                                train_dim, visual_encoder, 
                                                epochs)
            params         = dict_['model_params']
            mlp            = Contrastive(**params)
            
            mlp.load_state_dict(dict_['model_state_dict'])
        model          = MyClip(base, mlp)

    if model_name == 'align':
        from my_align import MyAlign
        if model_type == 'pretrained':
            model = MyAlign()
    return model
