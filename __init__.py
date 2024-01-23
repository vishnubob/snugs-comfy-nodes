import rembg
from PIL import Image
import torch
import numpy as np

YES = 'yes'
NO = 'no'
YES_NO = (YES, NO)
REMBG_MODELS = tuple(rembg.sessions.sessions_names)

def yes_no_to_bool(val):
    if val == YES:
        return True
    if val == NO:
        return False
    raise ValueError(f"value must be '{yes}' or '{no}'")

class ImageRemoveBackgroundNode:
    _CACHED_SESSION = [None, None]

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'image': ('IMAGE',),
                'model': (REMBG_MODELS, {'default': 'u2net'}),
                'only_mask': (YES_NO, {'default': NO}),
                'post_process_mask': (YES_NO, {'default': NO}),
            },
        }

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'remove_background'
    CATEGORY = 'image'

    def remove_background(self, image, model, only_mask, post_process_mask):
        only_mask = yes_no_to_bool(only_mask)
        post_process_mask = yes_no_to_bool(post_process_mask)
        if self._CACHED_SESSION[0] != model:
            self._CACHED_SESSION[0] = model
            self._CACHED_SESSION[1] = rembg.new_session(model)
        rembg_session = self._CACHED_SESSION[1]
        rembg_output = rembg.remove(
            image.cpu().numpy(),
            only_mask=only_mask,
            post_process_mask=post_process_mask,
            session=rembg_session
        )
        rembg_output = torch.from_numpy(rembg_output)
        return (rembg_output,)

NODE_CLASS_MAPPINGS = {
    'Remove Background': ImageRemoveBackgroundNode
}
