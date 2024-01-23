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
                'post_process_mask': (YES_NO, {'default': NO}),
            },
        }

    RETURN_TYPES = ('IMAGE', 'MASK')
    FUNCTION = 'remove_background'
    CATEGORY = 'image'

    def remove_background(self, image, model, post_process_mask):
        post_process_mask = yes_no_to_bool(post_process_mask)
        image = image.cpu().numpy()
        image_dtype = image.dtype
        image = np.clip(image * 0xFF, 0, 0xFF).astype(np.uint8)

        if self._CACHED_SESSION[0] != model:
            self._CACHED_SESSION[0] = model
            self._CACHED_SESSION[1] = rembg.new_session(model)
        rembg_session = self._CACHED_SESSION[1]
        rembg_image = rembg.remove(
            image,
            putalpha=True,
            post_process_mask=post_process_mask,
            session=rembg_session
        )
        rembg_image = np.clip(rembg_image / 0xFF, 0, 0xFF).astype(image_dtype)
        rembg_mask = rembg_image[..., -1]
        rembg_image = rembg_image[..., :3] * rembg_mask[..., np.newaxis]
        rembg_image = torch.from_numpy(rembg_image[np.newaxis, ...])
        rembg_mask = torch.from_numpy(rembg_mask[np.newaxis, ...])
        return (rembg_image, rembg_mask)


NODE_CLASS_MAPPINGS = {
    'Remove Background': ImageRemoveBackgroundNode
}
