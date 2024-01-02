import numpy as np
import matplotlib
import os
from cv2 import imwrite, imread, pyrDown, resize, INTER_CUBIC, INTER_AREA
import torch
import time
from tqdm import tqdm
from constants import *

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def create_dirs(paths):
    for path in paths:
        create_dir(path)

create_dirs(['tests',
    zerodepth_tests_folderpath,
    packnet_tests_folderpath,
    zerodepth_pcloud_tests_folderpath,
    packnet_pcloud_tests_folderpath])


def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None,outpath=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    
    if outpath:
        imwrite(outpath, img)   
    
    return img

def intrinsics_matrix(cameradict=sample_camera):
    height = cameradict['height']
    width  = cameradict['width']
    f_norm = cameradict['f_norm'] 
        
    t_h = (height - 1)/2
    t_w = (width - 1)/2 

    ref_dim = max(height,width)

    f_pix = f_norm * ref_dim

    return np.array([[f_pix, 0, t_w], [0, f_pix, t_h], [0, 0, 1]])

def save_pcloud(color_img, depth_data, pcloud_path, f_pix,scale_factor=1.0):

    height,width,_ = color_img.shape
    
    t_h = (height - 1)/2 
    t_w = (width - 1)/2 
    
    with open(pcloud_path,'w+') as writer:
        for c in tqdm(range(width)):
            for l in range(height):
                try:
                    pi = np.array([c-t_w,l-t_h,f_pix])
                    pn = pi / np.linalg.norm(pi)

                    p = pn * depth_data[l,c] * scale_factor

                    color = color_img[l,c]

                    if color[0] == 255 and color[1] == 255 and color[2] == 255:
                        pass
                    else:
                        writer.write(f'{p[0]:.3f},{p[1]:.3f},{p[2]:.3f},{color[2]},{color[1]},{color[0]}\n')
                except:
                    pass

def int_perc_divide(value, scale_percent):
    return int(value * (scale_percent / 100))

def reduce_img_size(img, scale_percent=50):
    width = int_perc_divide(img.shape[1], scale_percent)
    height = int_perc_divide(img.shape[0], scale_percent)
    dim = (width, height)
    return resize(img, dim)

def reduce_img_fixed_width(img, width):
    height = int(width * img.shape[0] / img.shape[1])
    dim = (width, height)
    
    return resize(img, dim)

def simple_tooktime(ref_time,text=None):
    interval = time.time() - ref_time

    if text:
        print(f'{text} took {interval} seconds')

    return interval

def cut_image_aspect_ratio(img, aspect_ratio):
    # thx: https://stackoverflow.com/a/44724368/4436950 , but largely modified

    # get size
    h, w = img.shape[:2]

    # aspect ratio of image
    original_aspect_ratio = w / h

    # compute new dimensions
    if original_aspect_ratio == aspect_ratio:
        return img
    elif original_aspect_ratio > aspect_ratio: # horizontal image
        new_h = h
        new_w = int(h * aspect_ratio)
        pad_left = int((w - new_w) / 2)
        pad_right = w - new_w - pad_left
        cut_img = img[:, pad_left:w - pad_right]
    else: # vertical image
        new_w = w
        new_h = int(w / aspect_ratio)
        pad_top = int((h - new_h) / 2)
        pad_bottom = h - new_h - pad_top
        cut_img = img[pad_top:h - pad_bottom, :]

    return cut_img

class camera_params:

    def __init__(self, params=sample_camera):
        """
         Example of camera_params:

            sample_camera = {
                'height': 3456,
                'width': 5184,
                'height_mm' : 14.9,
                'width_mm' : 22.3,
                'pix_size' : 22.3/5184,
                'f_mm' : 19.2,
                'f_norm' : 19.2/22.3,
                'f_pix' : 5184 * (19.2/22.3),
            }

        """

        self.params = params
        self.ref_dim = max(params['height'],params['width'])
        self.ref_mm = max(params['height_mm'],params['width_mm'])

    def __call__(self):
        return self.params
    
    def __str__(self):
        return str(self.params)

    def resized_version(self, factor):
        params = self.params

        params['height'] = int_perc_divide(params['height'], factor)
        params['width'] = int_perc_divide(params['width'], factor)

        ref_dim = max(params['height'],params['width'])

        params['pix_size'] = self.ref_mm / ref_dim

        params['f_pix'] = params['f_norm'] * ref_dim

        return params
    
    def resized_by_width(self, width):

        ratio = (width / self.params['width'])*100

        return self.resized_version(ratio)

def packnet_as_numpy(depth_pred):
    return depth_pred[0][0][0].cpu().numpy()