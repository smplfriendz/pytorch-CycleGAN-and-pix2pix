import cv2 as cv
from typing import NamedTuple, Tuple
import numpy as np

class TransformParams(NamedTuple):
    x0: int
    x1: int
    y0: int
    y1: int
    flip: bool

def get_transform_params(size: Tuple[int], crop_width: int, crop_height: int, no_flip: bool) -> TransformParams:
    max_x = size[0] - crop_width
    max_y = size[1] - crop_height

    x = np.random.randint(0, max_x) if max_x > 0 else 0
    y = np.random.randint(0, max_y) if max_y > 0 else 0
    flip = False if no_flip else np.random.randint(0, 1)

    return TransformParams(x, x + crop_width, y, y + crop_height, flip)


def transform_image(img: np.ndarray, size: Tuple[int], params: TransformParams):
    num_channels = img.shape[0]

    upsampled = np.zeros((num_channels, *size), dtype=img.dtype)
    for i in range(num_channels):
        upsampled[i] = cv.resize(img[i], size, interpolation=cv.INTER_CUBIC)

    cropped = upsampled[:, params.y0 : params.y1, params.x0 : params.x1]
    flipped = cv.flip(cropped) if params.flip else cropped

    return flipped

def take_single_channel(img, channel_idx, num_channels):
    channel = img[channel_idx]
    if num_channels == 3:
        out = np.zeros((3, *channel.shape), dtype=channel.dtype)
        out[0:,:,] = channel
        out[1,:,:] = channel
        out[2,:,:] = channel
    else:
        out = np.expand_dims(channel, axis=0)

    return out