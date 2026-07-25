import re
from functools import lru_cache
from inspect import signature

import torch

from qwen_vl_utils import process_vision_info
from qwen_vl_utils.vision_process import fetch_image, fetch_video

from src.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
)


def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)

def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data


def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

@lru_cache(maxsize=1)
def _process_vision_info_parameters():
    return set(signature(process_vision_info).parameters)

def _process_vision_info_compat(messages, **requested_kwargs):
    supported_parameters = _process_vision_info_parameters()
    kwargs = {
        key: value
        for key, value in requested_kwargs.items()
        if key in supported_parameters
    }
    return process_vision_info(messages, **kwargs)

def _vision_size_factor(image_patch_size):
    return image_patch_size * 2 if image_patch_size else 28

def get_image_info(image_path, min_pixel, max_pixel, width, height, image_patch_size):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    content = {
        "type": "image", 
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {
            "role": "user", 
            "content": [content]
        }
    ]

    if "image_patch_size" in _process_vision_info_parameters():
        vision_outputs = _process_vision_info_compat(
            messages,
            image_patch_size=image_patch_size,
        )
        image_input = vision_outputs[0]
    else:
        image_input = [fetch_image(content, size_factor=_vision_size_factor(image_patch_size))]

    return image_input[0]

def get_video_info(video_path, min_pixels, max_pixels, width, height, fps, image_patch_size, return_video_metadata=False):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    content = {
        "type": "video", 
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "fps": fps
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {
            "role": "user", 
            "content": [content]
        }
    ]

    if "image_patch_size" in _process_vision_info_parameters():
        vision_outputs = _process_vision_info_compat(
            messages, 
            return_video_kwargs=True, 
            image_patch_size=image_patch_size, 
            return_video_metadata=return_video_metadata
        )
        video_input = vision_outputs[1]
        video_kwargs = vision_outputs[2] if len(vision_outputs) > 2 else {}
    else:
        video, fps = fetch_video(
            content,
            image_factor=_vision_size_factor(image_patch_size),
            return_video_sample_fps=True,
        )
        video_input = [video]
        video_kwargs = {"fps": [fps]}

    return video_input[0], video_kwargs

def samples_per_class_from_ids(label_ids, num_classes):
    
    counts = torch.bincount(
        torch.as_tensor(label_ids, dtype=torch.long),
        minlength=num_classes
    )
    
    return counts.tolist()
