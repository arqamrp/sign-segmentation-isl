# Importing Libraries

import os
import sys
import math
import pickle as pkl
import shutil
import argparse
from pathlib import Path
import torch.nn as nn
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.special
import matplotlib.pyplot as plt
from scipy.io import loadmat
from zsvision.zs_utils import BlockTimer
from tqdm import tqdm
import scipy.misc
import scipy.ndimage
import scipy.io
import pickle
from math import floor
import datetime
from PIL import Image, ImageDraw, ImageFont
import copy
from beartype import beartype

# UTILS FUNCTIONS:

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError(f"Cannot convert {type(tensor)} to numpy array")
    return tensor

def save_pred(features, logits, checkpoint: Path, filename=Path("i3d_dict_valid.mat")):
        
    features = to_numpy(features)
    logits = to_numpy(logits)
    checkpoint.mkdir(exist_ok=True, parents=True)
    filepath = os.path.join(checkpoint, filename)
    mdict = {"logits": logits, 'features': features}
    print(f"Saving to {filepath}")
    scipy.io.savemat(filepath, mdict=mdict, do_compression=False, format="4")

def torch_to_list(torch_tensor):
    return torch_tensor.cpu().numpy().tolist()

def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError(f"Cannot convert {type(ndarray)} to torch tensor")
    return ndarray

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError(f"Cannot convert {type(tensor)} to numpy array")
    return tensor

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def resize_generic(img, oheight, owidth, interp="bilinear", is_flow=False):
    """
    Args
    inp: numpy array: RGB image (H, W, 3) | video with 3*nframes (H, W, 3*nframes)
          |  single channel image (H, W, 1) | -- not supported:  video with (nframes, 3, H, W)
    """

    # resized_image = cv2.resize(image, (100, 50))
    ht, wd, chn = img.shape[0], img.shape[1], img.shape[2]
    if chn == 1:
        resized_img = scipy.misc.imresize(
            img.squeeze(), [oheight, owidth], interp=interp, mode="F"
        ).reshape((oheight, owidth, chn))
    elif chn == 3:
        # resized_img = scipy.misc.imresize(img, [oheight, owidth], interp=interp)  # mode='F' gives an error for 3 channels
        resized_img = cv2.resize(img, (owidth, oheight))  # inverted compared to scipy
    elif chn == 2:
        # assert(is_flow)
        resized_img = np.zeros((oheight, owidth, chn), dtype=img.dtype)
        for t in range(chn):
            # resized_img[:, :, t] = scipy.misc.imresize(img[:, :, t], [oheight, owidth], interp=interp)
            # resized_img[:, :, t] = scipy.misc.imresize(img[:, :, t], [oheight, owidth], interp=interp, mode='F')
            # resized_img[:, :, t] = np.array(Image.fromarray(img[:, :, t]).resize([oheight, owidth]))
            resized_img[:, :, t] = scipy.ndimage.interpolation.zoom(
                img[:, :, t], [oheight, owidth]
            )
    else:
        in_chn = 3
        # Workaround, would be better to pass #frames
        if chn == 16:
            in_chn = 1
        if chn == 32:
            in_chn = 2
        nframes = int(chn / in_chn)
        img = img.reshape(img.shape[0], img.shape[1], in_chn, nframes)
        resized_img = np.zeros((oheight, owidth, in_chn, nframes), dtype=img.dtype)
        for t in range(nframes):
            frame = img[:, :, :, t]  # img[:, :, t*3:t*3+3]
            frame = cv2.resize(frame, (owidth, oheight)).reshape(
                oheight, owidth, in_chn
            )
            # frame = scipy.misc.imresize(frame, [oheight, owidth], interp=interp)
            resized_img[:, :, :, t] = frame
        resized_img = resized_img.reshape(
            resized_img.shape[0], resized_img.shape[1], chn
        )

    if is_flow:
        # print(oheight / ht)
        # print(owidth / wd)
        resized_img = resized_img * oheight / ht
    return resized_img

def color_normalize(x, mean, std):
    """Normalize a tensor of images by subtracting (resp. dividing) by the mean (resp.
    std. deviation) statistics of a dataset in RGB space.
    """
    if x.dim() in {3, 4}:
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        assert x.size(0) == 3, "For single video format, expected RGB along first dim"
        for t, m, s in zip(x, mean, std):
            t.sub_(m)
            t.div_(s)
    elif x.dim() == 5:
        assert (
            x.shape[1] == 3
        ), "For batched video format, expected RGB along second dim"
        x[:, 0].sub_(mean[0]).div_(std[0])
        x[:, 1].sub_(mean[1]).div_(std[1])
        x[:, 2].sub_(mean[2]).div_(std[2])
    return x

def get_labels_start_end_time(frame_wise_labels, bg_class=["Sign"]):
    """get list of start and end times of each interval/ segment.

    Args:
        frame_wise_labels: list of framewise labels/ predictions.
        bg_class: list of all classes in frame_wise_labels which should be ignored

    Returns:
        labels: list of labels of the segments
        starts: list of start times of the segments
        ends: list of end times of the segments
    """
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends

def generate_vtt_file(all_preds, logits, save_path):
    vtt = WebVTT()
    predictions = all_preds

    labels, starts, ends = get_labels_start_end_time(predictions, [1])

    # smaller boundaries
    for ix in range(len(labels)):
        if ix == len(labels)-1:
            break
        diff = starts[ix+1]-ends[ix]
        starts[ix+1] -= floor(diff/2)
        ends[ix] += floor(diff/2)
    
    if logits is not None:
        # load i3d classes
        i3d_scores = logits
        with open('data/info/bslcp/info.pkl', 'rb') as f:
            info_data = pickle.load(f)

    # for start, end in zip(starts, ends):
    for start, end in zip(starts, ends):

        if logits is not None:
            i3d_score = np.sum(np.asarray(i3d_scores)[start:end], axis=0)
            ind = np.argpartition(i3d_score, -10)[-10:]       
            ind = ind[np.argsort(-i3d_score[ind])]
            classes = [info_data['words'][ix] for ix in ind]

            class_str = ','.join(classes)
        else:
            class_str = ''

        start = (start + 8) / 25
        end = (end + 8) / 25

        start_dt = datetime.timedelta(seconds=start)
        start_str = str(start_dt)
        if '.' not in start_str:
            start_str = f'{start_str}.000000'

        end_dt = datetime.timedelta(seconds=end)
        end_str = str(end_dt)
        if '.' not in end_str:
            end_str = f'{end_str}.000000'
        # creating a caption with a list of lines
        caption = Caption(
            start_str,
            end_str,
            [class_str]
        )

        # adding a caption
        vtt.captions.append(caption)


    # save to a different file
    vtt.save(f'{save_path}/demo.vtt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")

def load_rgb_video(video_path: Path, fps: int) -> torch.Tensor:
    """
    Load frames of a video using cv2.
    """
    cap = cv2.VideoCapture(str(video_path))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    # cv2 won't be able to change frame rates for all encodings, so we use ffmpeg
    if cap_fps != fps:
        tmp_video_path = f"{video_path}.tmp.{video_path.suffix}"
        shutil.move(video_path, tmp_video_path)
        cmd = (f"ffmpeg -i {tmp_video_path} -pix_fmt yuv420p "
               f"-filter:v fps=fps={fps} {video_path}")
        print(f"Generating new copy of video with frame rate {fps}")
        os.system(cmd)
        Path(tmp_video_path).unlink()
        cap = cv2.VideoCapture(str(video_path))
        cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        assert cap_fps == fps, f"ffmpeg failed to produce a video at {fps}"

    f = 0
    rgb = []
    while True:
        # frame: BGR, (h, w, 3), dtype=uint8 0..255
        ret, frame = cap.read()
        if not ret:
            break
        # BGR (OpenCV) to RGB (Torch)
        frame = frame[:, :, [2, 1, 0]]
        rgb_t = im_to_torch(frame)
        rgb.append(rgb_t)
        f += 1
    cap.release()
    # (nframes, 3, cap_height, cap_width) => (3, nframes, cap_height, cap_width)
    rgb = torch.stack(rgb).permute(1, 0, 2, 3)
    print(f"Loaded video {video_path} with {f} frames [{cap_height}hx{cap_width}w] res. "
          f"at {cap_fps}")
    return rgb

def prepare_input(
    rgb: torch.Tensor,
    resize_res: int = 256,
    inp_res: int = 224,
    mean: torch.Tensor = 0.5 * torch.ones(3), std=1.0 * torch.ones(3),
):
    """
    Process the video:
    1) Resize to [resize_res x resize_res]
    2) Center crop with [inp_res x inp_res]
    3) Color normalize using mean/std
    """
    iC, iF, iH, iW = rgb.shape
    rgb_resized = np.zeros((iF, resize_res, resize_res, iC))
    for t in range(iF):
        tmp = rgb[:, t, :, :]
        tmp = resize_generic(
            im_to_numpy(tmp), resize_res, resize_res, interp="bilinear", is_flow=False
        )
        rgb_resized[t] = tmp
    rgb = np.transpose(rgb_resized, (3, 0, 1, 2))
    # Center crop coords
    ulx = int((resize_res - inp_res) / 2)
    uly = int((resize_res - inp_res) / 2)
    # Crop 256x256
    rgb = rgb[:, :, uly : uly + inp_res, ulx : ulx + inp_res]
    rgb = to_torch(rgb).float()
    assert rgb.max() <= 1
    rgb = color_normalize(rgb, mean, std)
    return rgb

def sliding_windows(
        rgb: torch.Tensor,
        num_in_frames: int,
        stride: int,
) -> tuple:
    """
    Return sliding windows and corresponding (middle) timestamp
    """
    C, nFrames, H, W = rgb.shape
    # If needed, pad to the minimum clip length
    if nFrames < num_in_frames:
        rgb_ = torch.zeros(C, num_in_frames, H, W)
        rgb_[:, :nFrames] = rgb
        rgb_[:, nFrames:] = rgb[:, -1].unsqueeze(1)
        rgb = rgb_
        nFrames = rgb.shape[1]

    num_clips = math.ceil((nFrames - num_in_frames) / stride) + 1
    plural = ""
    if num_clips > 1:
        plural = "s"
    print(f"{num_clips} clip{plural} resulted from sliding window processing.")

    rgb_slided = torch.zeros(num_clips, 3, num_in_frames, H, W)
    t_mid = []
    # For each clip
    for j in range(num_clips):
        # Check if num_clips becomes 0
        actual_clip_length = min(num_in_frames, nFrames - j * stride)
        if actual_clip_length == num_in_frames:
            t_beg = j * stride
        else:
            t_beg = nFrames - num_in_frames
        t_mid.append(t_beg + num_in_frames / 2)
        rgb_slided[j] = rgb[:, t_beg : t_beg + num_in_frames, :, :]
    return rgb_slided, np.array(t_mid)

class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,ms.shaw
        # out_t = np.ceil(float(t) / float(self.stride[0]))
        # out_h = np.ceil(float(h) / float(self.stride[1]))
        # out_w = np.ceil(float(w) / float(self.stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)

class Unit3D(nn.Module):
    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        padding=0,
        activation_fn=F.relu,
        use_batch_norm=True,
        use_bias=False,
        name="unit_3d",
        num_domains=1,
    ):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._num_domains = num_domains
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,  # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
            bias=self._use_bias,
        )

        if self._use_batch_norm:
            if self._num_domains == 1:
                self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)
            else:
                self.bn = DomainSpecificBatchNorm3d(
                    self._output_channels, self._num_domains, eps=0.001, momentum=0.01
                )

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        # out_t = np.ceil(float(t) / float(self._stride[0]))
        # out_h = np.ceil(float(h) / float(self._stride[1]))
        # out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name, num_domains=1):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_0/Conv3d_0a_1x1",
        )
        self.b1a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_1/Conv3d_0a_1x1",
        )
        self.b1b = Unit3D(
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_1/Conv3d_0b_3x3",
        )
        self.b2a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_2/Conv3d_0a_1x1",
        )
        self.b2b = Unit3D(
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_2/Conv3d_0b_3x3",
        )
        self.b3a = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0
        )
        self.b3b = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_3/Conv3d_0b_1x1",
        )
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)

class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
        "Logits",
        "Predictions",
    )

    def __init__(
        self,
        num_classes=400,
        spatiotemporal_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
        num_in_frames=64,
        include_embds=False,
    ):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatiotemporal_squeeze: Whether to squeeze the 2 spatial and 1 temporal dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          in_channels: Number of input channels (default 3 for RGB).
          dropout_keep_prob: Dropout probability (default 0.5).
          name: A string (optional). The name of this module.
          num_in_frames: Number of input frames (default 64).
          include_embds: Whether to return embeddings (default False).
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError("Unknown final endpoint %s" % final_endpoint)

        super().__init__()
        self._num_classes = num_classes
        self._spatiotemporal_squeeze = spatiotemporal_squeeze
        self._final_endpoint = final_endpoint
        self.include_embds = include_embds
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError("Unknown final endpoint %s" % self._final_endpoint)

        self.end_points = {}
        end_point = "Conv3d_1a_7x7"
        self.end_points[end_point] = Unit3D(
            in_channels=in_channels,
            output_channels=64,
            kernel_shape=[7, 7, 7],
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_2a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Conv3d_2b_1x1"
        self.end_points[end_point] = Unit3D(
            in_channels=64,
            output_channels=64,
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Conv3d_2c_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=64,
            output_channels=192,
            kernel_shape=[3, 3, 3],
            padding=1,
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_3a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_3b"
        self.end_points[end_point] = InceptionModule(
            192, [64, 96, 128, 16, 32, 32], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_3c"
        self.end_points[end_point] = InceptionModule(
            256, [128, 128, 192, 32, 96, 64], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_4a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4b"
        self.end_points[end_point] = InceptionModule(
            128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4c"
        self.end_points[end_point] = InceptionModule(
            192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4d"
        self.end_points[end_point] = InceptionModule(
            160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4e"
        self.end_points[end_point] = InceptionModule(
            128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4f"
        self.end_points[end_point] = InceptionModule(
            112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_5a_2x2"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_5b"
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_5c"
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Logits"

        last_duration = int(math.ceil(num_in_frames / 8))  # 8
        last_size = 7  # int(math.ceil(sample_width / 32))  # this is for 224
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)

        self.dropout = nn.Dropout(dropout_keep_prob)

        self.logits = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=self._num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=self._num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        # [batch x featuredim x 1 x 1 x 1]
        embds = self.dropout(self.avgpool(x))

        # [batch x classes x 1 x 1 x 1]
        x = self.logits(embds)
        if self._spatiotemporal_squeeze:
            # [batch x classes]
            logits = x.squeeze(3).squeeze(3).squeeze(2)

        # logits [batch X classes]
        if self.include_embds:
            return {"logits": logits, "embds": embds}
        else:
            return {"logits": logits}

def load_i3d_model(
        i3d_checkpoint_path: Path,
        num_classes: int,
        num_in_frames: int,
) -> torch.nn.Module:
    """Load pre-trained I3D checkpoint, put in eval mode."""
    model = InceptionI3d(
        num_classes=num_classes,
        spatiotemporal_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
        num_in_frames=num_in_frames,
        include_embds=True,
    )
    model = torch.nn.DataParallel(model).to(device)
    checkpoint = torch.load(i3d_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

i3dmodel = load_i3d_model(
            i3d_checkpoint_path= Path("i3d_kinetics_bsl1k_bslcp.pth.tar"),
            num_classes=981,
            num_in_frames= 16,
        )


def main_i3d(
    video_path: Path,
    save_path: Path,
    filename: str = "i3d_dict.mat",
    fps: int = 25,
    num_classes: int = 981,
    num_in_frames: int = 16,
    batch_size: int = 1,
    stride: int = 1,
    slowdown_factor: int = 1,
    save_features: bool = False,
    viz: bool = False,
    generate_vtt: bool = False,
):
    
    with BlockTimer("Loading video frames"):
        rgb = load_rgb_video(
            video_path=video_path,
            fps=fps,
        ).to(device)
    # Prepare: resize/crop/normalize
    rgb = prepare_input(rgb).to(device)
    # Sliding window
    rgb, t_mid = sliding_windows(
        rgb=rgb,
        stride=stride,
        num_in_frames=num_in_frames,
    )
    # Number of windows/clips
    num_clips = rgb.shape[0]
    # Group the clips into batches
    num_batches = math.ceil(num_clips / batch_size)
    all_features = torch.Tensor(num_clips, 1024)
    all_logits = torch.Tensor(num_clips, num_classes)
    for b in range(num_batches):
        inp = rgb[b * batch_size : (b + 1) * batch_size]
        # Forward pass
        out = i3dmodel(inp)
        logits = out["logits"].data.cpu()
        all_features[b] = out["embds"].squeeze().data.cpu()
        all_logits[b] = logits.squeeze().data.cpu()

    if save_features:
        save_pred(
            all_features, all_logits, checkpoint=save_path, filename=filename,
        )
    
    return all_features.detach(), all_logits.detach()

@beartype
def give_change_points_labels_for_single_video_given_i3d_features(features:torch.Tensor, pen:int) -> list:
    
    '''
    returns the change point labels in the form of a tensor
    pen: penalty used to control complexity of the segmentation model
    '''
    
    CP_dict = {}
    model ='l2'
    jump = 2
    
    if len(features) < 2:
        CP_dict['video'] = np.zeros(len(features))
        
    algo = rpt.Pelt(model=model, jump=jump).fit(features)
    res = algo.predict(pen=pen)
    res_np = [1 if ix in res else 0 for ix in range(len(features))]
    return res_np

if __name__ == "__main__":

    videos_list = os.listdir('ISLRTC_videos')
    videos_list = [Path(f'ISLRTC_videos/{vid}') for vid in videos_list]

    current_done_list = os.listdir('islrtc_i3d_1')
    current_done_list = [str(x).split('_')[0] for x in current_done_list]

    for vid_path in tqdm(videos_list):

        print('*' * 100)
        print('\n')
        print(f"video = {vid_path}")

        my_vid_name = str(vid_path).split('/')[1].split('.')[0]

        if my_vid_name in current_done_list:
            continue

        try:
            save_filename = str(vid_path).split('/')[-1].split('.')[0] + '_i3d.mat'
            features, logits = main_i3d(vid_path, save_path=Path('islrtc_i3d_1'), filename=save_filename, save_features=True)

        except:
            with open('islrtc_i3d_1/failed_videos.txt', 'a') as f: 
                f.write(f"video = {my_vid_name}")    

    