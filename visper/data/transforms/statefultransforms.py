import random
import math
import numbers

import gin
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms.functional as functional


@gin.configurable
class StatefulRandomCrop(object):
    def __init__(self, insize, outsize):
        self.size = outsize
        self.cropParams = self.get_params(insize, self.size)

    @staticmethod
    def get_params(insize, outsize):
        """Get parameters for ``crop`` for a random crop.
        Args:
            insize (PIL Image): Image to be cropped.
            outsize (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = insize
        th, tw = outsize
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """

        i, j, h, w = self.cropParams

        return functional.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

@gin.configurable
class StatefulRandomHorizontalFlip(object):
    def __init__(self, probability=0.5):
        self.p = probability
        self.rand = random.random()

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if self.rand < self.p:
            return functional.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

@gin.configurable
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, inp):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(inp, weight=self.weight, groups=self.groups)


if __name__ == "__main__":
    from .preprocess import load_video
    import cv2
    video = load_video("/home/dmitry.klimenkov/Documents/projects/3DResNet/demo_bot/bot_videos/2019:06:12:13:14:14_Dmitry Klimenkov/model_input.mp4")

    smoothing = GaussianSmoothing(channels=3, kernel_size=4, sigma=1.5, dim=2)
    inp = video[0].unsqueeze_(0) #torch.rand(1, 1, 100, 100)
    print(inp.size())
    inp = F.pad(inp, (2, 2, 2, 2), mode='reflect')
    output = smoothing(inp)

    inp_np = inp.numpy()[0]
    inp_np = np.transpose(inp_np, [1,2,0])

    out_np = output.numpy()[0]
    out_np = np.transpose(out_np, [1,2,0])

    print(inp_np.shape)
    print(out_np.shape)

    cv2.imshow("inp", inp_np)
    cv2.imshow("out", out_np)

    cv2.waitKey(0)
    cv2.destroyAllWindows()