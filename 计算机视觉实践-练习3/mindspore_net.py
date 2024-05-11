import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import Normal
from mindspore.ops import operations as P


class Shift8(nn.Cell):
    def __init__(self, groups=4, stride=1, mode="constant"):
        super().__init__()
        self.g = groups
        self.mode = mode
        self.stride = stride
        self.zeros_like = ops.ZerosLike()
        self.pad = ops.Pad(((0, 0), (0, 0), (stride, stride), (stride, stride)))

    def construct(self, x):
        b, c, h, w = x.shape
        out = self.zeros_like(x)

        pad_x = self.pad(x)
        assert c == self.g * 8

        cx, cy = self.stride, self.stride
        stride = self.stride
        # MindSpore does not support item assignment, so we use concatenation
        out = ops.concat(
            (
                pad_x[
                    :,
                    0 * self.g : 1 * self.g,
                    cx - stride : cx - stride + h,
                    cy : cy + w,
                ],
                pad_x[
                    :,
                    1 * self.g : 2 * self.g,
                    cx + stride : cx + stride + h,
                    cy : cy + w,
                ],
                pad_x[
                    :,
                    2 * self.g : 3 * self.g,
                    cx : cx + h,
                    cy - stride : cy - stride + w,
                ],
                pad_x[
                    :,
                    3 * self.g : 4 * self.g,
                    cx : cx + h,
                    cy + stride : cy + stride + w,
                ],
                pad_x[
                    :,
                    4 * self.g : 5 * self.g,
                    cx + stride : cx + stride + h,
                    cy + stride : cy + stride + w,
                ],
                pad_x[
                    :,
                    5 * self.g : 6 * self.g,
                    cx + stride : cx + stride + h,
                    cy - stride : cy - stride + w,
                ],
                pad_x[
                    :,
                    6 * self.g : 7 * self.g,
                    cx - stride : cx - stride + h,
                    cy + stride : cy + stride + w,
                ],
                pad_x[
                    :,
                    7 * self.g : 8 * self.g,
                    cx - stride : cx - stride + h,
                    cy - stride : cy - stride + w,
                ],
            ),
            1,
        )
        return out


class ResidualBlockShift(nn.Cell):
    """
    Residual block without BN in MindSpore.

    It has a style of:
        ---Conv-Shift-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features. Default: 64.
        res_scale (float): Residual scale. Default: 1.
        mindspore_init (bool): If set to True, use MindSpore default init,
            otherwise, use custom init. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, mindspore_init=False):
        super(ResidualBlockShift, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, kernel_size=1, has_bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, kernel_size=1, has_bias=True)
        self.relu = nn.ReLU()
        self.shift = Shift8(groups=num_feat // 8, stride=1)

        if not mindspore_init:
            # Use custom initializer
            self.conv1.weight.set_data(Normal(0.1)(self.conv1.weight.shape))
            self.conv2.weight.set_data(Normal(0.1)(self.conv2.weight.shape))

    def construct(self, x):
        identity = x
        out = self.conv2(self.relu(self.shift(self.conv1(x))))
        return identity + out * self.res_scale


class UpShiftPixelShuffle(nn.Cell):
    """
    UpShiftPixelShuffle with DepthToSpace in MindSpore.

    Args:
        dim (int): Number of input channels.
        scale (int): Upscale factor. Default: 2.
    """

    def __init__(self, dim, scale=2):
        super(UpShiftPixelShuffle, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, has_bias=True)
        self.leaky_relu = nn.LeakyReLU(0.02)
        self.shift = Shift8(groups=dim // 8)
        self.conv2 = nn.Conv2d(dim, dim * scale * scale, kernel_size=1, has_bias=True)
        self.depth_to_space = P.DepthToSpace(block_size=scale)

    def construct(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.shift(x)
        x = self.conv2(x)
        x = self.depth_to_space(x)
        return x


class UpShiftMLP(nn.Cell):
    """
    UpShiftMLP in MindSpore.

    Args:
        dim (int): Number of input channels.
        mode (str): Upsampling mode, either 'bilinear' or 'nearest'. Default: 'bilinear'.
        scale (int): Upscale factor. Default: 2.
    """

    def __init__(self, dim, mode="bilinear", scale=2):
        super(UpShiftMLP, self).__init__()
        self.scale = scale
        self.mode = mode
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, has_bias=True)
        self.leaky_relu = nn.LeakyReLU(0.02)
        self.shift = Shift8(groups=dim // 8)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, has_bias=True)
        if mode == "bilinear":
            self.upsample = P.ResizeBilinear(scale_factor=scale, align_corners=False)
        else:
            self.upsample = P.ResizeNearestNeighbor(scale_factor=scale)

        self.up_layer = nn.CellList(
            [self.upsample, self.conv1, self.leaky_relu, self.shift, self.conv2]
        )

    def construct(self, x):
        for layer in self.up_layer:
            x = layer(x)
        return x


# 定义 SCNet 的 MindSpore 版本
class SCNet(nn.Cell):
    """
    SCNet based on the Modified SRResNet for MindSpore.
    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(SCNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 1, has_bias=True)
        self.body = nn.CellList(
            [ResidualBlockShift(num_feat) for _ in range(num_block)]
        )

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = UpShiftMLP(num_feat, scale=self.upscale)

        elif self.upscale == 4:
            self.upconv1 = UpShiftMLP(num_feat)
            self.upconv2 = UpShiftMLP(num_feat)
        elif self.upscale == 8:
            self.upconv1 = UpShiftMLP(num_feat)
            self.upconv2 = UpShiftMLP(num_feat)
            self.upconv3 = UpShiftMLP(num_feat)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, kernel_size=1, has_bias=True)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, kernel_size=1, has_bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(0.1)

        # initialization
        for layer in [self.conv_first, self.conv_hr, self.conv_last]:
            layer.weight.set_data(Normal(0.1)(layer.weight.shape))

    def construct(self, x):
        feat = self.lrelu(self.conv_first(x))
        for block in self.body:
            feat = block(feat)

        if self.upscale == 4:
            feat = self.lrelu(self.upconv1(feat))
            feat = self.lrelu(self.upconv2(feat))
        elif self.upscale in [2, 3]:
            feat = self.lrelu(self.upconv1(feat))
        elif self.upscale == 8:
            feat = self.lrelu(self.upconv1(feat))
            feat = self.lrelu(self.upconv2(feat))
            feat = self.lrelu(self.upconv3(feat))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        base = nn.ResizeBilinear()(x, scale_factor=self.upscale)
        out += base
        return out


import os
from mindspore import load_checkpoint, load_param_into_net
from PIL import Image

if __name__ == "__main__":
    # 初始化模型
    model = SCNet(upscale=4)
    # 加载预训练的模型权重
    param_dict = load_checkpoint("SCNet-T-D64B16.ckpt")
    load_param_into_net(model, param_dict)
    model.set_train(False)

    # 设置输入输出文件夹路径
    input_folder = "Set5/LR"
    output_folder = "Set5/SR2"

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有PNG图片
    for image_file in os.listdir(input_folder):
        if image_file.endswith(".png"):
            # 图像预处理
            image_path = os.path.join(input_folder, image_file)
            image = Image.open(image_path)
            # MindSpore没有transforms.Compose，需要手动进行转换
            image = image.resize((image.size[1] * 4, image.size[0] * 4))
            image = mindspore.Tensor(np.array(image).astype(np.float32) / 255.0)

            # 进行超分辨率处理
            output = model.predict(image)

            # 将输出转换为图像
            output_image = Image.fromarray((output.asnumpy() * 255).astype(np.uint8))
            output_image_path = os.path.join(output_folder, image_file)
            output_image.save(output_image_path)
