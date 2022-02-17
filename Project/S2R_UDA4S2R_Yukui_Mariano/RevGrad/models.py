import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torchvision.models as models
import math

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class RevGrad(nn.Module):

    def __init__(self, num_classes = 10):
        super(RevGrad, self).__init__()
        self.nclasses = num_classes

        self.feature = nn.Sequential()

# RevGrad:
       self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
       self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
       self.feature.add_module('f_pool1', nn.MaxPool2d(2))
       self.feature.add_module('f_relu1', nn.ReLU(True))
       self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
       self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
       self.feature.add_module('f_drop1', nn.Dropout2d())
       self.feature.add_module('f_pool2', nn.MaxPool2d(2))
       self.feature.add_module('f_relu2', nn.ReLU(True))

# Depthwise Separable RevGrad:
        # self.feature.add_module('f_conv1_depth', nn.Conv2d(in_channels=3, out_channels=3, groups=3, kernel_size=5))
        # self.feature.add_module('f_bn1_depth', nn.BatchNorm2d(3))
        # self.feature.add_module('f_relu1_depth', nn.ReLU(True))
        # self.feature.add_module('f_conv1_point', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1))
        # self.feature.add_module('f_bn1_point', nn.BatchNorm2d(64))
        # self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        # self.feature.add_module('f_relu1_point', nn.ReLU(True))
        # self.feature.add_module('f_conv2_depth', nn.Conv2d(in_channels=64, out_channels=64, groups=64, kernel_size=5))
        # self.feature.add_module('f_bn2_depth', nn.BatchNorm2d(64))
        # self.feature.add_module('f_relu2_depth', nn.ReLU(True))
        # self.feature.add_module('f_conv2_point', nn.Conv2d(in_channels=64, out_channels=50, kernel_size=1))
        # self.feature.add_module('f_bn2_point', nn.BatchNorm2d(50))
        # self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        # self.feature.add_module('f_relu2_point', nn.ReLU(True))

# Deeper Depthwise Separable RevGrad:
        # self.feature.add_module('f_conv1_depth', nn.Conv2d(in_channels=3, out_channels=3, groups=3, kernel_size=4, stride=1, padding=0))
        # self.feature.add_module('f_bn1_depth', nn.BatchNorm2d(3))
        # self.feature.add_module('f_relu1_depth', nn.ReLU(True))
        # self.feature.add_module('f_conv1_point', nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1))
        # self.feature.add_module('f_bn1_point', nn.BatchNorm2d(16))
        # self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        # self.feature.add_module('f_relu1_point', nn.ReLU(True))
        # self.feature.add_module('f_conv2_depth', nn.Conv2d(in_channels=16, out_channels=16, groups=16, kernel_size=3, stride=1, padding=1))
        # self.feature.add_module('f_bn2_depth', nn.BatchNorm2d(16))
        # self.feature.add_module('f_relu2_depth', nn.ReLU(True))
        # self.feature.add_module('f_conv2_point', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1))
        # self.feature.add_module('f_bn2_point', nn.BatchNorm2d(32))
        # self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        # self.feature.add_module('f_relu2_point', nn.ReLU(True))
        # self.feature.add_module('f_conv3_depth', nn.Conv2d(in_channels=32, out_channels=32, groups=32, kernel_size=3, stride=1, padding=1))
        # self.feature.add_module('f_bn3_depth', nn.BatchNorm2d(32))
        # self.feature.add_module('f_relu3_depth', nn.ReLU(True))
        # self.feature.add_module('f_conv3_point', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1))
        # self.feature.add_module('f_bn3_point', nn.BatchNorm2d(64))
        # self.feature.add_module('f_pool3', nn.MaxPool2d(2))
        # self.feature.add_module('f_relu3_point', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
# RevGrad and Depthwise Separable RevGrad:
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, self.nclasses))
# Deeper Depthwise Separable RevGrad:
        # self.class_classifier.add_module('c_fc1', nn.Linear(64 * 3 * 3, 500))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(500))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_fc2', nn.Linear(500, 400))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(400))
        # self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        # self.class_classifier.add_module('c_fc3', nn.Linear(400, 100))
        # self.class_classifier.add_module('c_bn3', nn.BatchNorm1d(100))
        # self.class_classifier.add_module('c_relu3', nn.ReLU(True))
        # self.class_classifier.add_module('c_fc4', nn.Linear(100, self.nclasses))


        self.domain_classifier = nn.Sequential()
# RevGrad and Depthwise Separable RevGrad:
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
# Deeper Depthwise Separable RevGrad:
        # self.domain_classifier.add_module('d_fc1', nn.Linear(64 * 3 * 3, 100))

        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))


    def forward(self, input_data, alpha):
        input_data = input_data.expand(len(input_data), 3, 28, 28)
        feature = self.feature(input_data)
# RevGrad and Depthwise Separable RevGrad:
        feature = feature.view(-1, 50 * 4 * 4)
# Deeper Depthwise Separable RevGrad:
        # feature = feature.view(-1, 64 * 3 * 3)

        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return feature, class_output, domain_output
