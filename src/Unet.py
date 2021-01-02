import torch 
import torch.nn as nn
from torch.nn import functional as F

def doubleConv(inChannels, outChannels):
    """
    Applies two convolutional layers each followed by ReLU activation
    :param inChannels: # of input channels
    :param outChannels: # of output channels
    :return: a down-conv layer 
    """
    conv = nn.Sequential(
        nn.Conv2d(inChannels, outChannels, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(outChannels, outChannels, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv

def cropTensor(tensor, targetTensor):
    """
    Center crops tensor to size of given target tensor
    Both tensors have shape (bs, c, h, w)
    :param tensor: tensor needing to be cropped
    :param tensorTarget: target tensor of smaller size
    :return: cropped tensor 
    """
    targetSize = targetTensor.size()[2]
    tensorSize = tensor.size()[2]
    delta = tensorSize - targetSize
    delta = delta // 2
    return tensor[:, :, delta:tensorSize - delta, delta:tensorSize - delta]

 
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.maxPool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.downConv1 = doubleConv(1, 64)
        self.downConv2 = doubleConv(64, 128)
        self.downConv3 = doubleConv(128, 256)
        self.downConv4 = doubleConv(256, 512)
        self.downConv5 = doubleConv(512, 1024)

        self.upTrans1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2
        )
        self.upConv1 = doubleConv(1024, 512)
        
        self.upTrans2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2
        )
        self.upConv2 = doubleConv(512, 256)
        
        self.upTrans3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )
        self.upConv3 = doubleConv(256, 128)
        
        self.upTrans4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )
        self.upConv4 = doubleConv(128, 64)
        
        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=1
        )
        
    def forward(self, image):
        # encoder
        x1 = self.downConv1(image)
        x2 = self.maxPool2x2(x1)
        x3 = self.downConv2(x2)
        x4 = self.maxPool2x2(x3)
        x5 = self.downConv3(x4)
        x6 = self.maxPool2x2(x5)
        x7 = self.downConv4(x6)
        x8 = self.maxPool2x2(x7)
        x9 = self.downConv5(x8)
        
        # decoder
        x = self.upTrans1(x9)
        y = cropTensor(x7, x)
        x = self.upConv1(torch.cat([x, y], axis=1))
        x = self.upTrans2(x)
        y = cropTensor(x5, x)
        x = self.upConv2(torch.cat([x, y], axis=1))
        x = self.upTrans3(x)
        y = cropTensor(x3, x)
        x = self.upConv3(torch.cat([x, y], axis=1))
        x = self.upTrans4(x)
        y = cropTensor(x1, x)
        x = self.upConv4(torch.cat([x, y], axis=1))
        
        # output layer
        out = self.out(x)
        
        return out

if __name__ == "__main__":
    image = torch.rand((1,1,572, 572))
    model = UNet()
    print(model(image))
        
    
        
        