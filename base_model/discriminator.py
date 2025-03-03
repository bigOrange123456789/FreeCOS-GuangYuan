import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.structure_extract import eightway_affinity_kld

class FCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(
            ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(
            ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x


class EightwayASADiscriminator(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(EightwayASADiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(
            ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        # self.pam = PAM_Module(512)
        self.classifier = nn.Conv2d(
            ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

class PredictDiscriminator(nn.Module): # 由5个卷基层+4个relu构成
    def __init__(self, num_classes, ndf=64): # num_classes=1
        super(PredictDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            num_classes, ndf,
            kernel_size=4, stride=2, padding=1)
        # Conv2d(in_channels, out_channels,
        # kernel_size, stride=1, padding=0)[
        self.conv2 = nn.Conv2d(
            ndf, ndf*2,
            kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            ndf*2, ndf*4,
            kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(
            ndf*4, ndf*8,
            kernel_size=4, stride=2, padding=1)
        # self.pam = PAM_Module(512)
        self.classifier = nn.Conv2d(
            ndf*8, 1,
            kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # 如果x>0,f(x)=x 如果x<0,f(x)=0.2*x

    def forward(self, x): # [n,1,256**2]
        x = self.conv1(x) # [n,64,128**2]<-
        x = self.leaky_relu(x)
        x = self.conv2(x) # [n,2*64,64**2]<-
        x = self.leaky_relu(x)
        x = self.conv3(x) # [n,4*64,32**2]<-
        x = self.leaky_relu(x)
        x = self.conv4(x) # [n,8*64,16**2]<-
        x = self.leaky_relu(x)
        x = self.classifier(x) # [n,1,8**2]
        return x

class PredictDiscriminator_affinity(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(PredictDiscriminator_affinity, self).__init__()
        self.conv1 = nn.Conv2d(
            8*num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(
            ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        # self.pam = PAM_Module(512)
        self.classifier = nn.Conv2d(
            ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = eightway_affinity_kld(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

class Latent_Discriminator(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(Latent_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(
            ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        # self.pam = PAM_Module(512)
        self.classifier = nn.Conv2d(
            ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x


class FourwayASADiscriminator(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(FourwayASADiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            4*num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(
            ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(
            ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    net = EightwayASADiscriminator(num_classes=19)
    input = torch.randn(1, 19, 512, 512)
    output = net(input)
    print("outshape",output.shape)
