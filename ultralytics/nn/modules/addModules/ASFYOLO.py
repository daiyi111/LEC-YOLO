import torch.nn as nn
import torch.nn.functional as F
import torch
import math

__all__=['TFE', 'CPAM', 'SSFF']

class TFE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.downchannel=nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=2*c, out_channels=c, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )
        self.upchannel=nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=c//2, out_channels=c, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )
        # 可以调整成2*c
        self.reshape_channel = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=3*c, out_channels=c, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )

        self.atten_weight = nn.Conv2d(kernel_size=3,in_channels=3*c,out_channels=3,stride=1, padding=1, bias=False,groups=3)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        s, m, l = x[0], x[1], x[2]
        b,c,h,w =m.shape[0], m.shape[1], m.shape[2],m.shape[3]
        tgt_size = m.shape[2:]
        l = self.downchannel(l)
        l = F.interpolate(l, tgt_size, mode='nearest')
        # l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        #l = self.conv_l_post_down(l)
        # m = self.conv_m(m)
        # s = self.conv_s_pre_up(s)
        s=self.upchannel(s)
        # s = F.adaptive_max_pool2d(s, tgt_size)+F.adaptive_avg_pool2d(l, tgt_size)
        s=F.max_pool2d(s,kernel_size=3,padding=1,stride=2)+F.avg_pool2d(s,kernel_size=3,padding=1,stride=2)
        # 改进1
        # sm =torch.add(s,m)
        # sml=torch.add(sm,l)
        #改进2
        sml = torch.cat([s, m, l], dim=1)
        sml = self.reshape_channel(sml)
        #改进3 分组卷积 生成自适应权重 乘以对应的维度的特征
        # concat_feat = torch.cat([s, m, l], dim=1)
        # weights = self.atten_weight(concat_feat)
        # weights= weights.view(b,3,h*w)
        # weights= self.softmax(weights)
        # weights= weights.view(b,3,h,w)
        # weight_s, weight_m, weight_l = weights[:, 0:1, :, :], weights[:, 1:2, :, :], weights[:, 2:3, :, :]
        # s_att = s * weight_s
        # m_att = m * weight_m
        # l_att = l * weight_l
        # fused = torch.cat([s_att, m_att, l_att], dim=1)
        # res = self.reshape_channel(fused)
        return sml

class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out

class CPAM(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, ch=256):
        super().__init__()
        self.channel_att = channel_att(ch)
        self.local_att = local_att(ch)

    def forward(self, x):
        input1, input2 = x[0], x[1]
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class SSFF(nn.Module):
    def __init__(self, channel):
        super(SSFF, self).__init__()
        self.conv1 =  Conv(channel*2, channel,1)
        self.conv2 =  Conv(channel*4, channel,1)
        self.conv3d = nn.Conv3d(channel,channel,kernel_size=(1,1,1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3,1,1))

    def forward(self, x):
        p3, p4, p5 = x[0],x[1],x[2]
        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d,p4_3d,p5_3d],dim = 2)  #b c d h w 4*256*3*80*80
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x



if __name__ == '__main__':
    model=TFE(512)
    input1= torch.randn(4,256,80,80)
    input2 = torch.randn(4,512,40,40)
    input3 = torch.randn(4,1024,20,20)
    list = [input1,input2,input3]
    output = model(list)
    print(output.shape)



