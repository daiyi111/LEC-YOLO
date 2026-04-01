import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MDCModule', 'MSFR']
class MDCModule(nn.Module):
    """多扩张卷积融合模块(MDC) - 使用不同扩张率的卷积核提取多尺度特征"""

    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 3]):
        """
        初始化MDC模块

        参数:
            in_channels: 输入特征图通道数
            out_channels: 输出特征图通道数
            dilation_rates: 扩张卷积的扩张率列表
        """
        super(MDCModule, self).__init__()

        # 定义不同扩张率的卷积分支
        self.branches = nn.ModuleList()
        for dilation in dilation_rates:
            # 每个分支包含卷积、批量归一化和激活函数
            branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          dilation=dilation, padding=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU()  # 使用SiLU激活函数，符合文档中的实现
            )
            self.branches.append(branch)

        # 原始特征映射的卷积层
        self.original_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

        # 融合层：将所有分支特征与原始特征合并
        self.fusion_conv = nn.Sequential(
            nn.Conv2d((len(dilation_rates) + 1) * out_channels, out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        """前向传播"""
        # 处理原始特征
        original_features = self.original_conv(x)

        # 处理不同扩张率的卷积分支
        branch_features = [branch(x) for branch in self.branches]

        # 合并所有特征
        all_features = [original_features] + branch_features
        fused_features = torch.cat(all_features, dim=1)

        # 融合特征
        output = self.fusion_conv(fused_features)
        return output




class MSFR(nn.Module):
    """多尺度特征重用模块(MSFR) - 整合MDC和残差块，实现级联特征重用"""

    def __init__(self, c1, c2, e=0.25):
        """
        初始化MSFR模块

        参数:
            in_channels: 输入特征图通道数
            mid_channels: 中间层特征图通道数
            out_channels: 输出特征图通道数
        """
        super(MSFR, self).__init__()
        self.c=int(c1*e)
        self.cbs1=nn.Sequential(nn.Conv2d(c1, self.c, kernel_size=1),
                                 nn.BatchNorm2d(self.c),
                                 nn.SiLU())
        # 第一层MDC和残差块 (L1)
        self.l1_mdc = MDCModule(self.c, self.c)
        # 第二层MDC和残差块 (L2) - 输入为L1输出和原始特征
        self.l2_mdc = MDCModule(self.c , self.c)
        # 第三层MDC和残差块 (L3) - 输入为L1、L2输出和原始特征
        self.l3_mdc = MDCModule(self.c , self.c)
        self.cbs2=nn.Sequential(nn.Conv2d(self.c, c2, kernel_size=1),
                                 nn.BatchNorm2d(c2),
                                 nn.SiLU())
        # 最终融合层

    def forward(self, x):
        """前向传播 - 实现级联特征重用"""
        # 保存原始输入用于跨层连接
        x0=self.cbs1(x)
        # 第一层处理 (L1)
        x1 = self.l1_mdc(x0)+x0


        # 第二层处理 (L2) - 输入为L1输出和原始特征
        # l2_input = torch.cat([l1_output, original], dim=1)
        x2 = self.l2_mdc(x1)+x1+x0

        # 第三层处理 (L3) - 输入为L1、L2输出和原始特征
        # l3_input = torch.cat([l1_output, l2_output, original], dim=1)
        x3 = self.l3_mdc(x2)+x2+x1+x0
        output=self.cbs2(x3)
        # 融合所有层的输出
        # final_input = torch.cat([l1_output, l2_output, l3_output], dim=1)
        return output

    """
    # 创建一个示例输入
    input_tensor = torch.randn(1,64,40,40)
    model = MDCModule(in_channels=64, out_channels=32)
    output = model(input_tensor)
    print(output.shape)
    """
if __name__ == '__main__':
    # 创建一个示例输入
    input_tensor = torch.randn(1,64,40,40)
    model=MSFR(64,32)
    output = model(input_tensor)
    print(output.shape)