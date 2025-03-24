from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None, max_size=None, shape=None):
    """加载图像并将其转换为tensor张量。"""
    image = Image.open(image_path)
    
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.LANCZOS)
    
    if shape:
        image = image.resize(shape, Image.LANCZOS)
    
    if transform:
        image = transform(image).unsqueeze(0)
    
    return image.to(device)  # 将图像移到 GPU 或 CPU

class VGGNet(nn.Module):
    def __init__(self):
        """选择 conv1_1 ~ conv5_1 激活图。"""
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28'] 
        self.vgg = models.vgg19(pretrained=False)
        self.vgg.load_state_dict(torch.load('C:\\style-transfer\\vgg19-dcbb9e9d.pth'))
        self.vgg = self.vgg.features
        
    def forward(self, x):
        """提取多个（5）个卷积特征向量。"""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

def main(config):
    # 图像预处理
    style_losses = [] 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.472, 0.436, 0.424), 
                             std=(0.227, 0.226, 0.225))])
    
    # 加载内容和样式图片
    content = load_image(config.content, transform, max_size=config.max_size)
    style = load_image(config.style, transform, shape=[content.size(2), content.size(3)])
    
    # 使用内容图像初始化目标图像
    target = torch.randn_like(content, device=device).requires_grad_(True)

    
    optimizer = torch.optim.Adam([target], lr=config.lr, betas=[0.5, 0.999])
    vgg = VGGNet().to(device).eval()  # 将VGGNet模型移到GPU或CPU
    
    for step in range(config.total_step):
        # 提取多个（5）个卷积特征向量
        target_features = vgg(target)
        content_features = vgg(content)
        style_features = vgg(style)
        style_loss = 0
        content_loss = 0
        content_loss = torch.mean((target_features[3] - content_features[3]) ** 2)
        for f1, f3 in zip(target_features,  style_features):
            # 使用目标图像和内容图像计算内容损失
 
            # 重塑卷积特征图
            _, c, h, w = f1.size()
            f1 = f1.view(c, h * w)
            f3 = f3.view(c, h * w)
 
            # 计算 gram 矩阵
            f1 = torch.mm(f1, f1.t())
            f3 = torch.mm(f3, f3.t())
 
            # 使用目标和风格图像计算风格损失
            style_loss += torch.mean((f1 - f3)**2) / (c * h * w) 
        
        # 计算总损失、反向传播和优化
        loss = content_loss + config.style_weight * style_loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 200 == 0:
            style_losses.append(style_loss.item())
        if (step+1) % config.log_step == 0:
            print ('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}' 
                   .format(step+1, config.total_step, content_loss.item(), style_loss.item()))
 
        if (step+1) % config.sample_step == 0:
            # 保存生成的图片
            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = target.clone().squeeze()
            img = denorm(img).clamp_(0, 1)
            torchvision.utils.save_image(img, 'output-{}.png'.format(step+1))
    plt.plot(range(200, config.total_step + 1, 200), style_losses, label='Style Loss')
    plt.xlabel('Steps')
    plt.ylabel('Style Loss')
    plt.title('Style Loss Decrease Over Time')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='C:\style-transfer\98ebaba5-8468-4122-bfef-561e47fe3441.jpg')
    parser.add_argument('--style', type=str, default='C:\\style-transfer\\udnie.jpg')
    parser.add_argument('--max_size', type=int, default=400)
    parser.add_argument('--total_step', type=int, default=200)
    parser.add_argument('--log_step', type=int, default=80)
    parser.add_argument('--sample_step', type=int, default=200)
    parser.add_argument('--style_weight', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.01)
    config = parser.parse_args()
    print(config)
    main(config)
