## backbone.py
用于搭建基础模型，通过generate_model函数来接收需要搭建的模型的深度并搭建模型。

### generate_model(model_depth: int, **kwargs) -> ResNet
``` python
def generate_model(model_depth: int, **kwargs) -> ResNet:
    configs = {
        10: (BasicBlock, [1, 1, 1, 1]),
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
        50: (Bottleneck, [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
        152: (Bottleneck, [3, 8, 36, 3]),
        200: (Bottleneck, [3, 24, 36, 3]),
    }
    if model_depth not in configs:
        raise ValueError(f"Unsupported ResNet depth: {model_depth}")
    block, layers = configs[model_depth]
    return ResNet(block, layers, get_inplanes(), **kwargs)
```
通过model_depth来选择cofigs中对应的配置，确定使用哪一种block, 以及每一个layer中使用几个block, get_inplanes()获得一个列表，列表中的值表示4个layer的输出channel维度

### get_inplanes() -> list[int]
``` python
def get_inplanes() -> list[int]:
    return [64, 128, 256, 512]
```

### conv3x3x3() 和 conv1x1x1()
```python
def conv3x3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
```
用于构建最基础的3D卷积块。

### class BasicBlock(nn.Module):
``` python
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)
```
基础模块，由generate_model()函数可知，深度为10，18，34的模型使用此基础模块搭建。
该模块主要包含2个conv3x3x3, 2个BatchNorm3d, 和一个downsample模块
其主要流程是：
```text
后续插入图片
```

```python 
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)
```
颈缩模块，度为50，101，152，200的模型使用此基础模块搭建。
该模块主要包括2个conv1x1x1，3个batchnorm3d, 和一个downsample模块
其主要流程是
```text
后续出入图片
```

### class ResNet(nn.Module):
完整构建ResNet的类
其主要输入有：
```text
    --block: 使用哪一种block来搭建模型
    --layers: list[int] 每一个stage的layer的数目（也就是block）的数目
    --block_inplanes: list[int] 每一个block的输出维度
    --n_input_channels: int=3, 最初是的输入维度，图片是3channel
    --conv1_t_size: int = 7, 第一个3d卷积层的时间维度上卷积核大小
    --conv1_t_stride: int = 1,第一个3d卷积层的时间维度上步长
    --widen_factor: float = 1.0, 将block的输出维度扩大倍数
    --shortcut_type: str = "B"， 每一个block中的downsample的模式
    --num_classes: int = 400, 最终输出维度
```

其主要方法有：
```text
    --def _make_layer(), 用于构建layer
    --def forward_stem()
    --def forward_features()
    --def forward_head()
    --def forward()， 以上三个用于将将前向传递过程拆分开，方便提取特征
    --def _downsample_basic_block(), 用于shortcut="A"
```

``` python
class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers: list[int],
        block_inplanes: list[int],
        n_input_channels: int = 3,
        conv1_t_size: int = 7,
        conv1_t_stride: int = 1,
        no_max_pool: bool = False,
        shortcut_type: str = "B",
        widen_factor: float = 1.0,
        num_classes: int = 400,
        l2_normalize_before_fc: bool = False,
    ):
        super().__init__()
        block_inplanes = [int(value * widen_factor) for value in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.l2_normalize_before_fc = bool(l2_normalize_before_fc)
        self.feature_dim = block_inplanes[3] * block.expansion

        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(self.feature_dim, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _downsample_basic_block(self, x: torch.Tensor, planes: int, stride: int) -> torch.Tensor:
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zeros = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4),
            device=out.device, dtype=out.dtype,
        )
        return torch.cat([out, zeros], dim=1)

    def _make_layer(self, block, planes: int, blocks: int, shortcut_type: str, stride: int = 1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )
        layers = [block(self.in_planes, planes, stride=stride, downsample=downsample)]
        self.in_planes = planes * block.expansion
        layers.extend(block(self.in_planes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward_stem(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        return x if self.no_max_pool else self.maxpool(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"ResNet3D expects [B,C,T,H,W], got {tuple(x.shape)}")
        x = self.forward_stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return torch.flatten(self.avgpool(x), 1)

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        if self.l2_normalize_before_fc:
            features = F.normalize(features, p=2, dim=1, eps=1e-12)
        return self.fc(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))


def generate_model(model_depth: int, **kwargs) -> ResNet:
    configs = {
        10: (BasicBlock, [1, 1, 1, 1]),
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
        50: (Bottleneck, [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
        152: (Bottleneck, [3, 8, 36, 3]),
        200: (Bottleneck, [3, 24, 36, 3]),
    }
    if model_depth not in configs:
        raise ValueError(f"Unsupported ResNet depth: {model_depth}")
    block, layers = configs[model_depth]
    return ResNet(block, layers, get_inplanes(), **kwargs)
```
我们以depth-18为例：

```python
block_inplanes = [int(value * widen_factor) for value in block_inplanes] # [64, 128, 256, 512] 因为widen_factor=1.0
self.in_planes = block_inplanes[0] # 64
self.no_max_pool = no_max_pool
self.l2_normalize_before_fc = bool(l2_normalize_before_fc)
self.feature_dim = block_inplanes[3] * block.expansion # 512, 因为depth-18使用basicblock其expansion=1

self.conv1 = nn.Conv3d(
    n_input_channels,
    self.in_planes,
    kernel_size=(conv1_t_size, 7, 7),
    stride=(conv1_t_stride, 2, 2),
    padding=(conv1_t_size // 2, 3, 3),
    bias=False,
)
self.bn1 = nn.BatchNorm3d(self.in_planes) 
self.relu = nn.ReLU(inplace=True)
self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1) #
```

```python
"""使用._make_layer()方法构建4个stage的块"""
self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)
```

```python
    def _downsample_basic_block(self, x: torch.Tensor, planes: int, stride: int) -> torch.Tensor:
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zeros = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4),
            device=out.device, dtype=out.dtype,
        )
        return torch.cat([out, zeros], dim=1)

    def _make_layer(self, block, planes: int, blocks: int, shortcut_type: str, stride: int = 1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )
        layers = [block(self.in_planes, planes, stride=stride, downsample=downsample)]
        self.in_planes = planes * block.expansion
        layers.extend(block(self.in_planes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)
```
我们以前两个layer为例：
对于self.layer1, 输入block=basicblock, block_inplanes[0]=64, layers[0]=2, shortcut_type="B"，stride=1
由于其不满足

```python
if stride != 1 or self.in_planes != planes * block.expansion:
```
所以直接执行

```python
layers = [block(self.in_planes, planes, stride=stride, downsample=downsample)]
self.in_planes = planes * block.expansion
layers.extend(block(self.in_planes, planes) for _ in range(1, blocks))
return nn.Sequential(*layers)
```
最终返回的是两个basic block顺序连接在一起，每一个block的输入通道维度是64，输出通道维度是64，strid是1，没有downsample.
对于self.layer2, 输入block=basicblock, block_inplanes[0]=128, layers[0]=2, shortcut_type="B"，stride=2
其满足

```python
if stride != 1 or self.in_planes != planes * block.expansion:
```
并且shortcut_type == "B",
所以其downsample是一个conv1x1x1+batchnorm3d的sequential module，其输入通道维度是64，输出通道维度是128
随后进入

```python
layers = [block(self.in_planes, planes, stride=stride, downsample=downsample)]
self.in_planes = planes * block.expansion
layers.extend(block(self.in_planes, planes) for _ in range(1, blocks))
return nn.Sequential(*layers)
```
同样是basic block顺序连接在一起，第一个block的输入通道维度是64，输出是128，第二个blcok的输入通道维度是128，输出通道维度是128，这里修改了self.inplane的值所以下一个self.layer3的第一个block输入维度会变成128.

如果使用的是shortcut_type == "A",
则会进入

```python
def _downsample_basic_block(self, x: torch.Tensor, planes: int, stride: int) -> torch.Tensor:
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zeros = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4),
            device=out.device, dtype=out.dtype,
        )
        return torch.cat([out, zeros], dim=1)
```
这个函数，

```python
out = F.avg_pool3d(x, kernel_size=1, stride=stride)
```
会将[B, C, T, H, W] 的输入x 变为[B, 1, T, H, W] 然后

```python
zeros = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4),
            device=out.device, dtype=out.dtype,
        )
```
zeros 的维度会是[B, C-1, T, H, W]

```python
return torch.cat([out, zeros], dim=1)
```
拼接起来，所以最后的输出还是[B, C, T, H, W] 但是C维度中只有一个有值，其他是0

```python
for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
对卷积和批量归一化模块进行初始化。
```

```python
def forward_stem(self, x: torch.Tensor) -> torch.Tensor:
    x = self.relu(self.bn1(self.conv1(x)))
    return x if self.no_max_pool else self.maxpool(x)

def forward_features(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"ResNet3D expects [B,C,T,H,W], got {tuple(x.shape)}")
    x = self.forward_stem(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    return torch.flatten(self.avgpool(x), 1)

def forward_head(self, features: torch.Tensor) -> torch.Tensor:
    if self.l2_normalize_before_fc:
        features = F.normalize(features, p=2, dim=1, eps=1e-12)
    return self.fc(features)

def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.forward_head(self.forward_features(x))
```
使用构造的模块进行前向传播



