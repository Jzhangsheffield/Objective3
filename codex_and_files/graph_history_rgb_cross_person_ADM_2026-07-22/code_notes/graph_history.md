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


## data.py

### safe_torch_load(path: str | Path) -> Any
```python
def safe_torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
```
用于加载 pytorch格式数据，其中weights_only表示只加载保存的张量数据，而不可以加载整个模型。这里weights_only的weights不是指权重和偏置中的weights。而是指模型所有参数值。直接加载整个模型可能会有一些自定义的类或者代码被加载，而这些代码中可能含有病毒或者有害信息。

### uniform_frame_indices(num_frames: int, target_frames: int) -> list[int]:
``` python
def uniform_frame_indices(num_frames: int, target_frames: int) -> list[int]:
    if num_frames <= 0 or target_frames <= 0:
        raise ValueError(f"Invalid frame counts: num_frames={num_frames}, target={target_frames}")
    return np.linspace(0, num_frames - 1, target_frames).astype(np.int64).tolist()
```
这个函数是从num_frames中均匀的采样target_frames帧。
这里的np.linspace(start, end, nums) 中是包含start 和 end的。.astype(np.int64)可以保证不能整除的值被设置为整数，并且是直接去掉小数部分不是四舍五入。

### class RGBVideoTransform:
```python
class RGBVideoTransform:
    def __init__(
        self,
        train: bool,
        size: int = 224,
        mean: tuple[float, float, float] = DEFAULT_RGB_MEAN,
        std: tuple[float, float, float] = DEFAULT_RGB_STD,
    ) -> None:
        if train:
            self.transform = v2.Compose(
                [
                    v2.RandomResizedCrop(
                        size=(size, size),
                        scale=(0.6, 1.0),
                        ratio=(0.75, 1.3333333333),
                        antialias=True,
                    ),
                    v2.RandomHorizontalFlip(p=0.0),
                    v2.RandomVerticalFlip(p=0.0),
                    v2.RandomApply(
                        [v2.ColorJitter(brightness=0.24, contrast=0.24, saturation=0.24, hue=0.16)],
                        p=0.5,
                    ),
                    v2.RandomGrayscale(p=0.5),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.Resize(size=(size, size), antialias=True),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=mean, std=std),
                ]
            )

    def __call__(self, video_tchw: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.transform(tv_tensors.Video(video_tchw))).contiguous()
```
这是一个基础的RGB数据增强类，如果是训练模型，则使用RandomResizedCrop，RandomHorizontalFlip，RandomVerticalFlip，ColorJitter， RandomGrayscale，ToDtype，Normalize数据增强管路；如果不是训练模式则只使用Resize，ToDtype，Normalize三种数据增强。
这里要注意tv_tensors.Video()将一般的tensor进行优化，可以让相同的数据增强在所有帧上应用，而不破坏帧与帧之间的时间信息。在应用完数据增强后，使用torch.as_tensor()将其转变回普通的tensor，并用.contiguous()保持起在内容块中的连续性。注意这里使用torch.as_tensor()转回普通不是必要的，因为tv_tensor是tensor的一个子类。
对于contiguous(), 一般一个tensor在创建时，其数据在内容中的存储是连续的，若使用了.permute()等方法，将其维度进行调换，通常不会将底层tensor进行复制，而是改变数据显示的shape和读取的stride，此时显示的tensor在内存中就不是连续的了。
比如
``` python
a = torch.tensor([[1, 2, 3], [2, 3, 4]])
```
```text
其在内存中是这样排列的了1, 2, 3, 2, 3, 4
如果进行了b = a.permute(1, 0)则其变成了[[2, 3, 4], [1, 2, 3]], 但其内存排列没变，所以对于b其内存排列是不连续的，需要使用.contiguous()。
```

### class RGBClipDataset(Dataset):
```python
class RGBClipDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        manifest: str | Path,
        camera_id: str = DEFAULT_CAMERA_ID,
        n_frames: int = 16,
        rgb_size: int = 224,
        train: bool = False,
        verify_paths: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest_path = resolve_manifest(self.dataset_root, manifest)
        self.camera_id = str(camera_id)
        self.n_frames = int(n_frames)
        self.rows = read_jsonl(self.manifest_path)
        self.transform = RGBVideoTransform(train=train, size=rgb_size)
        if verify_paths:
            missing: list[str] = []
            field = f"{self.camera_id}_rgb"
            for row in self.rows:
                rel = row.get(field)
                if not rel or not (self.dataset_root / str(rel)).is_file():
                    missing.append(str(row.get("sample_name", "<unknown>")))
                    if len(missing) >= 10:
                        break
            if missing:
                raise FileNotFoundError(
                    f"Missing camera {self.camera_id} RGB tensors for examples: {missing}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        rel_path = row[f"{self.camera_id}_rgb"]
        obj = safe_torch_load(self.dataset_root / rel_path)
        video = obj["frames"] if isinstance(obj, dict) else obj
        if not torch.is_tensor(video) or video.ndim != 4 or video.shape[1] != 3:
            raise ValueError(
                f"Invalid RGB tensor for {row['sample_name']}: {type(video)} {getattr(video, 'shape', None)}"
            )
        indices = uniform_frame_indices(int(video.shape[0]), self.n_frames)
        video = self.transform(video[indices])
        # ResNet3D input is [C,T,H,W] per sample.
        video = video.permute(1, 0, 2, 3).contiguous()
        return {
            "video": video,
            "tier3_target": int(row["tier3_id"]),
            "node_target": int(row["node_idx"]) - 1,
            "stage_id": int(row["stage_id"]),
            "sample_name": str(row["sample_name"]),
            "participant": str(row["participant"]),
            "run": str(row["run"]),
            "annotation_row_index": int(row["annotation_row_index"]),
        }
```
self.rows = read_jsonl(self.manifest_path)代码如下：
```python
def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows
```
```text
json和jsonl梳理:
json可以理解为一种文件结构，该结构中只能有以下几种类型的数据组成：
JSON 类型	            示例	           json.load() 是否可以解析
Object          {"name":"Tom"}	         ✅
Array	        [1,2,3]	                   ✅
String	        "Hello"	                   ✅
Number	        123	                   ✅
Boolean	        true	                   ✅
Null	        null	                   ✅
其可以是txt文件也可以是其他文件，不一定非要是json文件。
这是因为如此，json结构可以直接和python中的数据集结构进行转换。
json包中的json.load(),json.loads(),json.dump(),json.dumps()就是为读取和写入json文件而设计的。
具体的区别是json.load()是直接将json格式的数据读取成python对应的数据类型。
而json.loads()则是从字符串转换成python对应的数据类型，比如"{"boy": 3, "girl": 2}", 使用json.loads()可以直接读取成dict。
json.load(), 相当于f = open("r", path)-> x = f.read()-> json.loads(x)
对于.dump()和.dumps()也是一样的。

jsonl和json不是一种文件结构，json中只能有一个整体的json对象，可以理解为json文件中最外层只能有一个json类型。
而jsonl中每一行都是一个json对象，因此jsonl往往要逐行读取。
```
对于read_jsonl()代码的解析:
```python
with Path(path).open("r", encoding="utf-8") as handle:
```
这一句打开jsonl文件，
```python
for line_number, line in enumerate(handle, start=1):
```
开始遍历每一行，每一行是一个字符串，这里文件对象是按行迭代的。
```python
line = line.strip()
if not line:
    continue
try:
    rows.append(json.loads(line))
```
每一行两端去空格，非空则将其解析成python对象，并添加到rows列表中。注意这里只能使用json.loads()因为line是字符串
```python
except json.JSONDecodeError as exc:
    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
```
如果解析出错误就报错。

继续进行后续代码解析
```python
if verify_paths:
    missing: list[str] = []
    field = f"{self.camera_id}_rgb"
    for row in self.rows:
        rel = row.get(field)
        if not rel or not (self.dataset_root / str(rel)).is_file():
            missing.append(str(row.get("sample_name", "<unknown>")))
            if len(missing) >= 10:
                break
    if missing:
        raise FileNotFoundError(
            f"Missing camera {self.camera_id} RGB tensors for examples: {missing}"
        )
```
这里检查每一行中是否有field键开头的数据，如果有呢就.get()取其对应的值（也就是文件对于数据集路径的相对路径），如果值为空，或者路径不是一个文件，则将该row对应的样本明添加到missing列表里，如果超过10个missing，就直接中断循环。最后如果missing不为空，也就是有缺失样本则报错。
```python
def __len__(self) -> int:
        return len(self.rows)
```
返回数据集长度
```python
def __getitem__(self, index: int) -> dict[str, Any]:
    row = self.rows[index]
    rel_path = row[f"{self.camera_id}_rgb"]
    # 获取某一个row对应的数据的路径
    obj = safe_torch_load(self.dataset_root / rel_path)
    # 载入这个数据
    video = obj["frames"] if isinstance(obj, dict) else obj
    # 如果数据是字典则获得其中"frames"键对应的值，不是则返回obj
    if not torch.is_tensor(video) or video.ndim != 4 or video.shape[1] != 3:
        raise ValueError(
            f"Invalid RGB tensor for {row['sample_name']}: {type(video)} {getattr(video, 'shape', None)}"
        )
    # video如果不是tensor, 或者维度不是4，或者第2维度大小不是3则报错。
    # 也就是说video形状应该是 tchw
    indices = uniform_frame_indices(int(video.shape[0]), self.n_frames)
    # 根据视频帧数来均匀采样帧
    video = self.transform(video[indices])
    # 进行数据增强
    # ResNet3D input is [C,T,H,W] per sample.
    video = video.permute(1, 0, 2, 3).contiguous()
    return {
        "video": video,
        "tier3_target": int(row["tier3_id"]),
        "node_target": int(row["node_idx"]) - 1,
        "stage_id": int(row["stage_id"]),
        "sample_name": str(row["sample_name"]),
        "participant": str(row["participant"]),
        "run": str(row["run"]),
        "annotation_row_index": int(row["annotation_row_index"]),
    }
    # 返回一个字典，也就是一个样本
```

```python
@dataclass(frozen=True)
class HistoryExample:
    current_cache_index: int
    history_cache_indices: tuple[int, ...]
    current_row: dict[str, Any]
    history_rows: tuple[dict[str, Any], ...]
```
这里dataclass可以快速构建只有数据的类，在实例化时直接传入对应数据即可。

```python
def load_feature_cache(path: str | Path) -> dict[str, Any]:
    cache = safe_torch_load(path)
    required = {"features", "tier3_logits", "records", "metadata"}
    if not isinstance(cache, dict) or not required.issubset(cache):
        raise ValueError(f"Feature cache {path} does not contain {sorted(required)}")
    if len(cache["records"]) != int(cache["features"].shape[0]):
        raise ValueError("Feature cache record/feature count mismatch")
    return cache
```
载入预提取的特征，检查特征文件中是否含有所需的关键字，以及样本数量是否对应。

```python
class FeatureHistoryDataset(Dataset):
    def __init__(
        self,
        feature_cache_path: str | Path,
        selection_manifest: str | Path,
        history_order: str,
        graph: TaskGraphSpec | None = None,
        shuffle_seed: int = 1,
    ) -> None:
        if history_order not in {"actual", "graph_valid"}:
            raise ValueError(f"Unsupported history_order: {history_order}")
        self.cache = load_feature_cache(feature_cache_path)
        self.features: torch.Tensor = self.cache["features"].float()
        self.selection_rows = read_jsonl(selection_manifest)
        self.history_order = history_order
        self.graph = graph
        self.shuffle_seed = int(shuffle_seed)
        if history_order == "graph_valid" and graph is None:
            raise ValueError("graph_valid history requires a TaskGraphSpec")

        cache_lookup = {
            str(row["sample_name"]): index
            for index, row in enumerate(self.cache["records"])
        }
        missing = [
            str(row["sample_name"])
            for row in self.selection_rows
            if str(row["sample_name"]) not in cache_lookup
        ]
        if missing:
            raise KeyError(f"Selection manifest samples missing from feature cache: {missing[:10]}")

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in self.selection_rows:
            grouped.setdefault(run_key(row), []).append(row)

        self.examples: list[HistoryExample] = []
        for rows in grouped.values():
            rows.sort(key=lambda row: int(row["annotation_row_index"]))
            for current_position, current_row in enumerate(rows):
                actual_history = list(rows[:current_position])
                if history_order == "graph_valid":
                    actual_history = randomized_graph_valid_history(
                        actual_history,
                        graph=self.graph,
                        seed=stable_sample_seed(self.shuffle_seed, str(current_row["sample_name"])),
                    )
                self.examples.append(
                    HistoryExample(
                        current_cache_index=cache_lookup[str(current_row["sample_name"])],
                        history_cache_indices=tuple(
                            cache_lookup[str(row["sample_name"])] for row in actual_history
                        ),
                        current_row=current_row,
                        history_rows=tuple(actual_history),
                    )
                )
        self.examples.sort(
            key=lambda example: (
                str(example.current_row["participant"]),
                str(example.current_row["run"]),
                int(example.current_row["annotation_row_index"]),
            )
        )

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        if example.history_cache_indices:
            history_indices = torch.tensor(example.history_cache_indices, dtype=torch.long)
            history_features = self.features.index_select(0, history_indices)
        else:
            history_features = self.features.new_zeros((0, self.feature_dim))
        length = len(example.history_rows)
        # Position IDs encode distance in the presented sequence; 1 means most recent.
        position_ids = torch.arange(length, 0, -1, dtype=torch.long)
        history_node_classes = torch.tensor(
            [int(row["node_idx"]) - 1 for row in example.history_rows], dtype=torch.long
        )
        row = example.current_row
        return {
            "current_feature": self.features[example.current_cache_index],
            "history_features": history_features,
            "history_position_ids": position_ids,
            "history_node_classes": history_node_classes,
            "node_target": int(row["node_idx"]) - 1,
            "tier3_target": int(row["tier3_id"]),
            "stage_id": int(row["stage_id"]),
            "sample_name": str(row["sample_name"]),
            "participant": str(row["participant"]),
            "run": str(row["run"]),
            "annotation_row_index": int(row["annotation_row_index"]),
            "history_sample_names": [str(hist["sample_name"]) for hist in example.history_rows],
        }
```
```python
class FeatureHistoryDataset(Dataset):
    def __init__(
        self,
        feature_cache_path: str | Path,
        selection_manifest: str | Path,
        history_order: str,
        graph: TaskGraphSpec | None = None,
        shuffle_seed: int = 1,
    ) -> None:
        if history_order not in {"actual", "graph_valid"}:
            raise ValueError(f"Unsupported history_order: {history_order}")
        self.cache = load_feature_cache(feature_cache_path)
        self.features: torch.Tensor = self.cache["features"].float()
        self.selection_rows = read_jsonl(selection_manifest)
        self.history_order = history_order
        self.graph = graph
        self.shuffle_seed = int(shuffle_seed)
        if history_order == "graph_valid" and graph is None:
            raise ValueError("graph_valid history requires a TaskGraphSpec")
```
这里的重点是理解graph: TaskGraphSpec和history_order, 这两个将在graph.py这一个文件中进行详细分析
```python
cache_lookup = {
    str(row["sample_name"]): index
    for index, row in enumerate(self.cache["records"])
}
missing = [
    str(row["sample_name"])
    for row in self.selection_rows
    if str(row["sample_name"]) not in cache_lookup
]
if missing:
    raise KeyError(f"Selection manifest samples missing from feature cache: {missing[:10]}")
```
cache_lookup建立"sample_name"和index之间的映射。
如果"sample_name在self.selection_rows但是不在cache_lookup中，则说明没有这个样本的特征，所以将其添加到missing列表中。
如果missing列表不是空，则报错并显示出前10个没有特征的样本。
```python
grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
for row in self.selection_rows:
    grouped.setdefault(run_key(row), []).append(row)
```
run_key()这个函数如下:
```python
def run_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["participant"]), str(row["run"])
```
其返回一个元组，第一个元素是参与人，第二个元素是参与人的第几个run。
grouped.setdefault(run_key(row), []).append(row)
这一句对于每一个row, 将run_key()返回的元组作为key，并初始化一个空列表。并将对应的row添加到列表中。
这样grouped是一个以一个元组为键，以一个列表为值，列表中的row都属于同一个人的同一个run。
这里如果grouped中有对应的键，就返回键对应的值，如果没有就返回一个空列表，这就是为什么可以直接.append()的原因。
```python
self.examples: list[HistoryExample] = []
# self.examples: list[HistoryExample] = [] 中的元素是HistoryExample(dataclass)
for rows in grouped.values():
    # 对于grouped的值进行遍历（每一个值是一个列表）
    rows.sort(key=lambda row: int(row["annotation_row_index"]))
    # 将列表进行原地排序，排序的键是每一个样本对应的"annotation_row_index"
    for current_position, current_row in enumerate(rows):
        # current_position 从0开始
        actual_history = list(rows[:current_position])
        # 对于每一个样本，获得其前面的样本（不包括自己）
        if history_order == "graph_valid": # 如果要使用"graph_valid" history_order，就使用andomized_graph_valid_history 重新排列。
            actual_history = randomized_graph_valid_history(
                actual_history,
                graph=self.graph,
                seed=stable_sample_seed(self.shuffle_seed, str(current_row["sample_name"])),
            )
        self.examples.append( #构建一个样本的HistoryExample并添加到self.examples中
            HistoryExample(
                current_cache_index=cache_lookup[str(current_row["sample_name"])],
                # 找到样本在cache中对应的index
                history_cache_indices=tuple(
                    cache_lookup[str(row["sample_name"])] for row in actual_history
                ),
                # 找到样本之前的样本在cache中对应的index
                current_row=current_row,
                # 样本row
                history_rows=tuple(actual_history),
                # 样本之前的样本的row
            )
        )
self.examples.sort(
    key=lambda example: (
        str(example.current_row["participant"]),
        str(example.current_row["run"]),
        int(example.current_row["annotation_row_index"]),
    )
)
# 对self.examples进行原地排序，排序的原则是先按参与人员排，再按run排，再按样本再该run中发生的顺序排。
```
```python
@property
def feature_dim(self) -> int:
    return int(self.features.shape[1])

def __len__(self) -> int:
    return len(self.examples)
```
这里@property可以让方法向属性一样使用，获得特征的维度。
```python
def __getitem__(self, index: int) -> dict[str, Any]:
    example = self.examples[index]
    # 取一个样本
    if example.history_cache_indices:
        history_indices = torch.tensor(example.history_cache_indices, dtype=torch.long)
        history_features = self.features.index_select(0, history_indices)
    # 如果样本有对应的cache index，则获得该样本的特征
    else:
        history_features = self.features.new_zeros((0, self.feature_dim))
    # 样本没有对应的cache index, 则构建一个0张量
    length = len(example.history_rows)
    # 获得对应的历史信息长度
    # Position IDs encode distance in the presented sequence; 1 means most recent.
    position_ids = torch.arange(length, 0, -1, dtype=torch.long)
    # 给历史信息编码，倒着编码, 则1表示最近的历史动作
    history_node_classes = torch.tensor(
        [int(row["node_idx"]) - 1 for row in example.history_rows], dtype=torch.long
    )
    这里将每一个样本对应的node_idx减去1， 因为node中有start node, start node，不参与分类。
    row = example.current_row
    return { # 返回一个样本的字典
        "current_feature": self.features[example.current_cache_index],
        "history_features": history_features,
        "history_position_ids": position_ids,
        "history_node_classes": history_node_classes,
        "node_target": int(row["node_idx"]) - 1,
        "tier3_target": int(row["tier3_id"]),
        "stage_id": int(row["stage_id"]),
        "sample_name": str(row["sample_name"]),
        "participant": str(row["participant"]),
        "run": str(row["run"]),
        "annotation_row_index": int(row["annotation_row_index"]),
        "history_sample_names": [str(hist["sample_name"]) for hist in example.history_rows],
    }
```
这里.index_select(0, history_indices), 表示沿着第一维度取对应的索引的特征。
.new_zeros((0, self.feature_dim))表示新建一个全零张量，其第一维是0，有512列。
具体长这样：
tensor([], size=(0, 512))
torch.Size([0, 512])
```text
            第0列 第1列 第2列 第3列 第4列
            ───────────────────────────
(没有任何一行)
```

### def collate_history_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
```python
def collate_history_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    # 获得batch 大小
    feature_dim = int(batch[0]["current_feature"].shape[0])
    # 获得特征的维度
    max_history = max(int(item["history_features"].shape[0]) for item in batch)
    # 获得batch中样本的最大的历史特征数
    history_features = torch.zeros((batch_size, max_history, feature_dim), dtype=torch.float32)
    # 构建大小为batch_size, max_history, feature_dim的空张量，类型为float32
    history_positions = torch.zeros((batch_size, max_history), dtype=torch.long)
    # 构建大小为batch_szie, max_history的空张量，类型为long
    history_nodes = torch.full((batch_size, max_history), -1, dtype=torch.long)
    # 构建大小为batch_size, max_history，值全为-1的张量
    history_mask = torch.ones((batch_size, max_history), dtype=torch.bool)
    # 构建batch_size, max_history，值全为1的张量
    for row_index, item in enumerate(batch):
        length = int(item["history_features"].shape[0])
        # 对于batch 中每一个样本，获得其历史特征长度。
        if length:
            history_features[row_index, :length] = item["history_features"]
            history_positions[row_index, :length] = item["history_position_ids"]
            history_nodes[row_index, :length] = item["history_node_classes"]
            history_mask[row_index, :length] = False
            # 将历史信息对应的位置的值设置为False
            # 使用样本具体的值来填充刚刚构建的张量，以保持每一个样本的维度大小一致，才能够组成batch.
    return {
        "current_feature": torch.stack([item["current_feature"] for item in batch]),
        "history_features": history_features,
        "history_position_ids": history_positions,
        "history_node_classes": history_nodes,
        "history_padding_mask": history_mask,
        "node_target": torch.tensor([item["node_target"] for item in batch], dtype=torch.long),
        "tier3_target": torch.tensor([item["tier3_target"] for item in batch], dtype=torch.long),
        "stage_id": torch.tensor([item["stage_id"] for item in batch], dtype=torch.long),
        "sample_name": [item["sample_name"] for item in batch],
        "participant": [item["participant"] for item in batch],
        "run": [item["run"] for item in batch],
        "annotation_row_index": [item["annotation_row_index"] for item in batch],
        "history_sample_names": [item["history_sample_names"] for item in batch],
    }
```
这里在构建dataloader时传入，collate_history_batch，则dataloader一个batch的返回值就变成这个函数的返回值。具体流程如下：
```text
                                            Dataset.__getitem__()
                                                    │
                                                    │ 取出单个样本
                                                    ▼
                                            sample_1
                                            sample_2
                                            sample_3
                                            sample_4
                                                    │
                                                    ▼
                                            [
                                            sample_1,
                                            sample_2,
                                            sample_3,
                                            sample_4
                                            ]
                                                    │
                                                    │ 传给 collate_fn
                                                    ▼
                                            collate_history_batch(...)
                                                    │
                                                    ▼
                                            return {...}
                                                    │
                                                    ▼
                                            for batch in DataLoader:
                                                    │
                                                    ▼
                                            batch 就是这个 {...}
```

## graph.py
### class TaskGraphSpec:
```python
@dataclass(frozen=True)
class TaskGraphSpec:
    task_graph_path: Path
    relation_matrix_path: Path
    graph_json: dict[str, Any]
    relation_json: dict[str, Any]
    relation_ids: torch.Tensor
    node_to_tier3: torch.Tensor
    node_to_stage: torch.Tensor
    all_must_previous: dict[int, tuple[int, ...]]
    immediate_previous: dict[int, int | None]
    atomic_sequences: tuple[tuple[int, ...], ...]

    @classmethod
    def load(cls, task_graph_path: str | Path, relation_matrix_path: str | Path) -> "TaskGraphSpec":
        task_graph_path = Path(task_graph_path)
        relation_matrix_path = Path(relation_matrix_path)
        graph_json = read_json(task_graph_path)
        relation_json = read_json(relation_matrix_path)

        nodes = {int(node["node_idx"]): node for node in graph_json["nodes"]}
        expected = set(range(1, NUM_GRAPH_NODES + 1))
        if not expected.issubset(nodes):
            missing = sorted(expected - set(nodes))
            raise ValueError(f"Task graph is missing action nodes: {missing}")

        node_to_tier3 = torch.tensor(
            [int(nodes[idx]["action_id_tier3"]) for idx in range(1, NUM_GRAPH_NODES + 1)],
            dtype=torch.long,
        )
        node_to_stage = torch.tensor(
            [int(nodes[idx]["stage_id"]) for idx in range(1, NUM_GRAPH_NODES + 1)],
            dtype=torch.long,
        )

        columns = [int(value) for value in relation_json["column_node_idx"]]
        column_lookup = {node_idx: column for column, node_idx in enumerate(columns)}
        rows = {int(row["current_node_idx"]): row["values"] for row in relation_json["rows"]}
        relation_ids = torch.empty((NUM_GRAPH_NODES, NUM_GRAPH_NODES), dtype=torch.long)
        for current_node in range(1, NUM_GRAPH_NODES + 1):
            for previous_node in range(1, NUM_GRAPH_NODES + 1):
                code = rows[current_node][column_lookup[previous_node]]
                normalized = "X" if code == "." else str(code)
                if normalized not in RELATION_TO_ID:
                    raise ValueError(
                        f"Unsupported relation code {code!r} for ({current_node}, {previous_node})"
                    )
                relation_ids[current_node - 1, previous_node - 1] = RELATION_TO_ID[normalized]

        all_must_previous: dict[int, tuple[int, ...]] = {}
        immediate_previous: dict[int, int | None] = {}
        for node_idx in range(1, NUM_GRAPH_NODES + 1):
            node = nodes[node_idx]
            history = node["feature_history_constraints"]["all_must_previous_nodes"]
            all_must_previous[node_idx] = tuple(int(value) for value in history if 1 <= int(value) <= 35)
            immediate = node["execution_constraints"].get("must_immediately_previous_node")
            immediate_previous[node_idx] = int(immediate) if immediate is not None else None

        atomic_sequences = tuple(
            tuple(int(value) for value in item["nodes"] if 1 <= int(value) <= 35)
            for item in graph_json.get("atomic_sequences", [])
        )

        return cls(
            task_graph_path=task_graph_path,
            relation_matrix_path=relation_matrix_path,
            graph_json=graph_json,
            relation_json=relation_json,
            relation_ids=relation_ids,
            node_to_tier3=node_to_tier3,
            node_to_stage=node_to_stage,
            all_must_previous=all_must_previous,
            immediate_previous=immediate_previous,
            atomic_sequences=atomic_sequences,
        )
```
```python
@dataclass(frozen=True)
class TaskGraphSpec:
    task_graph_path: Path
    relation_matrix_path: Path
    graph_json: dict[str, Any]
    relation_json: dict[str, Any]
    relation_ids: torch.Tensor
    node_to_tier3: torch.Tensor
    node_to_stage: torch.Tensor
    all_must_previous: dict[int, tuple[int, ...]]
    immediate_previous: dict[int, int | None]
    atomic_sequences: tuple[tuple[int, ...], ...]

    @classmethod
    def load(cls, task_graph_path: str | Path, relation_matrix_path: str | Path) -> "TaskGraphSpec":
        task_graph_path = Path(task_graph_path)
        relation_matrix_path = Path(relation_matrix_path)
        graph_json = read_json(task_graph_path)
        relation_json = read_json(relation_matrix_path)

        nodes = {int(node["node_idx"]): node for node in graph_json["nodes"]}
        expected = set(range(1, NUM_GRAPH_NODES + 1))
        if not expected.issubset(nodes):
            missing = sorted(expected - set(nodes))
            raise ValueError(f"Task graph is missing action nodes: {missing}")

        node_to_tier3 = torch.tensor(
            [int(nodes[idx]["action_id_tier3"]) for idx in range(1, NUM_GRAPH_NODES + 1)],
            dtype=torch.long,
        )
        node_to_stage = torch.tensor(
            [int(nodes[idx]["stage_id"]) for idx in range(1, NUM_GRAPH_NODES + 1)],
            dtype=torch.long,
        )
```
这里graph_json中"nodes"下是的每一个节点元素是字典。
```python
nodes = {int(node["node_idx"]): node for node in graph_json["nodes"]}
```
相当于建立一个映射，是"node_idx"到每个node字典的映射。
```python
expected = set(range(1, NUM_GRAPH_NODES + 1))
```
是一个包含元素从1到NUM_GRAPH_NODES的集合。
```python
if not expected.issubset(nodes):
    missing = sorted(expected - set(nodes))
    raise ValueError(f"Task graph is missing action nodes: {missing}")
```
检查expected 是否是nodes的子集（因为nodes包含start node 和 end node, 这里NUM_GRAPH_NODES指的是不包含start 和 end node的真实有含义的节点数量）。NUM_GRAPH_NODES=35，包含start 和 end node 共有37个节点，index 从0到36.而真正有含义的节点编号是从1-35.
```python
node_to_tier3 = torch.tensor(
    [int(nodes[idx]["action_id_tier3"]) for idx in range(1, NUM_GRAPH_NODES + 1)],
    dtype=torch.long,
)
```
取出每一个node的对应的动作类别标签。
```python
node_to_stage = torch.tensor(
    [int(nodes[idx]["stage_id"]) for idx in range(1, NUM_GRAPH_NODES + 1)],
    dtype=torch.long,
)
```
取出每一个node对应的阶段标签。
```python
columns = [int(value) for value in relation_json["column_node_idx"]]
column_lookup = {node_idx: column for column, node_idx in enumerate(columns)}
rows = {int(row["current_node_idx"]): row["values"] for row in relation_json["rows"]}
relation_ids = torch.empty((NUM_GRAPH_NODES, NUM_GRAPH_NODES), dtype=torch.long)
for current_node in range(1, NUM_GRAPH_NODES + 1):
    for previous_node in range(1, NUM_GRAPH_NODES + 1):
        code = rows[current_node][column_lookup[previous_node]]
        normalized = "X" if code == "." else str(code)
        if normalized not in RELATION_TO_ID:
            raise ValueError(
                f"Unsupported relation code {code!r} for ({current_node}, {previous_node})"
            )
        relation_ids[current_node - 1, previous_node - 1] = RELATION_TO_ID[normalized]
```
这里relation_json中的"column_node_idx"是一个列表，其内的值从0到36表示这每一列对应的节点。
```python
columns = [int(value) for value in relation_json["column_node_idx"]]
column_lookup = {node_idx: column for column, node_idx in enumerate(columns)}
```
这里columns是一个列表内含元素从0到36.而column_lookup是一个字典，其键和值都是相同的内容，都是int类型，从0到36.
```python
rows = {int(row["current_node_idx"]): row["values"] for row in relation_json["rows"]}
relation_ids = torch.empty((NUM_GRAPH_NODES, NUM_GRAPH_NODES), dtype=torch.long)
```
这里relation_json["rows"]是一个列表，列表中元素是字典，字典有两个键值对，第一个键值对"current_node_idx"，表示当前节点的index，第二个键值对"values"，是一个列表，表示当前node与所有node的关系。
这里rows是一个字典，字典的键是当前节点index, 值是其与所有node的关系。
```python
relation_ids构建一个空的张量，大小为(NUM_GRAPH_NODES, NUM_GRAPH_NODES), 也就是(35, 35).
for current_node in range(1, NUM_GRAPH_NODES + 1):
    for previous_node in range(1, NUM_GRAPH_NODES + 1):
        code = rows[current_node][column_lookup[previous_node]]
        # 找到当前node和其他所有node的关系
        normalized = "X" if code == "." else str(code)
        # 如果code是"."则normalized 设置为"X", 否则为str(code)
        if normalized not in RELATION_TO_ID:
            raise ValueError(
                f"Unsupported relation code {code!r} for ({current_node}, {previous_node})"
            )
        relation_ids[current_node - 1, previous_node - 1] = RELATION_TO_ID[normalized]
        # 这里relation_ids构建时，就没有包括start, end node. 但是current_node和previous_node构建时，是包括
        # start 和 end node的。所以减去1才能对应的上。
```
```python
all_must_previous: dict[int, tuple[int, ...]] = {}
immediate_previous: dict[int, int | None] = {}
for node_idx in range(1, NUM_GRAPH_NODES + 1):
    node = nodes[node_idx]
    history = node["feature_history_constraints"]["all_must_previous_nodes"]
    # 取出每一个node对应的"all_must_previous_nodes"
    all_must_previous[node_idx] = tuple(int(value) for value in history if 1 <= int(value) <= 35)
    # 从history node 中去除掉 start 和 end node。
    immediate = node["execution_constraints"].get("must_immediately_previous_node")
    # 取出node 对应的必须紧接着在其前面的node
    immediate_previous[node_idx] = int(immediate) if immediate is not None else None

atomic_sequences = tuple(
    tuple(int(value) for value in item["nodes"] if 1 <= int(value) <= 35)
    for item in graph_json.get("atomic_sequences", [])
)
# "atomic_sequences" 是一个列表，其里面的元素是字典。 item["nodes"]是一个列表，其里面的元素是int值，表示节点编号。
# 其就是将原子序列列表，转换成元组，并放在一个元组中。
```
```python
return cls(
    task_graph_path=task_graph_path,
    relation_matrix_path=relation_matrix_path,
    graph_json=graph_json,
    relation_json=relation_json,
    relation_ids=relation_ids,
    node_to_tier3=node_to_tier3,
    node_to_stage=node_to_stage,
    all_must_previous=all_must_previous,
    immediate_previous=immediate_previous,
    atomic_sequences=atomic_sequences,
)
```
返回一个TaskGraphSpec实例。当我们调用类方法时，第一个变量cls就是类自己本身，所以cls等于TaskGraphSpec

### def stable_sample_seed(base_seed: int, sample_name: str) -> int:
```python
def stable_sample_seed(base_seed: int, sample_name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)
```
基于随机种子和样本名称，为每一个样本生成一个独有的整形值。

### def randomized_graph_valid_history()
```python
def randomized_graph_valid_history(
    history_rows: list[dict[str, Any]],
    graph: TaskGraphSpec,
    seed: int,
) -> list[dict[str, Any]]:
    """Return one deterministic randomized topological order of observed history.

    Only relations among observed history nodes are used.  The current target node is
    deliberately not an input, preventing current-label leakage.  Runs containing a
    repeated graph node fall back to actual order; the primary M3 protocol uses normal
    runs, where nodes are unique.
    """
    if len(history_rows) <= 1:
        return list(history_rows)
    # 如果history_rows长度小于等于1，直接返回

    node_indices = [int(row["node_idx"]) for row in history_rows]
    if len(set(node_indices)) != len(node_indices):
        return list(history_rows)
    # 取出 history_rows中每一个row，也就是样本对应的节点index
    # 如果历史中有重复的节点，则直接返回history_rows.

    row_by_node = {int(row["node_idx"]): row for row in history_rows}
    # 构建节点和历史row的对应关系。
    observed = set(row_by_node)
    # 去重
    assigned: set[int] = set()
    blocks: list[list[int]] = []

    for sequence in graph.atomic_sequences:
        # 这里graph.atomic_sequence是一个元组，里面元素也是元组
        block = [node for node in sequence if node in observed]
        if block:
            blocks.append(block)
            # 将block列表添加到blocks
            assigned.update(block)
            # 将block 中元素添加到集合中
        # 检查历史中是否含有atomic_sequences, 如果有的化，就将其整体添加到blocks和assigned中。
    
    for node_idx in node_indices:
        if node_idx not in assigned:
            blocks.append([node_idx])
            assigned.add(node_idx)
    # 如果节点没有被使用，则将单个节点放入列表，再将列表添加到blocks中，并将节点更新到集合中。

    node_to_block: dict[int, int] = {}
    for block_idx, block in enumerate(blocks):
        for node_idx in block:
            node_to_block[node_idx] = block_idx
    # 构建每一个节点到所属的block的映射。

    successors: dict[int, set[int]] = {idx: set() for idx in range(len(blocks))}
    # 
    indegree = {idx: 0 for idx in range(len(blocks))}
    # 为每一个block构建入度
    for current_node in observed:
        # 对于每一个历史节点
        current_block = node_to_block[current_node]
        # 找到节点所属的block
        for previous_node in graph.all_must_previous[current_node]:
            # 对于每一个当前节点对应的必须之前发生的节点
            if previous_node not in observed:
                continue
            # 如果该之前节点不在历史节点中，则跳过。
            previous_block = node_to_block[previous_node]
            # 查找之前必须节点所属的block
            if previous_block == current_block or current_block in successors[previous_block]:
                continue
            # 如果当前节点和其之前必须节点属于同一个block，或者当前节点所有的block是之前必须节点的后续节点则跳过
            successors[previous_block].add(current_block)
            # 当前节点block是之前必须节点的后续block则，添加到successors
            indegree[current_block] += 1
            # 当前节点block的入度加1，

    rng = random.Random(seed)
    available = [idx for idx, degree in indegree.items() if degree == 0]
    # 入度为0的block才可以参入排序
    ordered_blocks: list[int] = []
    while available:
        selected = rng.choice(available)
        available.remove(selected)
        ordered_blocks.append(selected)
        # 有可排序的block, 则从可排序的Block中随机抽一个，并将其从可选block中去除。
        for successor in sorted(successors[selected]):
            # 对于被选中block的每一个后续block
            indegree[successor] -= 1
            # 后续block的入度减少1
            if indegree[successor] == 0:
                available.append(successor)
            # 如果后续block的入度变为0，则将其添加到可排序block列表中。

    if len(ordered_blocks) != len(blocks):
        raise RuntimeError("Observed task-graph history unexpectedly contains a cycle")

    ordered_nodes = [node for block_idx in ordered_blocks for node in blocks[block_idx]]
    # 将排好序的block的中的node展开。
    return [row_by_node[node] for node in ordered_nodes]
    # 返回node对应的row列表
```

## models.py
### class FeatureNodeClassifier(nn.Module):
```python
class FeatureNodeClassifier(nn.Module):
    """M0: current frozen RGB feature -> 35 graph-node logits."""

    def __init__(self, feature_dim: int = 512, num_nodes: int = NUM_GRAPH_NODES, dropout: float = 0.0):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_nodes = int(num_nodes)
        self.norm = nn.LayerNorm(self.feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feature_dim, self.num_nodes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.norm(features)))
```
直接在动作识别提取的特征基础上，训练一个分类头用于识别节点。

### def freeze_module(module: nn.Module) -> None:
```python
def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
```
冻结指定模块，使其不更新参数

### class SingleQueryHistoryModel(nn.Module):
```python
class SingleQueryHistoryModel(nn.Module):
    """M1-M3: one current query attends to the same-run causal history."""

    def __init__(
        self,
        baseline: FeatureNodeClassifier,
        feature_dim: int = 512,
        d_model: int = 256,
        num_heads: int = 4,
        max_history: int = 35,
        dropout: float = 0.1,
        use_position: bool = True,
    ) -> None:
        super().__init__()
        self.baseline = baseline
        freeze_module(self.baseline)
        self.use_position = bool(use_position)
        self.max_history = int(max_history)
        self.current_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.history_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.position_embedding = nn.Embedding(max_history + 1, d_model)
        self.null_history = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_history, std=0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_GRAPH_NODES),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        self.history_scale_logit = nn.Parameter(torch.tensor(-2.0))

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        return self

    def forward(
        self,
        current_feature: torch.Tensor,
        history_features: torch.Tensor,
        history_position_ids: torch.Tensor,
        history_padding_mask: torch.Tensor,
        **_: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        current = self.current_projection(current_feature)
        history = self.history_projection(history_features)
        if self.use_position and history.shape[1] > 0:
            positions = history_position_ids.clamp(min=0, max=self.max_history)
            history = history + self.position_embedding(positions)

        null = self.null_history.expand(current.shape[0], -1, -1)
        history = torch.cat([null, history], dim=1)
        null_mask = torch.zeros((current.shape[0], 1), dtype=torch.bool, device=current.device)
        key_padding_mask = torch.cat([null_mask, history_padding_mask], dim=1)
        context, attention_weights = self.attention(
            current.unsqueeze(1), history, history,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        context = context.squeeze(1)
        delta = self.delta_head(torch.cat([current, context], dim=-1))
        scale = torch.sigmoid(self.history_scale_logit)
        with torch.no_grad():
            baseline_logits = self.baseline(current_feature)
        logits = baseline_logits + scale * delta
        return logits, {
            "baseline_logits": baseline_logits,
            "history_delta": delta,
            "history_scale": scale.detach(),
            "attention": attention_weights,
        }
```
```python
class SingleQueryHistoryModel(nn.Module):
    """M1-M3: one current query attends to the same-run causal history."""

    def __init__(
        self,
        baseline: FeatureNodeClassifier,
        feature_dim: int = 512,
        d_model: int = 256,
        num_heads: int = 4,
        max_history: int = 35,
        dropout: float = 0.1,
        use_position: bool = True,
    ) -> None:
        super().__init__()
        self.baseline = baseline
        freeze_module(self.baseline)
        # 冻结M0模型
        self.use_position = bool(use_position)
        self.max_history = int(max_history)
        self.current_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.history_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        # 当前节点和历史节点特征投影头， 将维度从512变为256
        self.position_embedding = nn.Embedding(max_history + 1, d_model)
        # 位置编码，构建一个大小为 （36， 256）的可学习张量
        self.null_history = nn.Parameter(torch.zeros(1, 1, d_model))
        # 构建一个全零的维度为（1，1，256)的可学习张量
        nn.init.normal_(self.null_history, std=0.02)
        # 使用正态分布初始化
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # 创建一个有4个注意力头的多头自注意力模块
        self.delta_head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_GRAPH_NODES),
        )
        # 创建一个线性模型，用于修正节点的预测
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        # 使用0初始化其权重和偏置
        self.history_scale_logit = nn.Parameter(torch.tensor(-2.0))
        # 创建一个可学习的张量，并且初始化值为-2.0
```
```python
def train(self, mode: bool = True):
    super().train(mode)
    self.baseline.eval()
    return self
```
借助nn.Module()的.train()方法，在训练时，先设置整个模型可训练，再通过.eval()将M0模型设置为评估模式，让其参数不更新，最后返回这个模型，即可以训练除了M0模型之外的模块参数。
```python
def forward(
    self,
    current_feature: torch.Tensor,
    history_features: torch.Tensor,
    history_position_ids: torch.Tensor,
    history_padding_mask: torch.Tensor,
    **_: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    current = self.current_projection(current_feature)
    history = self.history_projection(history_features)
    # 将特征进行投影，从512->256
    if self.use_position and history.shape[1] > 0:
        positions = history_position_ids.clamp(min=0, max=self.max_history)
        # 约束一下位置index的范围，最大为最长历史长度，最小值为0
        history = history + self.position_embedding(positions)
        # 将位置编码添加到历史特征中

    null = self.null_history.expand(current.shape[0], -1, -1)
    # 将全0可学习张量进行扩展，从（1，1，256）扩展到(batch_size, 1, 256)
    history = torch.cat([null, history], dim=1)
    # 其和历史特征拼接，拼接后历史维度为[batch_size, max_history+1, 256]
    # 用于某些节点没有历史特征时，可以与null进行注意力计算
    null_mask = torch.zeros((current.shape[0], 1), dtype=torch.bool, device=current.device)
    # 创建一个掩码均值，维度为(batch_size, 1), 初始化的值为0
    key_padding_mask = torch.cat([null_mask, history_padding_mask], dim=1)
    #  将掩码进行拼接，拼接后维度为 [batch_size, max_history+1]
    context, attention_weights = self.attention(
        current.unsqueeze(1), history, history,
        key_padding_mask=key_padding_mask,
        need_weights=True,
        average_attn_weights=False,
    )
    # 进行注意力计算， context维度为: [batch_size, 1, 256]
    context = context.squeeze(1)
    # 去掉中间维度，维度变为[batch_size, 256]
    delta = self.delta_head(torch.cat([current, context], dim=-1))
    # 将current 和 context进行拼接，维度为[batch_size, 768]

    scale = torch.sigmoid(self.history_scale_logit)
    with torch.no_grad():
        baseline_logits = self.baseline(current_feature)
    logits = baseline_logits + scale * delta
    # 计算得到修正后的logits.
    return logits, {
        "baseline_logits": baseline_logits,
        "history_delta": delta,
        "history_scale": scale.detach(),
        "attention": attention_weights,
    }
```
整个模块要详细讲解一下，注意力模块和embedding模块。








