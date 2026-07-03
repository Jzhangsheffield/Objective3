import argparse, warnings, random, time
import sys, os

import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision.io import read_video, write_video


class TemporallyConsistentSpatialAugmentation:
    """对一段视频统一应用一次随机增强
    """
    def __init__(
        self,
        size=(224, 224), # 最终尺寸
        crop_scale=(0.25, 1.0), # crop 后高和宽占原图的比例
        crop_ratio=(0.75, 1.3333), # crop 的高和宽的比例
        flip_p=0.5,
        vflip_p=0.5,
        jitter_p=0.5,
        jitter_brightness=0.24,
        jitter_contrast=0.24,
        jitter_saturation=0.24,
        jitter_hue=0.16,
        gray_p=0.2,
        blur_p=0.5,
        blur_kernel=7,                 # 奇数，>=3；你也可以改成 3/9/11 等
        blur_sigma=(0.1, 1.0),
        mean=(0.356, 0.363, 0.367),
        std=(0.288, 0.271, 0.270),
        ):
        
        self.mean = torch.tensor(mean).view(1, 1, 1, 3)
        self.std = torch.tensor(std).view(1, 1, 1, 3)
        
        self.transform = v2.Compose([
            # v2.Resize(size=size),
            v2.RandomResizedCrop(size=size, scale=crop_scale, ratio=crop_ratio, antialias=True),
            # 以下增强多用于 对比学习, 在训练 baseline 的 r3d 模型时,不需要使用.
            v2.RandomHorizontalFlip(p=flip_p),
            v2.RandomVerticalFlip(p=vflip_p),
            v2.RandomApply([v2.ColorJitter(
                brightness=jitter_brightness,
                contrast=jitter_contrast,
                saturation=jitter_saturation,
                hue=jitter_hue,
            )], p=jitter_p),
            v2.RandomGrayscale(p=gray_p), # 输出与输入的 channel 是一致的.
            v2.RandomApply([v2.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)], p=blur_p),
            # —— 下游一般需要 float + normalize ——
            v2.ToDtype(torch.float32, scale=True),         # [0,255] → [0,1] 并转 float32
            v2.Normalize(mean=mean, std=std),
        ])
        
    def denormalize(self, vid):
        # 将 transform 后的视频变回去，用于验证。
        thwc = vid.permute(0, 2, 3, 1)
        thwc = thwc * self.std + self.mean # 还原回归一化
        thwc = (thwc.clamp(0, 1) * 255).to(torch.uint8)

        return thwc
    
    
    def __call__(self, sample):
        if isinstance(sample, torch.Tensor):
            vid = tv_tensors.Video(sample)
        else:
            raise TypeError("请将 sample 转换成 torch.Tensor 形状 [T,H,W,C] 或 [T,C,H,W].")

        # 将维度转换为 [T,C,H,W]
        if vid.ndim == 4 and vid.shape[-1] in (1, 3):
            vid = vid.permute(0, 3, 1, 2)
            
        out = self.transform(vid)


        return out
    

class ValidationAugmentation:
    """对一段视频统一应用一次随机增强
    """
    def __init__(
        self,
        size=(224, 224), # 最终尺寸
        crop_scale=(0.25, 1.0), # crop 后高和宽占原图的比例
        crop_ratio=(0.75, 1.3333), # crop 的高和宽的比例
        flip_p=0.5,
        vflip_p=0.5,
        jitter_p=0.8,
        gray_p=0.2,
        blur_kernel=23,                 # 奇数，>=3；你也可以改成 3/9/11 等
        blur_sigma=(0.1, 2.0),
        mean=(0.356, 0.363, 0.367),
        std=(0.288, 0.271, 0.270),
        ):
        
        self.mean = torch.tensor(mean).view(1, 1, 1, 3)
        self.std = torch.tensor(std).view(1, 1, 1, 3)
        
        self.transform = v2.Compose([
            v2.Resize(size=size),
            # 以下增强多用于 对比学习, 在训练 baseline 的 r3d 模型时,不需要使用.
            # v2.RandomHorizontalFlip(p=flip_p),
            # v2.RandomVerticalFlip(p=vflip_p),
            # v2.RandomApply([v2.ColorJitter(
            #     brightness=0.8 * 0.3,   # 与你原代码一致
            #     contrast =0.8 * 0.3,
            #     saturation=0.8 * 0.3,
            #     hue=0.8 * 0.2
            # )], p=jitter_p),
            # v2.RandomGrayscale(p=gray_p), # 输出与输入的 channel 是一致的.
            # v2.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma),
            # # —— 下游一般需要 float + normalize ——
            v2.ToDtype(torch.float32, scale=True),         # [0,255] → [0,1] 并转 float32
            v2.Normalize(mean=mean, std=std),
        ])
        
    def denormalize(self, vid):
        # 将 transform 后的视频变回去，用于验证。
        thwc = vid.permute(0, 2, 3, 1)
        thwc = thwc * self.std + self.mean # 还原回归一化
        thwc = (thwc.clamp(0, 1) * 255).to(torch.uint8)

        return thwc
    
    
    def __call__(self, sample):
        if isinstance(sample, torch.Tensor):
            vid = tv_tensors.Video(sample)
        else:
            raise TypeError("请将 sample 转换成 torch.Tensor 形状 [T,H,W,C] 或 [T,C,H,W].")

        # 将维度转换为 [T,C,H,W]
        if vid.ndim == 4 and vid.shape[-1] in (1, 3):
            vid = vid.permute(0, 3, 1, 2)
            
        out = self.transform(vid)


        return out
    

class SwapMiddleFrame:
    """随机选中视频帧并将选中的随机帧乱序，
       越接近视频中间的帧，被选中的可能性越大。
    """
    def __init__(self,
        k: int,
        sigma_scale: float=0.2,
        rng=None,
        max_trials: int=5
        ):
        
        assert k >= 0, "k must be non-negative"
        assert 0 < sigma_scale <= 0.5, "sigma_scale should be in (0, 0.5]"
        self.k = k
        self.sigma_scale = sigma_scale
        self.rng = rng or random.Random()
        self.max_trials = max_trials
        
        
    def _center_weights(self, T: int) -> torch.Tensor:
        """根据 T 生成中心权重（中间大、两端小）：离散高斯权重。"""
        idx = torch.arange(T, dtype=torch.float32)
        c = (T - 1) / 2.0 # 控制对称中心
        sigma = max(self.sigma_scale * T, 1.0)  # 控制陡峭程度
        w = torch.exp(-0.5 * ((idx - c) / sigma) ** 2)  # shape [T]
        w = w / w.sum()
        # print(w)
        return w


    @staticmethod
    def _to_tensor_list(x):
        if isinstance(x, tv_tensors.Video):
            t = torch.as_tensor(x) 
            layout = "video_TCHW" if t.ndim==4 and t.shape[1] in (1,3) else "video_THWC" # video 表示是 video_tensor
            frames = [t[i] for i in range(t.shape[0])]
            return frames, layout
        if torch.is_tensor(x):
            t = x
            if t.ndim != 4:
                raise ValueError("Tensor video must be 4D: [T,C,H,W] or [T,H,W,C]")
            layout = "TCHW" if t.shape[1] in (1,3) else "THWC"
            frames = [t[i] for i in range(t.shape[0])]
            return frames, layout
        
    
    @staticmethod
    def _from_tensor_list(frames, layout: str):
        """把 List[frame] 还原为原布局的同类型输出（除非 list_unknown 则保持 list）。"""
        if layout in ("TCHW", "THWC", "video_TCHW", "video_THWC"):
            t = torch.stack(frames, dim=0)
            if layout in ("THWC", "video_THWC"):   # 保持 THWC
                return t
            # 需要转成 TCHW
            if t.ndim==4 and t.shape[-1] in (1,3):  # 当前为 THWC，需要转
                t = t.permute(0,3,1,2).contiguous()
            # 如果原本是 tv_tensors.Video，可再包回去；多数训练不必强制
            if layout.startswith("video_"):
                return tv_tensors.Video(t)
            return t
        # list_unknown: 保持 list
        return frames
        
    
    @staticmethod
    def _nontrivial_shuffle(indices: torch.Tensor, rng: random.Random, max_trials: int) -> torch.Tensor:
        """对选中的索引做随机置换，确保不是原顺序（若多次失败则接受原序）。"""
        if indices.numel() < 2:
            return indices
        base = indices.clone()
        for _ in range(max_trials):
            perm = torch.as_tensor(rng.sample(list(range(len(indices))), len(indices))) # rng.sample(sequence, k) 从 sequence 中随机选出 k 个元素 
            shuffled = indices[perm]
            if not torch.equal(shuffled, base):
                return shuffled
        return indices
    
    
    def __call__(self, video):
        frames, layout = self._to_tensor_list(video)
        T = len(frames)
        if T == 0 or self.k < 2:
            return self._from_tensor_list(frames, layout)
        k = min(self.k, T)

        # 1) 生成中心加权分布
        w = self._center_weights(T)

        # 2) 按权重无放回采样 k 个互换位置
        # chosen 为被选中的帧的索引
        #    使用 torch.multinomial 确保可微/与 torch 配合（尽管这里不求梯度）
        chosen = torch.multinomial(w, num_samples=k, replacement=False)  # [k]，升序/无序均可能
        # 为了操作稳定，按时间排序再做置换（便于读代码）
        chosen, _ = torch.sort(chosen) 

        # 3) 对 chosen 对应的帧做“非平凡置换”（不是原顺序）
        chosen_shuffled = self._nontrivial_shuffle(chosen, self.rng, self.max_trials)

        # 4) 应用置换
        new_frames = list(frames)  # 浅拷贝
        # 把被选中的帧提取出来，并按 shuffled 放回
        picked = [frames[i.item()] for i in chosen_shuffled] # 先把选中的帧给取出来
        for dst_pos, src_frame in zip(chosen.tolist(), picked):
            new_frames[dst_pos] = src_frame # 替换

        return self._from_tensor_list(new_frames, layout)
            
        
        
class TwoViewTransforms:
    """为同一个样本生成两个transform
    """
    def __init__(self, base_transforms):
        self.base_transforms = base_transforms

    def __call__(self, x):
        q = self.base_transforms(x)
        k = self.base_transforms(x)

        return [q, k]
        
        
    

# aug1 = TemporallyConsistentSpatialAugmentation()
# aug2 = SwapMiddleFrame(40, sigma_scale=0.4)

# # input_video, audio, info = read_video(f"F:\Objective2\VideoCL\MoCo\__gqSMtendc_000164_000174.mp4",
# #                                       output_format="THWC", pts_unit='sec')
# input_video, audio, info = read_video(r"F:\Objective2\VideoCL\MoCo\v_ApplyEyeMakeup_g01_c01.avi")
# print(input_video.size())
# # fps = info.get("video_fps", 25)

# output_video = aug1(input_video)
# vis = aug1.denormalize(output_video)
# vis = aug2(vis)

# write_video(filename=r"F:\Objective2\VideoCL\MoCo\v_ApplyEyeMakeup_g01_c01_aug.avi", video_array=vis, fps=25)
        
    

