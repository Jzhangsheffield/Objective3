## extract_features.py
    该文件则作用是从对应的训练集中抽取每个样本的特征。
### 主要输入：
    --dataset-root: 数据集路径
    --manifest：训练集样本jsonl文件
    --output: 抽取出的特征的保存路径
### 主要输出：
    提取出特征和一些元信息保存为.pt文件。
    .pt包含的内容有：
        "features": torch.Tensor, [样本数, 512], 每一个样本对应一个512维的特征
        "tier3_logits": torch.Tensor, [样本数, 31] 每一个样本对应一个31维的特征，其中的数值表示对应类别的预测概率。
        "records": list[dict[str, Any]], 每一个字典对应着manifest中一行
        "metadata": dict[str, Any], 里面包含着一些元信息，比如使用的哪个camera_id, 每个clip采样多少帧等。
