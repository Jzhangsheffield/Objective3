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

## train_direct_history_model.py
```python
DIRECT_MODEL_NAMES = {
    "m1_direct": "direct_history_no_position",
    "m2_direct": "direct_actual_history",
    "m3_direct": "direct_graph_valid_shuffle",
}
```
建立不同模型索引。

```python
def build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_history_batch,
    )
```
构建dataloader

```python
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train isolated direct-head M1-M3 variants on frozen Tier-3 feature caches "
            "without an M0 checkpoint or logit delta"
        )
    )
    parser.add_argument("--model", required=True, choices=sorted(DIRECT_MODEL_NAMES))
    parser.add_argument("--train-scope", default="normal_only", choices=["normal_only", "all_runs"])
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--task-graph", required=True)
    parser.add_argument("--relation-matrix", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-history", type=int, default=35)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--action-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    # 初始化参数解析器，并设置默认参数。

    seed_everything(args.seed)
    device = select_device(args.device)
    graph = TaskGraphSpec.load(args.task_graph, args.relation_matrix)
    model_dir = ensure_new_output_dir(
        Path(args.output_root) / args.train_scope / args.model,
        overwrite=args.overwrite,
    )
    train_manifest = Path(args.protocol_root) / args.train_scope / "train.jsonl"
    history_order = "graph_valid" if args.model == "m3_direct" else "actual"
    # 确定基本路径，图，等其他配置

    train_dataset = FeatureHistoryDataset(
        args.train_cache,
        train_manifest,
        history_order=history_order,
        graph=graph,
        shuffle_seed=args.seed,
    )
    train_loader = build_loader(
        train_dataset, args.batch_size, args.num_workers, shuffle=True, device=device
    )
    # 构建数据集和dataloader
    model = build_direct_context_model(
        model_name=args.model,
        feature_dim=train_dataset.feature_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        max_history=args.max_history,
        dropout=args.dropout,
    ).to(device)
    # 构建模型
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    # 设置可训练参数
    train_log = train_feature_model(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        node_to_tier3=graph.node_to_tier3,
        epochs=args.epochs,
        action_loss_weight=args.action_loss_weight,
        amp=args.amp,
    )
    # 训练模型

    checkpoint_path = model_dir / "last.pth"
    # 设置最后一个epoch的保存路径
    feature_metadata = dict(train_dataset.cache.get("metadata", {}))
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        args.epochs,
        vars(args),
        extra={
            "experiment_family": "direct_head_fusion",
            "model_long_name": DIRECT_MODEL_NAMES[args.model],
            "visual_backbone_frozen_via_feature_cache": True,
            "feature_cache_metadata": feature_metadata,
            "train_log": train_log,
        },
    )
    # 保存模型
    write_json(model_dir / "train_log.json", train_log)
    write_json(
        model_dir / "experiment_config.json",
        {
            **vars(args),
            "experiment_family": "direct_head_fusion",
            "model_long_name": DIRECT_MODEL_NAMES[args.model],
            "history_order": history_order,
            "visual_backbone_frozen_via_feature_cache": True,
            "feature_cache_metadata": feature_metadata,
            "uses_m0_checkpoint": False,
            "uses_logit_delta": False,
            "node_head_initialization": "random",
        },
    )
    write_json(
        model_dir / "learned_parameters.json",
        {
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
            "fusion_parameters": sum(parameter.numel() for parameter in model.fusion.parameters()),
            "node_classifier_parameters": sum(
                parameter.numel() for parameter in model.node_classifier.parameters()
            ),
            "checkpoint": str(checkpoint_path),
        },
    )
    print(f"Saved final epoch checkpoint: {checkpoint_path}")
    # 保存训练过程数据

    # Test manifests/caches are consumed only after the final checkpoint is saved.
    test_result_root = ensure_dir(model_dir / "test_results")
    for split_name in ("test_normal", "test_fault", "test_all"):
        selection_manifest = Path(args.protocol_root) / args.train_scope / f"{split_name}.jsonl"
        test_dataset = FeatureHistoryDataset(
            args.test_cache,
            selection_manifest,
            history_order=history_order,
            graph=graph,
            shuffle_seed=args.seed,
        )
        test_loader = build_loader(
            test_dataset, args.batch_size, args.num_workers, shuffle=False, device=device
        )
        metrics = evaluate_feature_model(
            model, test_loader, device, graph.node_to_tier3, test_result_root, split_name
        )
        print(
            f"{split_name}: node_acc={metrics['node']['accuracy']:.4f} "
            f"tier3_acc={metrics['tier3']['accuracy']:.4f} "
            f"tier3_macro_f1={metrics['tier3']['macro_f1']:.4f}",
            flush=True,
        )
    write_json(
        model_dir / "completed.json",
        {
            "experiment_family": "direct_head_fusion",
            "model": args.model,
            "checkpoint": str(checkpoint_path),
            "train_scope": args.train_scope,
            "tested_splits": ["test_normal", "test_fault", "test_all"],
            "uses_m0_checkpoint": False,
            "uses_logit_delta": False,
        },
    )
    # 测试模型并保存测试结果和相关数据。


if __name__ == "__main__":
    main()
```
该函数不进行详细讲解了。

### extract_features()
```python 
@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract deterministic 512-D RGB features")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=-1, help="Debug only; -1 extracts all")
    parser.add_argument("--completion-marker", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Feature cache already exists: {output_path}. Refusing to overwrite it."
        )

    seed_everything(args.seed)
    device = select_device(args.device)
    dataset = RGBClipDataset(args.dataset_root, args.manifest, args.camera_id, train=False)
    if args.max_samples > 0:
        dataset.rows = dataset.rows[: args.max_samples]
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    # 创建dataset 和 dataloader
    model = generate_model(18, num_classes=NUM_TIER3_CLASSES).to(device)
    report = load_compatible_state(model, args.checkpoint)
    if "fc.weight" in report["missing_keys"] or "fc.bias" in report["missing_keys"]:
        raise RuntimeError(f"Tier-3 checkpoint classifier was not loaded: {report}")
    model.eval()
    # 创建模型并加载权重，并设置测试模式。
    features: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    records: list[dict] = []
    row_cursor = 0
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            batch_features = model.forward_features(video)
            batch_logits = model.forward_head(batch_features)
        features.append(batch_features.float().cpu())
        logits.append(batch_logits.float().cpu())
        count = int(video.shape[0])
        records.extend(dataset.rows[row_cursor:row_cursor + count])
        row_cursor += count
        print(f"extracted={row_cursor}/{len(dataset)}", flush=True)

    ensure_dir(output_path.parent)
    metadata = {
        "dataset_root": str(args.dataset_root),
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "camera_id": args.camera_id,
        "feature_dim": 512,
        "n_frames": 16,
        "rgb_size": 224,
        "load_report": report,
    }
    torch.save(
        {
            "features": torch.cat(features, dim=0),
            "tier3_logits": torch.cat(logits, dim=0),
            "records": records,
            "metadata": metadata,
        },
        output_path,
    )
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    if args.completion_marker:
        write_json(
            args.completion_marker,
            {
                "stage": "feature_extraction",
                "final_output": str(output_path),
                "checkpoint": str(args.checkpoint),
            },
        )
    print(f"Saved feature cache: {output_path}")


if __name__ == "__main__":
    main()
```
主要明确：保存下来features的维度为:[num_samples, features_dim]
而records是一个列表，每一个元素都是一个包含样本基本所有信息的row，其一一和num_samples按顺序对应。