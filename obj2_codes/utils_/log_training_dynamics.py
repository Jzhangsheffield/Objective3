# utils/training_dynamics.py
import csv, os
from collections import defaultdict

# class TrainingDynamicsLogger:
#     """
#     按 epoch 累积 (uid, epoch, gold, prob_true, pred, loss) 并写盘；
#     训练结束后可以再做汇总。
#     """
#     def __init__(self, save_dir):
#         self.save_dir = save_dir
#         os.makedirs(self.save_dir, exist_ok=True)
#         self.buffer = []  # 当前 epoch 的缓存

#     def log_minibatch(self, uids, golds, logits, probs_true, preds, losses, epoch):
#         # 所有参数均为 CPU 张量或 list
#         for uid, g, l, p, pr, ls in zip(uids, golds, logits, probs_true, preds, losses):
#             self.buffer.append({
#                 'uid': uid,
#                 'epoch': int(epoch),
#                 'gold': int(g),
#                 'logits': l,
#                 'prob_true': float(p),
#                 'pred': int(pr),
#                 'loss': float(ls),
#             })

#     def flush_epoch(self, epoch):
#         # 逐 epoch 写一个文件，便于中断续训也不丢
#         out_csv = os.path.join(self.save_dir, f"td_epoch_{int(epoch):03d}.csv")
#         if not self.buffer:
#             return
#         with open(out_csv, 'w', newline='') as f:
#             w = csv.DictWriter(f, fieldnames=['uid','epoch','gold','logits', 'prob_true','pred','loss'])
#             w.writeheader()
#             w.writerows(self.buffer)
#         self.buffer.clear()


class TrainingDynamicsLogger:
    """
    按 epoch 累积 (uid, epoch, gold, prob_true, pred) 并写盘；
    训练结束后可以再做汇总。
    此版本没有 losses
    """
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.buffer = []  # 当前 epoch 的缓存

    def log_minibatch(self, uids, golds, logits, probs_true, preds, epoch):
        # 所有参数均为 CPU 张量或 list
        for uid, g, l, p, pr in zip(uids, golds, logits, probs_true, preds):
            self.buffer.append({
                'uid': uid,
                'epoch': int(epoch),
                'gold': int(g),
                'logits': l,
                'prob_true': float(p),
                'pred': int(pr),
            })

    def flush_epoch(self, epoch):
        # 逐 epoch 写一个文件，便于中断续训也不丢
        out_csv = os.path.join(self.save_dir, f"td_epoch_{int(epoch):03d}.csv")
        if not self.buffer:
            return
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['uid','epoch','gold','logits', 'prob_true','pred'])
            w.writeheader()
            w.writerows(self.buffer)
        self.buffer.clear()