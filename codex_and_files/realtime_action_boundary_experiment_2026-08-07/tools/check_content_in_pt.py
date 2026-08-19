import torch

run_1_sample_data = torch.load(r"D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\realtime_action_boundary_experiment_2026-08-07\cache\features\A_as_test\all_runs\seed_1\stride_4\run_sample_000001.pt",
                               map_location="cpu")

for key, value in run_1_sample_data.items():
    if hasattr(value, "shape"):
        print(f"{key}: {value.shape}")
    elif isinstance(value, str):
        print(f"{key}: {value}")
    else:
        print(f"{key}: {len(value)}")

print(run_1_sample_data["state"][:10])
print(run_1_sample_data["start"][:10])
print(run_1_sample_data["end"][:10])
print(run_1_sample_data["timestamps"][:10])