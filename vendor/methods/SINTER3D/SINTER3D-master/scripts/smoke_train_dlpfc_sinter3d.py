from pathlib import Path
import json
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import anndata as ad
import numpy as np
import torch

from SINTER3D.predict_selector import auto_select_and_train


ROOT = PROJECT_ROOT
PREPROCESS_DIR = ROOT / "results_DLPFC/preprocess"
OUT_DIR = ROOT / "results_DLPFC/smoke"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SLICE_IDS = [151673, 151674, 151675, 151676]

ADATA_ST_PATH = (
    PREPROCESS_DIR /
    "DLPFC_adata_st_preprocessed.h5ad"
)

ADATA_BASIS_PATH = (
    PREPROCESS_DIR /
    "DLPFC_adata_basis.h5ad"
)

CONFIG_PATH = (
    ROOT /
    "SINTER3D/config/DLPFC_73-76_smoke.json"
)

SELECTOR_PATH = (
    ROOT /
    "SINTER3D/config/model_selector.pkl"
)


print("========== 1. 环境检查 ==========")

print("Python：", sys.version.split()[0])
print("torch：", torch.__version__)
print("torch CUDA：", torch.version.cuda)
print("CUDA available：", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ERROR：当前无法使用 GPU")

print("GPU：", torch.cuda.get_device_name(0))

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


print("\n========== 2. 输入文件检查 ==========")

required_paths = [
    ADATA_ST_PATH,
    ADATA_BASIS_PATH,
    CONFIG_PATH,
    SELECTOR_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

    print("[PASS]", path)


print("\n========== 3. 加载预处理结果 ==========")

adata_st = ad.read_h5ad(ADATA_ST_PATH)
adata_basis = ad.read_h5ad(ADATA_BASIS_PATH)

print("adata_st：", adata_st)
print("adata_basis：", adata_basis)
print("adata_st obsm：", list(adata_st.obsm.keys()))
print("adata_st obs：", list(adata_st.obs.columns))

required_obsm = ["count", "3D_coor"]

for key in required_obsm:
    if key not in adata_st.obsm:
        raise KeyError(f"adata_st 缺少 obsm['{key}']")

if "slice" not in adata_st.obs:
    raise KeyError("adata_st.obs 缺少 slice")

if "library_size" not in adata_st.obs:
    raise KeyError("adata_st.obs 缺少 library_size")

print("切片数量：", adata_st.obs["slice"].nunique())
print("spot 总数：", adata_st.n_obs)
print("HVG 数量：", adata_st.n_vars)
print("细胞类型数量：", adata_basis.n_obs)


print("\n========== 4. 加载四张配准切片 ==========")

adata_st_list = []

for sid in SLICE_IDS:
    path = (
        PREPROCESS_DIR /
        f"DLPFC_{sid}_aligned.h5ad"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    obj = ad.read_h5ad(path)

    if "spatial_aligned" not in obj.obsm:
        raise KeyError(
            f"{sid} 缺少 obsm['spatial_aligned']"
        )

    adata_st_list.append(obj)

    print(
        f"{sid}: shape={obj.shape}, "
        f"aligned={obj.obsm['spatial_aligned'].shape}"
    )


print("\n========== 5. Smoke 配置 ==========")

config = json.loads(
    CONFIG_PATH.read_text(encoding="utf-8")
)

print(json.dumps(config, indent=2))

if int(config["training_steps"]) > 100:
    raise ValueError(
        "Smoke test 的 training_steps 不应超过100"
    )


print("\n========== 6. 自动选择模型并训练 ==========")

start = time.time()

model = auto_select_and_train(
    adata_st_list,
    adata_st,
    adata_basis,
    slice_idx=SLICE_IDS,
    selector_path=str(SELECTOR_PATH),
    config=str(CONFIG_PATH),
)

training_seconds = time.time() - start

selected_module = model.__class__.__module__
selected_class = model.__class__.__name__

print("\n训练完成")
print("模型模块：", selected_module)
print("模型类：", selected_class)
print("设备：", model.device)
print("训练步数：", model.net.training_steps)
print("训练耗时秒：", round(training_seconds, 2))

if model.device.type != "cuda":
    raise RuntimeError("模型没有使用 GPU")


print("\n========== 7. 训练后推理 ==========")

eval_start = time.time()

result = model.eval(
    adata_st_list,
    save=True,
    output_path=str(OUT_DIR),
)

eval_seconds = time.time() - eval_start

print("返回切片数量：", len(result))
print("推理耗时秒：", round(eval_seconds, 2))

if len(result) != len(SLICE_IDS):
    raise ValueError(
        f"预期返回4张切片，实际返回{len(result)}张"
    )


print("\n========== 8. 检查并保存结果 ==========")

celltypes = list(model.celltypes)

for sid, obj in zip(SLICE_IDS, result):
    missing = [
        celltype
        for celltype in celltypes
        if celltype not in obj.obs.columns
    ]

    if missing:
        raise KeyError(
            f"{sid} 缺少细胞比例字段：{missing}"
        )

    proportions = obj.obs[celltypes].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(proportions).all():
        raise ValueError(
            f"{sid} 的细胞类型比例包含 NaN/Inf"
        )

    if np.any(proportions < -1e-6):
        raise ValueError(
            f"{sid} 的细胞类型比例包含负值"
        )

    row_sums = proportions.sum(axis=1)

    print(
        f"{sid}: shape={obj.shape}, "
        f"比例和均值={row_sums.mean():.6f}, "
        f"范围=({row_sums.min():.6f}, "
        f"{row_sums.max():.6f})"
    )

    out_path = (
        OUT_DIR /
        f"DLPFC_{sid}_smoke_result.h5ad"
    )

    obj.write_h5ad(
        out_path,
        compression="lzf",
    )

    print("[SAVE]", out_path)


print("\n========== 9. 检查 latent ==========")

if "latent" not in model.adata_st.obsm:
    raise KeyError(
        "model.adata_st 中没有生成 latent"
    )

latent = np.asarray(
    model.adata_st.obsm["latent"]
)

print("latent shape：", latent.shape)
print("latent finite：", np.isfinite(latent).all())

if not np.isfinite(latent).all():
    raise ValueError("latent 中存在 NaN/Inf")

model.adata_st.write_h5ad(
    OUT_DIR / "DLPFC_smoke_adata_st_with_latent.h5ad",
    compression="lzf",
)


print("\n========== 10. 保存模型参数 ==========")

checkpoint = {
    "model_module": selected_module,
    "model_class": selected_class,
    "slice_ids": SLICE_IDS,
    "slice_remap": model.slice_remap,
    "celltypes": celltypes,
    "var_names": list(model.adata_st.var_names),
    "config": config,
    "model_state_dict": model.net.state_dict(),
    "optimizer_state_dict": model.optimizer.state_dict(),
}

torch.save(
    checkpoint,
    OUT_DIR / "DLPFC_smoke_model.pt",
)


peak_gpu_mb = (
    torch.cuda.max_memory_allocated() /
    1024 ** 2
)

summary = {
    "selected_model_module": selected_module,
    "selected_model_class": selected_class,
    "training_steps": int(model.net.training_steps),
    "training_seconds": training_seconds,
    "eval_seconds": eval_seconds,
    "peak_gpu_memory_mb": peak_gpu_mb,
    "adata_st_shape": list(model.adata_st.shape),
    "adata_basis_shape": list(model.adata_basis.shape),
    "latent_shape": list(latent.shape),
    "slice_ids": SLICE_IDS,
}

with (
    OUT_DIR / "smoke_summary.json"
).open("w", encoding="utf-8") as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )

print("峰值 GPU 显存 MB：", round(peak_gpu_mb, 2))
print("[SAVE]", OUT_DIR / "DLPFC_smoke_model.pt")
print("[SAVE]", OUT_DIR / "smoke_summary.json")

print(
    "\nPASS：SINTER3D 30步训练、推理和结果保存全部完成"
)
