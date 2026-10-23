from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import json
import time

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from SINTER3D import utils


ROOT = Path(
    "/mnt/nfs/linpeiran/data/AIgnite/002/baseline/"
    "SINTER3D/SINTER3D-master"
)

DLPFC_DIR = ROOT / "data/DLPFC"
REF_PATH = ROOT / "data/processed/DLPFC_scRNA_Control.h5ad"

OUT_DIR = ROOT / "results_DLPFC/preprocess"
FIG_DIR = ROOT / "results_DLPFC/figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SLICE_IDS = [151673, 151674, 151675, 151676]
SLICE_DIST_MICRON = [10.0, 300.0, 10.0]


def finite_matrix(x):
    if sp.issparse(x):
        values = x.data
    else:
        values = np.asarray(x)

    return bool(np.isfinite(values).all())


print("========== 1. 加载四张 DLPFC 切片 ==========")

adata_st_list_raw = []
slice_summary = {}

for sid in SLICE_IDS:
    path = DLPFC_DIR / f"DLPFC_{sid}.h5ad"

    if not path.exists():
        raise FileNotFoundError(path)

    adata = sc.read_h5ad(path)

    before = adata.n_obs

    # 官方教程删除 cluster 为字符串 nan 的 spot
    cluster_text = adata.obs["cluster"].astype(str).str.lower()
    keep = cluster_text != "nan"
    adata = adata[keep.to_numpy()].copy()

    after = adata.n_obs

    if "spatial" not in adata.obsm:
        raise KeyError(f"{sid} 缺少 obsm['spatial']")

    if not all(
        x in adata.obs.columns
        for x in ["array_row", "array_col", "cluster"]
    ):
        raise KeyError(f"{sid} 缺少必要 obs 字段")

    adata.obs["source_slice"] = str(sid)

    print(
        f"{sid}: 原始 {before} spots，"
        f"去除 nan 后 {after} spots，删除 {before-after}"
    )

    slice_summary[str(sid)] = {
        "n_obs_before": int(before),
        "n_obs_after": int(after),
        "n_vars": int(adata.n_vars),
    }

    adata_st_list_raw.append(adata)


print("\n========== 2. 绘制配准前坐标 ==========")

fig, ax = plt.subplots(figsize=(7, 7))

for sid, adata in zip(SLICE_IDS, adata_st_list_raw):
    xy = np.asarray(adata.obsm["spatial"])

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=2,
        alpha=0.45,
        label=str(sid),
    )

ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_title("DLPFC before ICP alignment")
ax.legend(markerscale=4)
ax.set_xlabel("x")
ax.set_ylabel("y")

fig.tight_layout()
fig.savefig(
    FIG_DIR / "DLPFC_before_alignment.png",
    dpi=200,
)
plt.close(fig)


print("\n========== 3. ICP 配准 ==========")

start = time.time()

adata_st_list = utils.align_spots(
    [adata.copy() for adata in adata_st_list_raw],
    method="icp",
    data_type="Visium",
    coor_key="spatial",
    plot=False,
)

alignment_seconds = time.time() - start

print("ICP 配准耗时秒：", round(alignment_seconds, 2))

for sid, adata in zip(SLICE_IDS, adata_st_list):
    if "spatial_aligned" not in adata.obsm:
        raise KeyError(
            f"{sid} 配准后缺少 obsm['spatial_aligned']"
        )

    aligned = np.asarray(adata.obsm["spatial_aligned"])

    if aligned.shape != (adata.n_obs, 2):
        raise ValueError(
            f"{sid} aligned shape 错误：{aligned.shape}"
        )

    if not np.isfinite(aligned).all():
        raise ValueError(f"{sid} 配准坐标存在 NaN/Inf")

    print(
        f"[PASS] {sid} spatial_aligned shape：",
        aligned.shape,
    )


print("\n========== 4. 绘制配准后坐标 ==========")

fig, ax = plt.subplots(figsize=(7, 7))

for sid, adata in zip(SLICE_IDS, adata_st_list):
    xy = np.asarray(adata.obsm["spatial_aligned"])

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=2,
        alpha=0.45,
        label=str(sid),
    )

ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_title("DLPFC after ICP alignment")
ax.legend(markerscale=4)
ax.set_xlabel("aligned x")
ax.set_ylabel("aligned y")

fig.tight_layout()
fig.savefig(
    FIG_DIR / "DLPFC_after_alignment.png",
    dpi=200,
)
plt.close(fig)


print("\n========== 5. 保存配准切片 ==========")

for sid, adata in zip(SLICE_IDS, adata_st_list):
    out = OUT_DIR / f"DLPFC_{sid}_aligned.h5ad"

    adata.write_h5ad(
        out,
        compression="lzf",
    )

    print("[SAVE]", out)


print("\n========== 6. 加载单细胞参考 ==========")

if not REF_PATH.exists():
    raise FileNotFoundError(REF_PATH)

adata_ref = sc.read_h5ad(REF_PATH)

print("adata_ref：", adata_ref)
print("obs columns：", list(adata_ref.obs.columns))
print("condition：")
print(adata_ref.obs["condition"].astype(str).value_counts())

required_ref_obs = ["celltype", "group", "condition"]

for col in required_ref_obs:
    if col not in adata_ref.obs:
        raise KeyError(f"单细胞参考缺少 obs['{col}']")

if "celltype_list_use" not in adata_ref.uns:
    raise KeyError(
        "单细胞参考 uns 中缺少 celltype_list_use"
    )

celltype_list_use = [
    str(x)
    for x in adata_ref.uns["celltype_list_use"]
]

print("使用细胞类型数量：", len(celltype_list_use))
print("使用细胞类型：")
for celltype in celltype_list_use:
    count = int(
        np.sum(
            adata_ref.obs["celltype"].astype(str)
            == celltype
        )
    )
    print(f"  {celltype:16s} {count:6d}")

missing = [
    x
    for x in celltype_list_use
    if x not in set(
        adata_ref.obs["celltype"].astype(str)
    )
]

if missing:
    raise ValueError(
        "参考数据缺少细胞类型：" + ", ".join(missing)
    )


print("\n========== 7. SINTER3D 预处理 ==========")

start = time.time()

adata_st, adata_basis = utils.preprocess(
    adata_st_list,
    adata_ref,
    celltype_ref_col="celltype",
    celltype_ref=celltype_list_use,
    sample_col="group",
    slice_dist_micron=SLICE_DIST_MICRON,
    n_hvg_group=500,
)

preprocess_seconds = time.time() - start

print(
    "SINTER3D 预处理耗时秒：",
    round(preprocess_seconds, 2),
)


print("\n========== 8. 预处理结果检查 ==========")

print("adata_st：", adata_st)
print("adata_basis：", adata_basis)

print("adata_st shape：", adata_st.shape)
print("adata_basis shape：", adata_basis.shape)
print("adata_st obs columns：", list(adata_st.obs.columns))
print("adata_st obsm keys：", list(adata_st.obsm.keys()))
print("adata_basis cell types：")
print(list(adata_basis.obs_names))

required_obsm = [
    "spatial",
    "spatial_aligned",
    "3D_coor",
    "count",
]

for key in required_obsm:
    if key not in adata_st.obsm:
        raise KeyError(f"adata_st 缺少 obsm['{key}']")

coords_3d = np.asarray(adata_st.obsm["3D_coor"])

print("3D_coor shape：", coords_3d.shape)
print(
    "3D_coor finite：",
    bool(np.isfinite(coords_3d).all()),
)

z_values = np.unique(
    np.round(coords_3d[:, 2], 8)
)

print("唯一 z 坐标：", z_values.tolist())
print("z 层数：", len(z_values))

print("每张切片 spot 数：")
print(
    adata_st.obs["slice"]
    .astype(int)
    .value_counts()
    .sort_index()
)

print("最终 HVG 数量：", adata_st.n_vars)
print("表达矩阵 finite：", finite_matrix(adata_st.X))
print(
    "原始 count finite：",
    finite_matrix(adata_st.obsm["count"]),
)
print(
    "basis finite：",
    finite_matrix(adata_basis.X),
)

if coords_3d.shape != (adata_st.n_obs, 3):
    raise ValueError("3D_coor 维度不正确")

if len(z_values) != 4:
    raise ValueError(
        f"预期 4 个 z 层，实际为 {len(z_values)}"
    )

if not np.isfinite(coords_3d).all():
    raise ValueError("3D 坐标存在 NaN/Inf")

if not finite_matrix(adata_st.X):
    raise ValueError("adata_st.X 存在 NaN/Inf")

if not finite_matrix(adata_basis.X):
    raise ValueError("adata_basis.X 存在 NaN/Inf")


print("\n========== 9. 保存预处理结果 ==========")

adata_st_path = OUT_DIR / "DLPFC_adata_st_preprocessed.h5ad"
adata_basis_path = OUT_DIR / "DLPFC_adata_basis.h5ad"

adata_st.write_h5ad(
    adata_st_path,
    compression="lzf",
)

adata_basis.write_h5ad(
    adata_basis_path,
    compression="lzf",
)

summary = {
    "slice_ids": SLICE_IDS,
    "slice_dist_micron": SLICE_DIST_MICRON,
    "slice_summary": slice_summary,
    "alignment_seconds": alignment_seconds,
    "preprocess_seconds": preprocess_seconds,
    "adata_st_shape": list(adata_st.shape),
    "adata_basis_shape": list(adata_basis.shape),
    "n_hvg": int(adata_st.n_vars),
    "n_celltypes": int(adata_basis.n_obs),
    "unique_z": [float(x) for x in z_values],
}

with (
    OUT_DIR / "preprocess_summary.json"
).open("w", encoding="utf-8") as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )

print("[SAVE]", adata_st_path)
print("[SAVE]", adata_basis_path)
print("[SAVE]", OUT_DIR / "preprocess_summary.json")

print(
    "\nPASS：DLPFC 配准和 SINTER3D 预处理全部完成"
)
