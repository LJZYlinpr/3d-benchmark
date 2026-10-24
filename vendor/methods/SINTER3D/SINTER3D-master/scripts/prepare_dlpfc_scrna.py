from pathlib import Path
import os
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


ROOT = Path(
    "/mnt/nfs/linpeiran/data/AIgnite/002/baseline/"
    "SINTER3D/SINTER3D-master"
)

SCRNA = ROOT / "data/spatialLIBD/scRNA"
OUT_DIR = ROOT / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MTX_PATH = SCRNA / "GSE144136_GeneBarcodeMatrix_Annotated.mtx"
CELL_PATH = SCRNA / "GSE144136_CellNames.csv"
GENE_PATH = SCRNA / "GSE144136_GeneNames.csv"

OUT_PATH = OUT_DIR / "DLPFC_scRNA_Control.h5ad"
TMP_PATH = OUT_DIR / "DLPFC_scRNA_Control.h5ad.tmp"


CELLTYPE_LIST_USE = [
    "Astros_1",
    "Astros_2",
    "Astros_3",
    "Endo",
    "Micro/Macro",
    "Oligos_1",
    "Oligos_2",
    "Oligos_3",
    "Ex_1_L5_6",
    "Ex_2_L5",
    "Ex_3_L4_5",
    "Ex_4_L_6",
    "Ex_5_L5",
    "Ex_6_L4_6",
    "Ex_7_L4_6",
    "Ex_8_L5_6",
    "Ex_9_L5_6",
    "Ex_10_L2_4",
]


def parse_cell_label(label: str):
    """
    按官方 DLPFC 教程解析：

    Ex_10_L2_4.3_Control_B3_AAAC...
    celltype = Ex_10_L2_4
    group = 3
    condition = Control
    """
    dot_parts = label.split(".")

    if len(dot_parts) < 2:
        raise ValueError(f"无法解析细胞标签：{label}")

    celltype = dot_parts[0]
    suffix_parts = dot_parts[1].split("_")

    if len(suffix_parts) < 2:
        raise ValueError(f"无法解析标签后半部分：{label}")

    group = suffix_parts[0]
    condition = suffix_parts[1]

    return celltype, group, condition


print("========== 1. 读取细胞和基因注释 ==========")

meta = pd.read_csv(CELL_PATH, index_col=0)
genes = pd.read_csv(GENE_PATH, index_col=0)

if "x" not in meta.columns:
    raise ValueError("CellNames.csv 缺少 x 列")

if "x" not in genes.columns:
    raise ValueError("GeneNames.csv 缺少 x 列")

cell_ids = meta["x"].astype(str)
gene_names = genes["x"].astype(str)

parsed = [parse_cell_label(x) for x in cell_ids]

meta.index = cell_ids.values
meta.index.name = "cell_id"
meta["celltype"] = [x[0] for x in parsed]
meta["group"] = [x[1] for x in parsed]
meta["condition"] = [x[2] for x in parsed]

genes.index = gene_names.values
genes.index.name = "gene"

print("细胞数：", meta.shape[0])
print("基因数：", genes.shape[0])
print("condition：")
print(meta["condition"].value_counts())
print("group 数量：", meta["group"].nunique())
print("celltype 数量：", meta["celltype"].nunique())

print("\n主要细胞类型：")
print(meta["celltype"].value_counts().head(30))

if not meta.index.is_unique:
    raise ValueError("细胞名称不唯一")

if not genes.index.is_unique:
    raise ValueError("基因名称不唯一")

if "Control" not in set(meta["condition"]):
    raise ValueError("没有解析到 Control 细胞")


print("\n========== 2. 检查官方18类细胞 ==========")

celltype_counts = meta.loc[
    meta["condition"] == "Control",
    "celltype"
].value_counts()

missing_celltypes = [
    x for x in CELLTYPE_LIST_USE
    if x not in celltype_counts.index
]

for celltype in CELLTYPE_LIST_USE:
    count = int(celltype_counts.get(celltype, 0))
    print(f"{celltype:15s} {count:6d}")

if missing_celltypes:
    raise ValueError(
        "缺少官方需要的细胞类型：" + ", ".join(missing_celltypes)
    )

print("[PASS] 官方18类细胞均存在")


print("\n========== 3. 读取 Matrix Market ==========")

start = time.time()

matrix = scipy.io.mmread(MTX_PATH)

print("读取完成，耗时秒：", round(time.time() - start, 2))
print("原始类型：", type(matrix).__name__)
print("原始 shape：", matrix.shape)

if matrix.shape != (genes.shape[0], meta.shape[0]):
    raise ValueError(
        f"矩阵维度 {matrix.shape} 与 "
        f"gene/cell 注释 {(genes.shape[0], meta.shape[0])} 不匹配"
    )

if not sp.issparse(matrix):
    matrix = sp.coo_matrix(matrix)

matrix = matrix.tocsr()

print("非零元素：", matrix.nnz)


print("\n========== 4. 筛选 Control 细胞 ==========")

control_mask = (
    meta["condition"].astype(str).values == "Control"
)

control_meta = meta.loc[control_mask].copy()

# 原始矩阵为 gene × cell，先筛选列，再转成 cell × gene
matrix_control = matrix[:, control_mask]
X = matrix_control.T.tocsr().astype(np.float32)

del matrix
del matrix_control

print("Control细胞数：", control_meta.shape[0])
print("Control矩阵 shape：", X.shape)
print("Control矩阵非零元素：", X.nnz)
print("X dtype：", X.dtype)

if X.shape != (control_meta.shape[0], genes.shape[0]):
    raise ValueError("筛选后的矩阵维度不正确")

if X.data.size and not np.isfinite(X.data).all():
    raise ValueError("表达矩阵包含 NaN 或 Inf")

if X.data.size and np.any(X.data < 0):
    raise ValueError("原始表达矩阵出现负数")


print("\n========== 5. 构建 AnnData ==========")

adata_ref = ad.AnnData(
    X=X,
    obs=control_meta,
    var=genes.copy(),
)

adata_ref.uns["dataset"] = "GSE144136"
adata_ref.uns["condition"] = "Control"
adata_ref.uns["celltype_list_use"] = CELLTYPE_LIST_USE
adata_ref.uns["metadata_parsing"] = (
    "celltype=before '.', "
    "group=first field after '.', "
    "condition=second field after '.'"
)

print(adata_ref)
print("obs columns：", list(adata_ref.obs.columns))
print("var columns：", list(adata_ref.var.columns))


print("\n========== 6. 写入 h5ad ==========")

if TMP_PATH.exists():
    TMP_PATH.unlink()

adata_ref.write_h5ad(
    TMP_PATH,
    compression="gzip",
)

os.replace(TMP_PATH, OUT_PATH)

print("输出文件：", OUT_PATH)
print(
    "输出大小 GB：",
    round(OUT_PATH.stat().st_size / 1024**3, 3)
)


print("\n========== 7. 回读验证 ==========")

check = ad.read_h5ad(OUT_PATH, backed="r")

print("回读 shape：", check.shape)
print("回读 obs：", list(check.obs.columns))
print("回读 var：", list(check.var.columns))

assert check.shape == adata_ref.shape
assert "celltype" in check.obs.columns
assert "group" in check.obs.columns
assert "condition" in check.obs.columns
assert set(check.obs["condition"].astype(str)) == {"Control"}

check.file.close()

print("\nPASS：DLPFC单细胞Control参考数据生成完成")
