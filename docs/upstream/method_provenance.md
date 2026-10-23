# method_provenance — 跑出结果的那份方法代码，究竟是什么

对应 GitHub issue #1（也即 `agent_control/OPEN_ISSUES.md` 问题 7「版本锁定」）。

## 为什么需要这个目录

`methods/` 被 `.gitignore` 排除，方法代码不进本仓库的版本控制。`methods/` 下 12 个目录中，
**5 个自带 `.git`**（SpatialZ、isoST、UniST、NTF、DeepSpatial），**4 个没有**
（SINTER3D、Spa3D、ASIGN、Spateo）；另有 3 个 `*_adapter/` 是本项目自有的适配器目录，
非第三方方法仓库。PASTE-OT-Linear 与 ReHo 没有 `methods/` 目录——它们是本项目在
`benchmark/src/adapters/naive_baselines.py` 里定义的基线。更要紧的是：**登记在 `benchmark/METHOD_CHANGES.md` 的那些 math-neutral 补丁，
是以「未提交的工作区改动」的形式存在的**——所以只记一个 HEAD SHA 并不能确定实际跑的是哪份代码：

> 实际代码 = 方法仓库的 HEAD + 工作区未提交改动

此前这两部分都没有机器可读的载体（`METHOD_CHANGES.md` 里只有散文描述）。
本目录把它们固化下来。

## 内容

| 文件 | 内容 |
|---|---|
| `method_commits.tsv` | 每个方法的 `repo_head` / `branch` / `vcs` / patch 文件 / patch 的 sha256 前 16 位 |
| `patches/<method>.patch` | 该方法**工作区源码改动**的完整 diff（字节精确，见下方"CRLF 陷阱"） |
| `new_files/<method>.txt` | 工作区里**未跟踪的新源码文件**清单——单靠 patch 抓不到它们，重建时必须一并带上 |

`vcs=no_vcs` 的方法（SINTER3D、PASTE-OT-Linear、ReHo、Spa3D、ASIGN、Spateo）目录下
没有 `.git`，无从取得 commit 锚点；这些方法的版本只能靠获取日期与来源记录，本目录无法补足。

## 怎么用

```bash
R=/data/home/xiepeiyan/xiepeiyan/lab/baseline
# 1) 把某方法退回上游原状（先确认工作区没有你自己的未保存改动！）
git -C $R/methods/isoST apply --reverse $R/state/method_provenance/patches/isoST.patch
# 2) 重新施加本项目登记的补丁
git -C $R/methods/isoST apply          $R/state/method_provenance/patches/isoST.patch
# 3) 校验 patch 是否仍与当前工作区一致（能干净反向应用即一致）
git -C $R/methods/isoST apply --reverse --check $R/state/method_provenance/patches/isoST.patch
```

**已验证**（2026-09-16）：两个 patch 均可干净反向应用，且「反向 → 正向」往返后
文件与操作前**逐字节一致**。

## 两个必须知道的坑

1. **CRLF 陷阱。** `isoST/model/utils/modules_ODE.py` 是 CRLF 行尾。生成 patch 时若用
   任何会做 universal-newline 转换的读取方式（例如 Python `subprocess` 的 `text=True`），
   patch 里的上下文行会丢掉 `\r`，导致反向应用在 `@@ -142` 处失败。本目录的 patch 是以
   **字节**方式捕获并写出的。
2. **patch 不含新文件。** `isoST/model/utils/batched_ode.py` 与 UniST 的
   `external/InterpolAI/*` 都是未跟踪的新源码；`git diff` 看不到它们。
   重建时务必对照 `new_files/*.txt`。

## 仍未解决的部分

- **`large_scale_benchmark_rerun_20260820/` 没有任何 `run_manifest.json`**——
  那是六方法 Common-task 主结果（论文 Table 1）的所在地，目前没有机器可读的 provenance。
  有 manifest 的只有较新的三棵树（`multimissing_20260907`、`stress_20260831`、
  `new_methods_20260911`）。补写主战役的 provenance 记录仍是待办。
- `run_manifest` 有 **4 套并存的 schema**（18/19/20/21 字段），与模板声明的 26 字段
  几乎无交集，见 issue #2。
- `templates/run_manifest.json` 声明的 `method_commit` 字段从未被任何 runner 填充。
