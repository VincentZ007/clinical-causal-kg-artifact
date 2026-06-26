# MIMIC-III 外部审计：HPC 中文操作手册

## 这次实验要证明什么

这次不是重跑完整的 QA benchmark，而是在第二个 EHR 数据集上复现核心审计结论。我们会从
50,000 份 MIMIC-III 出院小结中抽取因果边，并检查：

- lift 是否仍可作为“边存在性”的支持信号；
- 诊断编码的时间顺序是否仍不可靠地反映因果方向；
- SemMedDB 是否仍是独立但覆盖有限的方向参考。

因此，论文中最多只能写“外部数据集上的审计结论得到复现”。没有单独构造并评测
MIMIC-III QA benchmark 前，不能写成“跨数据集 QA 泛化”。

## 第 1 步：用 MobaXterm 连接 HPC

1. 打开 MobaXterm，点击左上角 `Session`。
2. 选择 `SSH`。
3. 填学校提供的 HPC 登录地址和你的用户名。
4. 如学校要求密钥认证，勾选 `Use private key` 并选择私钥；否则使用学校账号密码。
5. 点击 `OK` 连接。首次连接时接受主机指纹即可。

连接成功后，在右侧终端输入：

```bash
mkdir -p $HOME/projects/ickg-mimic3-audit
cd $HOME/projects/ickg-mimic3-audit
pwd
```

最后一行必须显示你自己的 home 目录下的路径。不要在其他同学的目录里运行。

## 第 2 步：只上传代码，不上传患者数据

MobaXterm 左侧会显示远端目录（SFTP 面板）。把以下文件从本地项目目录上传到刚创建的
`ickg-mimic3-audit` 目录：

```text
prepare_mimic3_input.py
extract_llm_full.py
parse_llm_triples.py
link_umls.py
validate_edges.py
temporal_direction.py
semmeddb_ground_truth.py
direction_sensitivity.py
summarize_audit.py
extract_full.sbatch
```

上传后，在远端终端检查：

```bash
for f in prepare_mimic3_input.py extract_llm_full.py parse_llm_triples.py link_umls.py \
  validate_edges.py temporal_direction.py semmeddb_ground_truth.py direction_sensitivity.py \
  summarize_audit.py extract_full.sbatch
do
  test -s "$f" && echo "OK   $f" || echo "MISS $f"
done
```

只要出现任意一个 `MISS`，就先不要继续。MIMIC 原始文本和患者数据必须留在有权限的 HPC
存储位置，绝对不要通过 MobaXterm 上传到个人电脑或公共网盘。

## 第 3 步：确认数据路径和 Python 环境

你需要知道以下三项在 HPC 上的绝对路径：

1. `NOTEEVENTS.csv.gz` 的路径；
2. 包含 `ADMISSIONS.csv.gz`、`DIAGNOSES_ICD.csv.gz`、`D_ICD_DIAGNOSES.csv.gz` 的目录；
3. `semmedVER43_R_PREDICATION.csv.gz` 的路径。

把下面三行单引号中的路径替换为真实路径，再整段粘贴运行：

```bash
export MIMIC3_NOTES='/absolute/path/to/NOTEEVENTS.csv.gz'
export MIMIC3_HOSP='/absolute/path/to/mimic-iii-clinical-database-1.4'
export SEMMED='/absolute/path/to/semmedVER43_R_PREDICATION.csv.gz'
source /sw/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate test
test -r "$MIMIC3_NOTES" && echo 'notes OK'
test -r "$MIMIC3_HOSP/ADMISSIONS.csv.gz" && echo 'admissions OK'
test -r "$MIMIC3_HOSP/DIAGNOSES_ICD.csv.gz" && echo 'diagnoses OK'
test -r "$MIMIC3_HOSP/D_ICD_DIAGNOSES.csv.gz" && echo 'dictionary OK'
test -r "$SEMMED" && echo 'SemMedDB OK'
python -c "import pandas, scispacy; print('Python packages OK')"
```

六项检查都必须成功。然后确认 GPU 节点可取得 `Qwen/Qwen2.5-7B-Instruct-AWQ`；若模型没有
缓存或节点禁止下载，先咨询 HPC 管理员，暂时不要开始 50k 任务。

## 第 4 步：先构造 200 条的小样本

```bash
mkdir -p smoke_200 run_50000
python prepare_mimic3_input.py --notes "$MIMIC3_NOTES" \
  --out smoke_200/input.jsonl.gz --limit 200 --seed 2026
gzip -cd smoke_200/input.jsonl.gz | wc -l
gzip -cd smoke_200/input.jsonl.gz | head -1
```

输出行数必须大于 0。第一行应该有 `note_id` 和 `text` 两个字段，且 `text` 不是空字符串。
这里的随机种子 `2026` 不要改，它是可复现性记录的一部分。

## 第 5 步：提交 200 条 GPU 冒烟测试

```bash
INPUT=smoke_200/input.jsonl.gz OUTPUT=smoke_200/triples_llm.jsonl sbatch extract_full.sbatch
squeue -u "$USER"
```

等待任务结束后检查：

```bash
test -s smoke_200/triples_llm.jsonl && echo 'extractor output OK'
wc -l smoke_200/triples_llm.jsonl
tail -20 causal-llm-*.out
```

输入和输出的行数应相同，日志末尾应出现 `ALL DONE`。如果报显存不足，把
`extract_llm_full.py` 中的 `gpu_memory_utilization` 从 `0.90` 调为 `0.85`，然后只重跑这
个 200 条测试，确认无误后再继续。

## 第 6 步：构造固定的 50,000 条样本

```bash
python prepare_mimic3_input.py --notes "$MIMIC3_NOTES" \
  --out run_50000/input.jsonl.gz --limit 50000 --seed 2026
gzip -cd run_50000/input.jsonl.gz | wc -l
```

记录脚本打印的四个数字：符合条件的总数、抽样数、可用数、因没有目标章节而跳过的数。

## 第 7 步：提交 50k GPU 抽取任务

```bash
INPUT=run_50000/input.jsonl.gz OUTPUT=run_50000/triples_llm.jsonl sbatch extract_full.sbatch
squeue -u "$USER"
```

抽取脚本支持断点续跑。任务因时间限制中断时，重新提交完全相同的命令即可；它会跳过已经
写入 `triples_llm.jsonl` 的 `note_id`。不要删除已有的部分输出。

结束后检查：

```bash
gzip -cd run_50000/input.jsonl.gz | wc -l
wc -l run_50000/triples_llm.jsonl
tail -20 causal-llm-*.out
```

两项行数必须一致。若不一致，先用相同命令续跑一次，再检查日志后才继续。

## 第 8 步：解析、UMLS 链接和 EHR 审计

以下命令需要在有 UMLS/scispaCy 的环境中运行。若 HPC 不允许登录节点做较长 CPU 计算，
应使用学校规定的 CPU 队列提交这些命令。

```bash
python parse_llm_triples.py run_50000/triples_llm.jsonl \
  --out-triples run_50000/llm_triples.jsonl \
  --out-edges run_50000/edges_sectioned_llm.tsv

python link_umls.py --edges run_50000/edges_sectioned_llm.tsv \
  --map-out run_50000/phrase2cui.tsv \
  --cui-edges run_50000/edges_cui.tsv

python validate_edges.py --hosp "$MIMIC3_HOSP" --kg run_50000/edges_cui.tsv \
  --icd2cui run_50000/icd2cui.tsv --out run_50000/edges_cui_validated.tsv

python temporal_direction.py --hosp "$MIMIC3_HOSP" \
  --kg run_50000/edges_cui_validated.tsv --icd2cui run_50000/icd2cui.tsv \
  --out run_50000/edges_final.tsv
```

检查产物是否齐全：

```bash
for f in run_50000/llm_triples.jsonl run_50000/edges_sectioned_llm.tsv \
  run_50000/phrase2cui.tsv run_50000/edges_cui.tsv run_50000/icd2cui.tsv \
  run_50000/edges_cui_validated.tsv run_50000/edges_final.tsv
do
  test -s "$f" && echo "OK   $f" || echo "MISS $f"
done
```

## 第 9 步：运行独立文献方向审计

```bash
python semmeddb_ground_truth.py "$SEMMED" --kg run_50000/edges_final.tsv \
  --out run_50000/semmeddb_causal.tsv \
  --vs-out run_50000/semmeddb_vs_temporal.tsv

python direction_sensitivity.py --src run_50000/semmeddb_vs_temporal.tsv \
  --out run_50000/direction_sensitivity_results.json

python summarize_audit.py --dataset MIMIC-III-50k \
  --lift run_50000/edges_cui_validated.tsv \
  --temporal run_50000/edges_final.tsv \
  --semmed run_50000/semmeddb_vs_temporal.tsv \
  --out run_50000/audit_summary.json
```

最后一个命令会把主要统计量输出到屏幕，并写入 `run_50000/audit_summary.json`。

## 第 10 步：结果是否可以写进论文

只有同时满足以下条件，才能把它写成外部数据集验证：

1. 与 MIMIC-IV 使用相同 prompt、Qwen checkpoint、temperature、UMLS 阈值、lift 规则、
   temporal 阈值和 SemMedDB 版本；
2. 同时报告 temporal 的覆盖率、temporal 方向准确率、以及 LLM 在**同一 committed subset**
   上的方向准确率；
3. 报告 SemMedDB 覆盖率和不同 PMID 阈值下的敏感性结果；
4. 报告所有样本量和真实结果，即使结果与 MIMIC-IV 不一致。

不需要数值完全相同。可以支持论文主张的是“定性复现”：在它确实作出方向判断的边中，诊断
编码顺序仍明显弱于抽取器/文献参考。若 temporal 表现相近或更好，必须如实报告这种异质性，
并弱化论文中的普遍性表述。

## 第 11 步：保存可复现性记录

保留 Slurm 日志、`audit_summary.json`、随机种子、输入输出的 checksum。不要发布 MIMIC
原始文本或任何患者级标识；仅发布允许公开的派生产物，以及让有权限研究者复现实验的代码和
配置。
