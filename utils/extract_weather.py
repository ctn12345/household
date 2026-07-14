import pandas as pd

# =========================
# 1. 文件路径
# =========================
input_path = "/mnt/sdc/tnchen/matchine_learning/MENSQ_92_previous-1950-2024.csv"
output_path = "/mnt/sdc/tnchen/matchine_learning/MENSQ_92_200612_201011.csv"

# =========================
# 2. 读取 CSV
# Météo-France 这个数据一般是分号分隔
# =========================
df = pd.read_csv(
    input_path,
    sep=";",
    encoding="utf-8",
    low_memory=False
)

# 去除列名可能存在的空格
df.columns = df.columns.str.strip()

# =========================
# 3. 检查 AAAAMM 字段
# =========================
if "AAAAMM" not in df.columns:
    raise ValueError("文件中没有找到 AAAAMM 字段，请检查列名。")

# 转成数值型，防止读成字符串
df["AAAAMM"] = pd.to_numeric(df["AAAAMM"], errors="coerce")

# =========================
# 4. 筛选 2006年12月 到 2010年11月
# =========================
df_filtered = df[
    (df["AAAAMM"] >= 200612) &
    (df["AAAAMM"] <= 201011)
].copy()

# =========================
# 5. 保存结果
# =========================
df_filtered.to_csv(
    output_path,
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

# =========================
# 6. 打印检查信息
# =========================
print("原始数据行数:", len(df))
print("筛选后数据行数:", len(df_filtered))
print("最小月份:", int(df_filtered["AAAAMM"].min()))
print("最大月份:", int(df_filtered["AAAAMM"].max()))
print("保存路径:", output_path)

print("\n筛选后的月份数量:")
print(df_filtered["AAAAMM"].nunique())

print("\n前几行:")
print(df_filtered[["NUM_POSTE", "NOM_USUEL", "AAAAMM"]].head())

print("\n后几行:")
print(df_filtered[["NUM_POSTE", "NOM_USUEL", "AAAAMM"]].tail())