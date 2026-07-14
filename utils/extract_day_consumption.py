# import pandas as pd
# import numpy as np

# input_path = "/mnt/sdc/tnchen/matchine_learning/household_power_consumption.txt"
# output_path = "/mnt/sdc/tnchen/matchine_learning/household_power_daily.csv"

# df = pd.read_csv(
#     input_path,
#     sep=";",
#     na_values="?",
#     low_memory=False
# )

# df.columns = df.columns.str.strip()

# df["datetime"] = pd.to_datetime(
#     df["Date"] + " " + df["Time"],
#     format="%d/%m/%Y %H:%M:%S",
#     errors="coerce"
# )

# df = df.dropna(subset=["datetime"]).copy()
# df = df.sort_values("datetime").reset_index(drop=True)

# num_cols = [
#     "Global_active_power",
#     "Global_reactive_power",
#     "Voltage",
#     "Global_intensity",
#     "Sub_metering_1",
#     "Sub_metering_2",
#     "Sub_metering_3"
# ]

# for col in num_cols:
#     df[col] = pd.to_numeric(df[col], errors="coerce")

# df["date"] = df["datetime"].dt.date

# # =========================
# # 关键：删除不完整日期
# # =========================
# daily_count = df.groupby("date").size()

# complete_dates = daily_count[daily_count == 1440].index

# df = df[df["date"].isin(complete_dates)].copy()

# print("完整日期数量:", len(complete_dates))
# print("保留日期范围:", min(complete_dates), "到", max(complete_dates))

# # =========================
# # 缺失值处理
# # =========================
# df[num_cols] = df[num_cols].interpolate(method="linear")
# df[num_cols] = df[num_cols].ffill().bfill()

# # =========================
# # 计算剩余能耗
# # =========================
# df["global_active_energy_wh"] = df["Global_active_power"] * 1000 / 60

# df["sub_metering_remainder"] = (
#     df["global_active_energy_wh"]
#     - df["Sub_metering_1"]
#     - df["Sub_metering_2"]
#     - df["Sub_metering_3"]
# )

# df["sub_metering_remainder"] = df["sub_metering_remainder"].clip(lower=0)

# # =========================
# # 按天聚合
# # =========================
# daily = df.groupby("date").agg({
#     # 每日总有功电能 kWh
#     "Global_active_power": lambda x: x.sum() / 60,

#     # 每日无功功率累计，简单处理为日累计
#     "Global_reactive_power": lambda x: x.sum() / 60,

#     # 状态变量取均值
#     "Voltage": "mean",
#     "Global_intensity": "mean",

#     # 分表能耗 Wh，按天求和
#     "Sub_metering_1": "sum",
#     "Sub_metering_2": "sum",
#     "Sub_metering_3": "sum",

#     # 剩余能耗 Wh，按天求和
#     "sub_metering_remainder": "sum"
# }).reset_index()

# daily["date"] = pd.to_datetime(daily["date"])

# daily = daily.rename(columns={
#     "Global_active_power": "global_active_power_kwh",
#     "Global_reactive_power": "global_reactive_power_sum",
#     "Voltage": "voltage_mean",
#     "Global_intensity": "global_intensity_mean",
#     "Sub_metering_1": "sub_metering_1_wh",
#     "Sub_metering_2": "sub_metering_2_wh",
#     "Sub_metering_3": "sub_metering_3_wh",
#     "sub_metering_remainder": "sub_metering_remainder_wh"
# })

# daily["sub_metering_1_kwh"] = daily["sub_metering_1_wh"] / 1000
# daily["sub_metering_2_kwh"] = daily["sub_metering_2_wh"] / 1000
# daily["sub_metering_3_kwh"] = daily["sub_metering_3_wh"] / 1000
# daily["sub_metering_remainder_kwh"] = daily["sub_metering_remainder_wh"] / 1000

# daily.to_csv(output_path, index=False, encoding="utf-8-sig")

# print(daily.head())
# print(daily.tail())
# print("保存路径:", output_path)


import pandas as pd
import numpy as np

# =========================
# 路径设置
# =========================
input_path = "/mnt/sdc/tnchen/matchine_learning/household_power_consumption.txt"
output_path = "/mnt/sdc/tnchen/matchine_learning/household_power_daily_power_final.csv"

# =========================
# 读取原始数据
# =========================
df = pd.read_csv(
    input_path,
    sep=";",
    na_values="?",
    low_memory=False
)

df.columns = df.columns.str.strip()

# =========================
# 构造 datetime
# =========================
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

df = df.dropna(subset=["datetime"]).copy()
df = df.sort_values("datetime").reset_index(drop=True)

# =========================
# 数值列转换
# =========================
num_cols = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# 提取日期
# =========================
df["date"] = df["datetime"].dt.date

# =========================
# 删除不完整日期
322# =========================
daily_count = df.groupby("date").size()
complete_dates = daily_count[daily_count == 1440].index

df = df[df["date"].isin(complete_dates)].copy()

print("完整日期数量:", len(complete_dates))
print("保留日期范围:", min(complete_dates), "到", max(complete_dates))

# =========================
# 缺失值处理
# =========================
df[num_cols] = df[num_cols].interpolate(method="linear")
df[num_cols] = df[num_cols].ffill().bfill()

# =========================
# 计算剩余能耗
# 注意：
# Global_active_power 单位是 kW
# 每分钟电能 Wh = kW * 1000 / 60
# 这里只用于计算 sub_metering_remainder
# 不作为预测目标
# =========================
df["global_active_energy_wh"] = df["Global_active_power"] * 1000 / 60

df["sub_metering_remainder"] = (
    df["global_active_energy_wh"]
    - df["Sub_metering_1"]
    - df["Sub_metering_2"]
    - df["Sub_metering_3"]
)

df["sub_metering_remainder"] = df["sub_metering_remainder"].clip(lower=0)

# =========================
# 按天聚合
# 核心修改：
# Global_active_power 使用 mean，保留 kW 单位
# 不再使用 sum() / 60
# =========================
daily = df.groupby("date").agg({
    # 每日平均总有功功率，单位 kW
    "Global_active_power": "sum",

    # 每日平均无功功率
    "Global_reactive_power": "sum",

    # 每日平均电压
    "Voltage": "mean",

    # 每日平均电流
    "Global_intensity": "mean",

    # 三个子表原始单位是 Wh/min，按天求和得到 Wh/day
    "Sub_metering_1": "sum",
    "Sub_metering_2": "sum",
    "Sub_metering_3": "sum",

    # 剩余能耗 Wh/day
    "sub_metering_remainder": "sum"
}).reset_index()

daily["date"] = pd.to_datetime(daily["date"])

# =========================
# 重命名字段
# =========================
daily = daily.rename(columns={
    "Global_active_power": "global_active_power_sum_kw",
    "Global_reactive_power": "global_reactive_power_sum",
    "Voltage": "voltage_mean",
    "Global_intensity": "global_intensity_mean",
    "Sub_metering_1": "sub_metering_1_wh",
    "Sub_metering_2": "sub_metering_2_wh",
    "Sub_metering_3": "sub_metering_3_wh",
    "sub_metering_remainder": "sub_metering_remainder_wh"
})

# =========================
# 将子表日能耗 Wh/day 转换为日平均功率 kW
# 这样特征也可以和预测目标统一到功率尺度
# =========================
daily["sub_metering_1_mean_kw"] = daily["sub_metering_1_wh"] / 1000 / 24
daily["sub_metering_2_mean_kw"] = daily["sub_metering_2_wh"] / 1000 / 24
daily["sub_metering_3_mean_kw"] = daily["sub_metering_3_wh"] / 1000 / 24
daily["sub_metering_remainder_mean_kw"] = daily["sub_metering_remainder_wh"] / 1000 / 24

# =========================
# 添加时间特征
# =========================
daily["year"] = daily["date"].dt.year
daily["month"] = daily["date"].dt.month
daily["day"] = daily["date"].dt.day
daily["dayofweek"] = daily["date"].dt.dayofweek
daily["dayofyear"] = daily["date"].dt.dayofyear

# 是否周末
daily["is_weekend"] = daily["dayofweek"].isin([5, 6]).astype(int)

# =========================
# 调整列顺序
# =========================
daily = daily[
    [
        "date",

        # 预测目标：每日平均总有功功率 kW
        "global_active_power_sum_kw",

        # 其他功率/状态特征
        "global_reactive_power_sum",
        "voltage_mean",
        "global_intensity_mean",

        # 子表日能耗 Wh
        "sub_metering_1_wh",
        "sub_metering_2_wh",
        "sub_metering_3_wh",
        "sub_metering_remainder_wh",

        # 子表日平均功率 kW
        "sub_metering_1_mean_kw",
        "sub_metering_2_mean_kw",
        "sub_metering_3_mean_kw",
        "sub_metering_remainder_mean_kw",

        # 时间特征
        "year",
        "month",
        "day",
        "dayofweek",
        "dayofyear",
        "is_weekend"
    ]
]

# =========================
# 保存
# =========================
daily.to_csv(output_path, index=False, encoding="utf-8-sig")

print(daily.head())
print(daily.tail())
print("保存路径:", output_path)

# =========================
# 明确预测目标
# =========================
target_col = "global_active_power_mean_kw"
print("预测目标列:", target_col)
print("预测目标单位: kW")