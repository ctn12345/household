import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 路径配置
# ============================================================
ROOT = "/mnt/sdc/tnchen/matchine_learning"

POWER_PATH = os.path.join(ROOT, "household_power_daily_power_final.csv")
WEATHER_PATH = os.path.join(ROOT, "MENSQ_92_200612_201011.csv")

DATA_DIR = os.path.join(ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
EVAL_DIR = os.path.join(DATA_DIR, "eval")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


# ============================================================
# 2. 参数配置
# ============================================================
TRAIN_RATIO = 0.7

INPUT_LEN = 90

TASKS = [
    {
        "name": "90to90",
        "input_len": 90,
        "pred_len": 90,
    },
    {
        "name": "90to365",
        "input_len": 90,
        "pred_len": 365,
    }
]

# 课程任务中明确给出的天气变量
RAW_WEATHER_COLS = [
    "RR",
    "NBJRR1",
    "NBJRR5",
    "NBJRR10",
    "NBJBROU"
]


# ============================================================
# 3. 工具函数
# ============================================================
def to_numeric_safe(series):
    """
    兼容法国气象数据中可能出现的小数逗号。
    """
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )


def build_windows_with_dates(df, feature_cols, target_col, input_len, pred_len):
    """
    构造滑动窗口。

    X: [num_samples, input_len, feature_dim]
    y: [num_samples, pred_len]
    y_dates: [num_samples, pred_len]
    pred_start_dates: [num_samples]
    """
    X = []
    y = []
    y_dates = []
    pred_start_dates = []

    features = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    dates = pd.to_datetime(df["date"]).values

    max_start = len(df) - input_len - pred_len + 1

    for i in range(max_start):
        x_i = features[i: i + input_len]
        y_i = target[i + input_len: i + input_len + pred_len]
        d_i = dates[i + input_len: i + input_len + pred_len]

        X.append(x_i)
        y.append(y_i)
        y_dates.append(d_i.astype("datetime64[D]").astype(str))
        pred_start_dates.append(str(d_i[0].astype("datetime64[D]")))

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(y_dates),
        np.asarray(pred_start_dates),
    )


def save_npz(path, X, y, y_dates, pred_start_dates, feature_cols, target_col):
    np.savez_compressed(
        path,
        X=X,
        y=y,
        y_dates=y_dates,
        pred_start_dates=pred_start_dates,
        feature_cols=np.asarray(feature_cols),
        target_col=np.asarray([target_col]),
    )


# ============================================================
# 4. 读取日级电力数据
# ============================================================
power = pd.read_csv(POWER_PATH, low_memory=False)
power.columns = power.columns.str.strip()

if "date" not in power.columns:
    raise ValueError("household_power_daily.csv 中没有 date 字段，请检查列名。")

power["date"] = pd.to_datetime(power["date"])
power = power.sort_values("date").reset_index(drop=True)

print("=" * 80)
print("电力数据读取完成")
print("电力数据行数:", len(power))
print("电力数据日期范围:", power["date"].min(), "到", power["date"].max())


# ============================================================
# 5. 读取天气数据
# ============================================================
weather = pd.read_csv(
    WEATHER_PATH,
    sep=";",
    encoding="utf-8",
    low_memory=False
)

weather.columns = weather.columns.str.strip()

if "AAAAMM" not in weather.columns:
    raise ValueError("天气数据中没有 AAAAMM 字段，请检查列名。")

weather["AAAAMM"] = pd.to_numeric(weather["AAAAMM"], errors="coerce")

# 只保留 2006-12 到 2010-11
weather = weather[
    (weather["AAAAMM"] >= 200612) &
    (weather["AAAAMM"] <= 201011)
].copy()

weather["month"] = pd.to_datetime(
    weather["AAAAMM"].astype(int).astype(str),
    format="%Y%m"
)

# 只取文件中真实存在的天气列
weather_cols = [c for c in RAW_WEATHER_COLS if c in weather.columns]

if len(weather_cols) == 0:
    raise ValueError("天气文件中没有找到 RR / NBJRR1 / NBJRR5 / NBJRR10 / NBJBROU 等字段。")

for col in weather_cols:
    weather[col] = to_numeric_safe(weather[col])

# RR 是十分之一毫米，转成毫米
if "RR" in weather_cols:
    weather["RR"] = weather["RR"] / 10.0


# ============================================================
# 6. 按月聚合天气
# 如果同一个月有多个气象站，按月求平均
# ============================================================
weather_monthly_raw = (
    weather
    .groupby("month", as_index=False)[weather_cols]
    .mean()
)

print("=" * 80)
print("原始天气月度数据")
print("天气月份数量:", len(weather_monthly_raw))
print("天气月份范围:", weather_monthly_raw["month"].min(), "到", weather_monthly_raw["month"].max())


# ============================================================
# 7. 构造完整月份序列，处理月份缺失
# ============================================================
start_month = power["date"].min().to_period("M").to_timestamp()
end_month = power["date"].max().to_period("M").to_timestamp()

full_months = pd.date_range(
    start=start_month,
    end=end_month,
    freq="MS"
)

weather_monthly = (
    weather_monthly_raw
    .set_index("month")
    .reindex(full_months)
)

weather_monthly.index.name = "month"

# 记录补齐前哪些月份缺失
missing_mask = weather_monthly[weather_cols].isna().any(axis=1)
missing_months = weather_monthly.loc[missing_mask].copy()

missing_report_path = os.path.join(DATA_DIR, "weather_missing_months_before_fill.csv")
missing_months.reset_index().to_csv(
    missing_report_path,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 80)
print("天气缺失月份检查")
print("完整月份数量:", len(full_months))
print("补齐前缺失月份数量:", int(missing_mask.sum()))
print("缺失月份报告已保存:", missing_report_path)

if missing_mask.sum() > 0:
    print("缺失月份如下:")
    print(missing_months.index.strftime("%Y-%m").tolist())


# ============================================================
# 8. 删除全空天气变量，然后对缺失月份插值补齐
# ============================================================
valid_weather_cols = []

for col in weather_cols:
    non_nan_count = weather_monthly[col].notna().sum()

    if non_nan_count > 0:
        valid_weather_cols.append(col)
    else:
        print(f"警告：天气变量 {col} 全部为空，已从特征中删除。")

weather_cols = valid_weather_cols

if len(weather_cols) == 0:
    raise ValueError("所有天气变量都是空值，无法合并天气数据。")

# 线性插值 + 前后填充
weather_monthly[weather_cols] = (
    weather_monthly[weather_cols]
    .interpolate(method="linear", limit_direction="both")
    .ffill()
    .bfill()
)

# 降雨天数、雾天数理论上是整数，插值后四舍五入
count_weather_cols = ["NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]

for col in count_weather_cols:
    if col in weather_monthly.columns:
        weather_monthly[col] = weather_monthly[col].round()

# 避免出现负数
for col in weather_cols:
    weather_monthly[col] = weather_monthly[col].clip(lower=0)

weather_monthly = weather_monthly.reset_index()

filled_weather_path = os.path.join(DATA_DIR, "weather_monthly_filled.csv")
weather_monthly.to_csv(
    filled_weather_path,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 80)
print("天气缺失补齐完成")
print("补齐后天气缺失数量:")
print(weather_monthly[weather_cols].isna().sum())
print("补齐后的天气数据已保存:", filled_weather_path)


# ============================================================
# 9. 合并日级电力数据和月度天气数据
# ============================================================
power["month"] = power["date"].dt.to_period("M").dt.to_timestamp()

data = power.merge(
    weather_monthly,
    on="month",
    how="left"
)

# 理论上不会再缺失，但保险起见再补一次
for col in weather_cols:
    data[col] = (
        data[col]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

# 删除辅助月份列
data = data.drop(columns=["month"])

# 添加时间特征
data["dayofweek"] = data["date"].dt.dayofweek
data["month_id"] = data["date"].dt.month
data["dayofyear"] = data["date"].dt.dayofyear

data["month_sin"] = np.sin(2 * np.pi * data["month_id"] / 12)
data["month_cos"] = np.cos(2 * np.pi * data["month_id"] / 12)
data["dow_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7)
data["dow_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7)

merged_path = os.path.join(ROOT, "household_power_daily_with_weather.csv")
data.to_csv(
    merged_path,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 80)
print("电力 + 天气合并完成")
print("合并后数据行数:", len(data))
print("合并后数据保存路径:", merged_path)
print("合并后天气缺失检查:")
print(data[weather_cols].isna().sum())


# ============================================================
# 10. 确定预测目标列
# ============================================================
# ============================================================
# 10. 确定预测目标列
# 现在预测的是每日平均总有功功率，单位 kW
# 不能再使用 kWh
# ============================================================
if "global_active_power_sum_kw" in data.columns:
    target_col = "global_active_power_sum_kw"
elif "Global_active_power" in data.columns:
    target_col = "Global_active_power"
else:
    raise ValueError(
        "找不到预测目标列，请检查是否有 global_active_power_mean_kw 或 Global_active_power。"
    )

print("=" * 80)
print("预测目标列:", target_col)
print("预测目标单位: kW")

print("=" * 80)
print("预测目标列:", target_col)


# ============================================================
# 11. 构造输入特征列
# ============================================================
exclude_cols = {"date"}

feature_cols = [
    c for c in data.columns
    if c not in exclude_cols and pd.api.types.is_numeric_dtype(data[c])
]

# 确保历史目标值也作为输入特征
if target_col not in feature_cols:
    feature_cols.insert(0, target_col)

print("输入特征数量:", len(feature_cols))
print("输入特征列:")
for c in feature_cols:
    print("  -", c)


# ============================================================
# 12. 按时间顺序划分 train / eval
# ============================================================
data = data.sort_values("date").reset_index(drop=True)

split_idx = int(len(data) * TRAIN_RATIO)

train_df = data.iloc[:split_idx].copy()
eval_df_plain = data.iloc[split_idx:].copy()

# eval 需要额外带上前 90 天历史作为输入窗口
eval_start_idx = max(0, split_idx - INPUT_LEN)
eval_df_with_history = data.iloc[eval_start_idx:].copy()

print("=" * 80)
print("时间顺序划分完成")
print("总天数:", len(data))
print("训练集天数:", len(train_df))
print("验证集天数:", len(eval_df_plain))
print("训练集日期:", train_df["date"].min(), "到", train_df["date"].max())
print("验证集日期:", eval_df_plain["date"].min(), "到", eval_df_plain["date"].max())
print("带历史验证集日期:", eval_df_with_history["date"].min(), "到", eval_df_with_history["date"].max())


# ============================================================
# 13. 保存 train/eval 原始 CSV
# ============================================================
train_daily_path = os.path.join(TRAIN_DIR, "train_daily.csv")
eval_daily_path = os.path.join(EVAL_DIR, "eval_daily.csv")
eval_history_daily_path = os.path.join(EVAL_DIR, "eval_daily_with_90day_history.csv")

train_df.to_csv(train_daily_path, index=False, encoding="utf-8-sig")
eval_df_plain.to_csv(eval_daily_path, index=False, encoding="utf-8-sig")
eval_df_with_history.to_csv(eval_history_daily_path, index=False, encoding="utf-8-sig")

print("=" * 80)
print("CSV 保存完成")
print(train_daily_path)
print(eval_daily_path)
print(eval_history_daily_path)


# ============================================================
# 14. 标准化
# 注意：Scaler 只能在 train 上 fit，eval 只能 transform，避免数据泄露
# ============================================================
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

train_scaled = train_df.copy()
eval_scaled = eval_df_with_history.copy()

train_scaled[feature_cols] = feature_scaler.fit_transform(train_df[feature_cols])
eval_scaled[feature_cols] = feature_scaler.transform(eval_df_with_history[feature_cols])
train_scaled[feature_cols] = train_df[feature_cols]
eval_scaled[feature_cols] = eval_df_with_history[feature_cols]
# y 目标单独用 target_scaler
train_scaled[[target_col]] = target_scaler.fit_transform(train_df[[target_col]])
eval_scaled[[target_col]] = target_scaler.transform(eval_df_with_history[[target_col]])


train_scaled[[target_col]] = train_df[[target_col]]
eval_scaled[[target_col]] = eval_df_with_history[[target_col]]
feature_scaler_path = os.path.join(TRAIN_DIR, "feature_scaler.pkl")
target_scaler_path = os.path.join(TRAIN_DIR, "target_scaler.pkl")

joblib.dump(feature_scaler, feature_scaler_path)
joblib.dump(target_scaler, target_scaler_path)

with open(os.path.join(TRAIN_DIR, "feature_cols.json"), "w", encoding="utf-8") as f:
    json.dump(feature_cols, f, ensure_ascii=False, indent=2)

with open(os.path.join(TRAIN_DIR, "target_col.json"), "w", encoding="utf-8") as f:
    json.dump({"target_col": target_col}, f, ensure_ascii=False, indent=2)

print("=" * 80)
print("Scaler 保存完成")
print(feature_scaler_path)
print(target_scaler_path)


# ============================================================
# 15. 构造 90->90 和 90->365 样本
# ============================================================
for task in TASKS:
    task_name = task["name"]
    input_len = task["input_len"]
    pred_len = task["pred_len"]

    print("=" * 80)
    print(f"开始构造任务: {task_name}")
    print(f"输入长度: {input_len}, 预测长度: {pred_len}")

    X_train, y_train, y_train_dates, train_pred_start_dates = build_windows_with_dates(
        train_scaled,
        feature_cols=feature_cols,
        target_col=target_col,
        input_len=input_len,
        pred_len=pred_len
    )

    X_eval_all, y_eval_all, y_eval_dates_all, eval_pred_start_dates_all = build_windows_with_dates(
        eval_scaled,
        feature_cols=feature_cols,
        target_col=target_col,
        input_len=input_len,
        pred_len=pred_len
    )

    # 只保留预测起点位于真正 eval 区间的样本
    eval_start_date = eval_df_plain["date"].min()
    eval_pred_start_dates_pd = pd.to_datetime(eval_pred_start_dates_all)

    valid_mask = eval_pred_start_dates_pd >= eval_start_date

    X_eval = X_eval_all[valid_mask]
    y_eval = y_eval_all[valid_mask]
    y_eval_dates = y_eval_dates_all[valid_mask]
    eval_pred_start_dates = eval_pred_start_dates_all[valid_mask]

    train_npz_path = os.path.join(TRAIN_DIR, f"train_{task_name}.npz")
    eval_npz_path = os.path.join(EVAL_DIR, f"eval_{task_name}.npz")

    save_npz(
        train_npz_path,
        X_train,
        y_train,
        y_train_dates,
        train_pred_start_dates,
        feature_cols,
        target_col
    )

    save_npz(
        eval_npz_path,
        X_eval,
        y_eval,
        y_eval_dates,
        eval_pred_start_dates,
        feature_cols,
        target_col
    )

    print(f"{task_name} 保存完成")
    print("train X:", X_train.shape)
    print("train y:", y_train.shape)
    print("eval X:", X_eval.shape)
    print("eval y:", y_eval.shape)
    print("train 文件:", train_npz_path)
    print("eval 文件:", eval_npz_path)


# ============================================================
# 16. 保存划分信息
# ============================================================
split_info = {
    "train_ratio": TRAIN_RATIO,
    "input_len": INPUT_LEN,
    "split_idx": int(split_idx),
    "total_days": int(len(data)),
    "train_days": int(len(train_df)),
    "eval_days": int(len(eval_df_plain)),
    "train_start": str(train_df["date"].min().date()),
    "train_end": str(train_df["date"].max().date()),
    "eval_start": str(eval_df_plain["date"].min().date()),
    "eval_end": str(eval_df_plain["date"].max().date()),
    "target_col": target_col,
    "feature_cols": feature_cols,
    "weather_cols": weather_cols,
    "weather_missing_months_report": missing_report_path,
    "weather_filled_path": filled_weather_path,
}

split_info_path = os.path.join(DATA_DIR, "split_info.json")

with open(split_info_path, "w", encoding="utf-8") as f:
    json.dump(split_info, f, ensure_ascii=False, indent=2)

print("=" * 80)
print("全部完成")
print("train 目录:", TRAIN_DIR)
print("eval 目录:", EVAL_DIR)
print("划分信息:", split_info_path)