import os
import sys
import argparse
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT_DIR = "/mnt/sdc/tnchen/matchine_learning"
sys.path.append(ROOT_DIR)

from modeling.LSTM import LSTMForecaster
from modeling.transformer import TransformerForecaster
from modeling.ADLGHFormer import ADLG_HFormer


def inverse_transform(array, target_scaler):
    original_shape = array.shape
    array_2d = array.reshape(-1, 1)
    array_inv = target_scaler.inverse_transform(array_2d)
    return array_inv.reshape(original_shape)


def build_model(model_type, ckpt_args, feature_dim, pred_len, device):
    if model_type == "lstm":
        model = LSTMForecaster(
            input_dim=feature_dim,
            hidden_dim=ckpt_args.get("hidden_dim", 128),
            num_layers=ckpt_args.get("num_layers", 2),
            pred_len=pred_len,
            dropout=ckpt_args.get("dropout", 0.2)
        )

    elif model_type == "transformer":
        model = TransformerForecaster(
            input_dim=feature_dim,
            d_model=ckpt_args.get("d_model", 128),
            nhead=ckpt_args.get("nhead", 4),
            num_layers=ckpt_args.get("num_layers", 3),
            dim_feedforward=ckpt_args.get("dim_feedforward", 256),
            pred_len=pred_len,
            dropout=ckpt_args.get("dropout", 0.1),
            pooling=ckpt_args.get("pooling", "last")
        )

    elif model_type == "adlg_hformer":
        model = ADLG_HFormer(
            input_dim=feature_dim,
            d_model=ckpt_args.get("d_model", 128),
            nhead=ckpt_args.get("nhead", 4),
            num_layers=ckpt_args.get("num_layers", 3),
            dim_feedforward=ckpt_args.get("dim_feedforward", 256),
            pred_len=pred_len,
            dropout=ckpt_args.get("dropout", 0.1)
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model.to(device)


@torch.no_grad()
def predict_one_sample(model, x_sample, device):
    model.eval()
    x = torch.from_numpy(x_sample).unsqueeze(0).float().to(device)  # [1, 90, C]
    pred = model(x)  # [1, pred_len]
    return pred.squeeze(0).detach().cpu().numpy()


def load_prediction_from_ckpt(ckpt_path, model_type, x_sample, feature_dim, pred_len, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt["args"]

    model = build_model(
        model_type=model_type,
        ckpt_args=ckpt_args,
        feature_dim=feature_dim,
        pred_len=pred_len,
        device=device
    )

    model.load_state_dict(ckpt["model_state_dict"])
    pred = predict_one_sample(model, x_sample, device)

    return pred


def compute_metrics(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    mae = np.mean(np.abs(pred - gt))
    return mse, mae


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="90to365", choices=["90to90", "90to365"])
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:1")

    parser.add_argument("--lstm_ckpt", type=str, required=True)
    parser.add_argument("--transformer_ckpt", type=str, required=True)
    parser.add_argument("--adlg_ckpt", type=str, required=True)

    parser.add_argument(
        "--eval_npz",
        type=str,
        default="/mnt/sdc/tnchen/matchine_learning/data/eval/eval_90to365.npz"
    )

    parser.add_argument(
        "--target_scaler_path",
        type=str,
        default="/mnt/sdc/tnchen/matchine_learning/data/train/target_scaler.pkl"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="/mnt/sdc/tnchen/matchine_learning/compare_three_models.png"
    )

    args = parser.parse_args()

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    # 1. 读取 eval 数据
    data = np.load(args.eval_npz, allow_pickle=True)
    X_eval = data["X"].astype(np.float32)
    y_eval = data["y"].astype(np.float32)

    feature_dim = X_eval.shape[-1]
    pred_len = y_eval.shape[-1]

    if args.sample_idx < 0 or args.sample_idx >= len(X_eval):
        raise ValueError(f"sample_idx 越界，当前 eval 样本数为 {len(X_eval)}")

    x_sample = X_eval[args.sample_idx]
    y_sample = y_eval[args.sample_idx]

    # 2. 三个模型分别预测
    pred_lstm = load_prediction_from_ckpt(
        ckpt_path=args.lstm_ckpt,
        model_type="lstm",
        x_sample=x_sample,
        feature_dim=feature_dim,
        pred_len=pred_len,
        device=device
    )

    pred_transformer = load_prediction_from_ckpt(
        ckpt_path=args.transformer_ckpt,
        model_type="transformer",
        x_sample=x_sample,
        feature_dim=feature_dim,
        pred_len=pred_len,
        device=device
    )

    pred_adlg = load_prediction_from_ckpt(
        ckpt_path=args.adlg_ckpt,
        model_type="adlg_hformer",
        x_sample=x_sample,
        feature_dim=feature_dim,
        pred_len=pred_len,
        device=device
    )

    # 3. 反归一化
    if args.target_scaler_path is not None and os.path.exists(args.target_scaler_path):
        target_scaler = joblib.load(args.target_scaler_path)

        y_sample = inverse_transform(y_sample, target_scaler)
        pred_lstm = inverse_transform(pred_lstm, target_scaler)
        pred_transformer = inverse_transform(pred_transformer, target_scaler)
        pred_adlg = inverse_transform(pred_adlg, target_scaler)

        scale_name = "original"
    else:
        scale_name = "scaled"

    # 4. 计算每个模型的 MSE / MAE
    lstm_mse, lstm_mae = compute_metrics(pred_lstm, y_sample)
    trans_mse, trans_mae = compute_metrics(pred_transformer, y_sample)
    adlg_mse, adlg_mae = compute_metrics(pred_adlg, y_sample)

    print("=" * 80)
    print("Sample index:", args.sample_idx)
    print("Scale:", scale_name)
    print(f"LSTM         -> MSE: {lstm_mse:.6f}, MAE: {lstm_mae:.6f}")
    print(f"Transformer  -> MSE: {trans_mse:.6f}, MAE: {trans_mae:.6f}")
    print(f"ADLG-HFormer -> MSE: {adlg_mse:.6f}, MAE: {adlg_mae:.6f}")
    print("=" * 80)

    # 5. 画图
    days = np.arange(1, pred_len + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(days, y_sample, label="Ground Truth", linewidth=2.5)
    plt.plot(days, pred_lstm, label=f"LSTM (MSE={lstm_mse:.4f}, MAE={lstm_mae:.4f})", linestyle="--", linewidth=2)
    plt.plot(days, pred_transformer, label=f"Transformer (MSE={trans_mse:.4f}, MAE={trans_mae:.4f})", linestyle="--", linewidth=2)
    plt.plot(days, pred_adlg, label=f"ADLG-HFormer (MSE={adlg_mse:.4f}, MAE={adlg_mae:.4f})", linestyle="--", linewidth=2)

    plt.xlabel("Forecast horizon / day")
    plt.ylabel("Daily average active power / kW")
    plt.title(f"{args.task} Prediction vs Ground Truth (sample {args.sample_idx})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300)
    plt.close()

    print("对比图已保存到:", args.save_path)


if __name__ == "__main__":
    main()