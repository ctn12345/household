import os
import sys
import json
import random
import argparse
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# 保证可以从项目根目录导入 dataset / modeling
ROOT_DIR = "/mnt/sdc/tnchen/matchine_learning"
sys.path.append(ROOT_DIR)

from dataset.dataset import build_dataloader
from modeling.LSTM import LSTMForecaster
from modeling.transformer import TransformerForecaster
from modeling.ADLGHFormer import ADLG_HFormer


def set_seed(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 提高可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()


def inverse_transform_target(tensor, target_scaler):
    """
    将 scaled 空间的 pred / y 还原到原始数值空间。

    支持形状：
        [B, pred_len]
        [B, pred_len, 1]
        [B]
    """
    original_shape = tensor.shape

    array = tensor.detach().cpu().numpy()
    array_2d = array.reshape(-1, 1)

    array_inv = target_scaler.inverse_transform(array_2d)
    array_inv = array_inv.reshape(original_shape)

    return array_inv


def compute_original_metrics(pred, target, target_scaler):
    """
    计算原始尺度下的 MSE / MAE。
    """
    pred_ori = inverse_transform_target(pred, target_scaler)
    target_ori = inverse_transform_target(target, target_scaler)

    mse_ori = np.mean((pred_ori - target_ori) ** 2)
    mae_ori = np.mean(np.abs(pred_ori - target_ori))

    return float(mse_ori), float(mae_ori)


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_num = 0

    pbar = tqdm(train_loader, desc="Train", ncols=120)

    for x, y in pbar:
        x = x.to(device).float()
        y = y.to(device).float()

        optimizer.zero_grad()

        pred = model(x)

        loss = criterion(pred, y)

        loss.backward()

        # 防止 LSTM / Transformer 训练时梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_size = x.size(0)

        batch_mae = compute_mae(pred.detach(), y.detach())

        total_loss += loss.item() * batch_size
        total_mae += batch_mae * batch_size
        total_num += batch_size

        pbar.set_postfix({
            "scaled_mse": f"{loss.item():.6f}",
            "scaled_mae": f"{batch_mae:.6f}"
        })

    avg_mse = total_loss / total_num
    avg_mae = total_mae / total_num

    return avg_mse, avg_mae


@torch.no_grad()
def evaluate(model, eval_loader, criterion, device, target_scaler=None):
    model.eval()

    total_loss = 0.0
    total_mae = 0.0

    total_ori_mse = 0.0
    total_ori_mae = 0.0

    total_num = 0

    pbar = tqdm(eval_loader, desc="Eval", ncols=120)

    for x, y in pbar:
        x = x.to(device).float()
        y = y.to(device).float()

        pred = model(x)

        loss = criterion(pred, y)

        batch_size = x.size(0)
        batch_scaled_mae = compute_mae(pred, y)

        total_loss += loss.item() * batch_size
        total_mae += batch_scaled_mae * batch_size

        if target_scaler is not None:
            batch_ori_mse, batch_ori_mae = compute_original_metrics(
                pred=pred,
                target=y,
                target_scaler=target_scaler
            )

            total_ori_mse += batch_ori_mse * batch_size
            total_ori_mae += batch_ori_mae * batch_size

            pbar.set_postfix({
                "scaled_mse": f"{loss.item():.6f}",
                "scaled_mae": f"{batch_scaled_mae:.6f}",
                "ori_mse": f"{batch_ori_mse:.6f}",
                "ori_mae": f"{batch_ori_mae:.6f}"
            })
        else:
            pbar.set_postfix({
                "scaled_mse": f"{loss.item():.6f}",
                "scaled_mae": f"{batch_scaled_mae:.6f}"
            })

        total_num += batch_size

    avg_scaled_mse = total_loss / total_num
    avg_scaled_mae = total_mae / total_num

    if target_scaler is not None:
        avg_ori_mse = total_ori_mse / total_num
        avg_ori_mae = total_ori_mae / total_num

        return avg_scaled_mse, avg_scaled_mae, avg_ori_mse, avg_ori_mae

    return avg_scaled_mse, avg_scaled_mae, None, None


def save_checkpoint(
    output_dir,
    epoch,
    model,
    optimizer,
    train_mse,
    train_mae,
    eval_mse,
    eval_mae,
    eval_ori_mse,
    eval_ori_mae,
    args,
    seed,
    is_best=False
):
    os.makedirs(output_dir, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "seed": seed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        "train_mse_scaled": train_mse,
        "train_mae_scaled": train_mae,

        "eval_mse_scaled": eval_mse,
        "eval_mae_scaled": eval_mae,

        "eval_mse_original": eval_ori_mse,
        "eval_mae_original": eval_ori_mae,

        "args": vars(args)
    }

    latest_path = os.path.join(output_dir, "latest.pt")
    torch.save(ckpt, latest_path)

    epoch_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch:03d}.pt")
    torch.save(ckpt, epoch_path)

    if is_best:
        best_path = os.path.join(output_dir, "best.pt")
        torch.save(ckpt, best_path)

    return latest_path


def build_model(args, feature_dim, pred_len, device):
    if args.model_type == "transformer":
        model = TransformerForecaster(
            input_dim=feature_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            pred_len=pred_len,
            dropout=args.dropout,
            pooling=args.pooling
        )

    elif args.model_type == "lstm":
        model = LSTMForecaster(
            input_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            pred_len=pred_len,
            dropout=args.dropout
        )

    elif args.model_type == "adlg_hformer":
        model = ADLG_HFormer(
            input_dim=feature_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            pred_len=pred_len,
            dropout=args.dropout,
        )

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    return model.to(device)


def save_final_summary(summary_records, summary_path):
    """
    保存多轮实验结果，并输出报告用的 MSE mean / MSE std / MAE mean / MAE std。

    如果存在 original 指标，则优先使用 original 指标；
    否则使用 scaled 指标。
    """
    df = pd.DataFrame(summary_records)

    df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    # 判断是否有 original 指标
    original_mse = df["best_eval_mse_original"].replace(-1, np.nan)
    original_mae = df["best_eval_mae_original"].replace(-1, np.nan)

    if original_mse.notna().any() and original_mae.notna().any():
        mse_values = original_mse.dropna()
        mae_values = original_mae.dropna()
        metric_scale = "original"
    else:
        mse_values = df["best_eval_mse_scaled"]
        mae_values = df["best_eval_mae_scaled"]
        metric_scale = "scaled"

    report_summary = {
        "metric_scale": metric_scale,
        "MSE mean": float(mse_values.mean()),
        "MSE std": float(mse_values.std(ddof=1)) if len(mse_values) > 1 else 0.0,
        "MAE mean": float(mae_values.mean()),
        "MAE std": float(mae_values.std(ddof=1)) if len(mae_values) > 1 else 0.0,
        "num_runs": int(len(df))
    }

    report_summary_path = summary_path.replace(".csv", "_report.csv")

    pd.DataFrame([report_summary]).to_csv(
        report_summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print("Final Evaluation Summary")
    print("=" * 80)
    print("metric_scale:", report_summary["metric_scale"])
    print("num_runs:", report_summary["num_runs"])
    print(f"MSE mean: {report_summary['MSE mean']:.8f}")
    print(f"MSE std : {report_summary['MSE std']:.8f}")
    print(f"MAE mean: {report_summary['MAE mean']:.8f}")
    print(f"MAE std : {report_summary['MAE std']:.8f}")
    print("每轮结果保存到:", summary_path)
    print("报告表格结果保存到:", report_summary_path)

    return report_summary


def run_one_seed(args, seed, exp_output_dir):
    set_seed(seed)

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    task_output_dir = os.path.join(
        exp_output_dir,
        f"seed_{seed}"
    )

    os.makedirs(task_output_dir, exist_ok=True)

    print("=" * 80)
    print(f"{args.model_type} Training")
    print("Task:", args.task)
    print("Seed:", seed)
    print("Device:", device)
    print("Output dir:", task_output_dir)
    print("=" * 80)

    # =========================
    # 1. 加载 target_scaler
    # =========================
    target_scaler = None

    target_scaler_path = args.target_scaler_path

    # 如果没有显式传入，则自动寻找默认路径
    if target_scaler_path is None:
        default_scaler_path = os.path.join(
            args.root_dir,
            "data",
            "train",
            "target_scaler.pkl"
        )

        if os.path.exists(default_scaler_path):
            target_scaler_path = default_scaler_path

    if target_scaler_path is not None:
        if os.path.exists(target_scaler_path):
            target_scaler = joblib.load(target_scaler_path)
            print("Loaded target scaler:", target_scaler_path)
        else:
            print("[Warning] target_scaler_path 不存在，将只输出 scaled 指标:")
            print(target_scaler_path)
    else:
        print("[Warning] 没有找到 target_scaler，将只输出 scaled 指标。")

    # =========================
    # 2. 加载数据
    # =========================
    train_dataset, train_loader = build_dataloader(
        root_dir=args.root_dir,
        split="train",
        task=args.task,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    eval_dataset, eval_loader = build_dataloader(
        root_dir=args.root_dir,
        split="eval",
        task=args.task,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    feature_dim = train_dataset.X.shape[-1]
    pred_len = train_dataset.y.shape[-1]

    print("feature_dim:", feature_dim)
    print("pred_len:", pred_len)
    print("train samples:", len(train_dataset))
    print("eval samples:", len(eval_dataset))

    # =========================
    # 3. 构建模型
    # =========================
    model = build_model(
        args=args,
        feature_dim=feature_dim,
        pred_len=pred_len,
        device=device
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 保存配置
    config = vars(args).copy()
    config["feature_dim"] = int(feature_dim)
    config["pred_len"] = int(pred_len)
    config["current_seed"] = int(seed)

    with open(os.path.join(task_output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # =========================
    # 4. 开始训练
    # =========================
    best_eval_mse = float("inf")
    best_eval_mae = float("inf")

    best_eval_ori_mse = None
    best_eval_ori_mae = None

    best_epoch = -1

    log_path = os.path.join(task_output_dir, "train_log.csv")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,"
            "train_mse_scaled,train_mae_scaled,"
            "eval_mse_scaled,eval_mae_scaled,"
            "eval_mse_original,eval_mae_original,"
            "best\n"
        )

    for epoch in range(1, args.epochs + 1):
        print("\n" + "=" * 80)
        print(f"Seed {seed} | Epoch [{epoch}/{args.epochs}]")
        print("=" * 80)

        train_mse, train_mae = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        eval_mse, eval_mae, eval_ori_mse, eval_ori_mae = evaluate(
            model=model,
            eval_loader=eval_loader,
            criterion=criterion,
            device=device,
            target_scaler=target_scaler
        )

        # 用 scaled eval_mse 判断 best，训练过程更稳定
        is_best = eval_mse < best_eval_mse

        if is_best:
            best_epoch = epoch
            best_eval_mse = eval_mse
            best_eval_mae = eval_mae

            if eval_ori_mse is not None:
                best_eval_ori_mse = eval_ori_mse
                best_eval_ori_mae = eval_ori_mae

        save_checkpoint(
            output_dir=task_output_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_mse=train_mse,
            train_mae=train_mae,
            eval_mse=eval_mse,
            eval_mae=eval_mae,
            eval_ori_mse=eval_ori_mse,
            eval_ori_mae=eval_ori_mae,
            args=args,
            seed=seed,
            is_best=is_best
        )

        eval_ori_mse_log = eval_ori_mse if eval_ori_mse is not None else -1
        eval_ori_mae_log = eval_ori_mae if eval_ori_mae is not None else -1

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},"
                f"{train_mse:.8f},{train_mae:.8f},"
                f"{eval_mse:.8f},{eval_mae:.8f},"
                f"{eval_ori_mse_log:.8f},{eval_ori_mae_log:.8f},"
                f"{int(is_best)}\n"
            )

        if eval_ori_mse is not None:
            print(
                f"Epoch {epoch:03d} | "
                f"Train MSE scaled: {train_mse:.6f} | "
                f"Train MAE scaled: {train_mae:.6f} | "
                f"Eval MSE scaled: {eval_mse:.6f} | "
                f"Eval MAE scaled: {eval_mae:.6f} | "
                f"Eval MSE original: {eval_ori_mse:.6f} | "
                f"Eval MAE original: {eval_ori_mae:.6f} | "
                f"Best: {is_best}"
            )
        else:
            print(
                f"Epoch {epoch:03d} | "
                f"Train MSE scaled: {train_mse:.6f} | "
                f"Train MAE scaled: {train_mae:.6f} | "
                f"Eval MSE scaled: {eval_mse:.6f} | "
                f"Eval MAE scaled: {eval_mae:.6f} | "
                f"Best: {is_best}"
            )

    print("\n当前 seed 训练完成")
    print("Seed:", seed)
    print("Best epoch:", best_epoch)
    print("Best eval MSE scaled:", best_eval_mse)
    print("Best eval MAE scaled:", best_eval_mae)

    if best_eval_ori_mse is not None:
        print("Best eval MSE original:", best_eval_ori_mse)
        print("Best eval MAE original:", best_eval_ori_mae)

    print("Checkpoint dir:", task_output_dir)

    return {
        "seed": int(seed),
        "best_epoch": int(best_epoch),

        "best_eval_mse_scaled": float(best_eval_mse),
        "best_eval_mae_scaled": float(best_eval_mae),

        "best_eval_mse_original": float(best_eval_ori_mse) if best_eval_ori_mse is not None else -1,
        "best_eval_mae_original": float(best_eval_ori_mae) if best_eval_ori_mae is not None else -1,

        "checkpoint_dir": task_output_dir
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", type=str, default="/mnt/sdc/tnchen/matchine_learning")
    parser.add_argument("--task", type=str, default="90to365", choices=["90to90", "90to365"])

    parser.add_argument(
        "--model_type",
        type=str,
        default="adlg_hformer",
        choices=["transformer", "lstm", "adlg_hformer"]
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # LSTM 参数
    parser.add_argument("--hidden_dim", type=int, default=128)

    # Transformer / ADLG-HFormer 参数
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--pooling", type=str, default="last", choices=["last", "mean"])

    # 公共模型参数
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--num_workers", type=int, default=4)

    # 多轮实验参数
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="重复实验次数，用于计算 MSE/MAE 的 mean 和 std"
    )

    parser.add_argument(
        "--seed_list",
        type=str,
        default="2026,2027,2028,2029,2030",
        help="多轮实验使用的随机种子，例如 2026,2027,2028,2029,2030"
    )

    # 保留单 seed 参数，方便只跑一轮时使用
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument(
        "--target_scaler_path",
        type=str,
        default=None,
        help="target_scaler.pkl 路径，用于把 scaled 的预测值还原到原始分布"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="例如 cuda:0 / cuda:1 / cuda:2 / cpu"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录。默认保存到 root_dir/outputs/model_type"
    )

    args = parser.parse_args()

    # =========================
    # 1. 处理 seed list
    # =========================
    if args.repeat <= 1:
        seed_list = [args.seed]
    else:
        seed_list = [
            int(s.strip())
            for s in args.seed_list.split(",")
            if s.strip() != ""
        ]

        if len(seed_list) == 0:
            seed_list = [args.seed]

        if args.repeat > len(seed_list):
            base = seed_list[-1]

            for i in range(args.repeat - len(seed_list)):
                seed_list.append(base + i + 1)

        seed_list = seed_list[:args.repeat]

    # =========================
    # 2. 设置输出目录
    # =========================
    if args.output_dir is None:
        output_dir = os.path.join(
            args.root_dir,
            "outputs",
            args.model_type
        )
    else:
        output_dir = args.output_dir

    exp_output_dir = os.path.join(
        output_dir,
        args.task,
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    os.makedirs(exp_output_dir, exist_ok=True)

    print("=" * 80)
    print("Multi-run experiment")
    print("Model:", args.model_type)
    print("Task:", args.task)
    print("Repeat:", len(seed_list))
    print("Seeds:", seed_list)
    print("Experiment output dir:", exp_output_dir)
    print("=" * 80)

    # 保存总配置
    with open(os.path.join(exp_output_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # =========================
    # 3. 多轮训练
    # =========================
    summary_records = []

    for seed in seed_list:
        record = run_one_seed(
            args=args,
            seed=seed,
            exp_output_dir=exp_output_dir
        )

        summary_records.append(record)

    # =========================
    # 4. 保存最终 mean/std
    # =========================
    summary_path = os.path.join(
        exp_output_dir,
        "final_eval_summary.csv"
    )

    save_final_summary(
        summary_records=summary_records,
        summary_path=summary_path
    )


if __name__ == "__main__":
    main()