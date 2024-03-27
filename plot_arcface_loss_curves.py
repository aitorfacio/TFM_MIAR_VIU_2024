import sys
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from parse import *
from parse import compile
import pandas as pd

def parse_training_logs(log_file):
    timestamps = []
    steps = []
    losses = []
    epochs = []
    with open(log_file, 'r') as file:
        log_format = compile(
            "Training: {} {}-Speed {} samples/sec   Loss {}   LearningRate {}   Epoch: {}   Global Step: {}   Fp16 Grad Scale: {}   Required: {} hours")
        for line in file:
            line = line.strip()
            if line.startswith('Training:') and line.endswith("hours"):
                data = log_format.parse(line)
                timestamp_str = data[0] + ' ' + data[1]
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                timestamps.append(timestamp)
                losses.append(float(data[3]))
                steps.append(int(data[6]))
                epochs.append(int(data[5]))

    return pd.DataFrame({'Timestamp': timestamps, 'Loss': losses, 'Step': steps, 'Epoch': epochs})


def plot_loss_curve(df, save_folder):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Step'], df['Loss'] )
    plt.title('Training Loss Curve')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    # Filtering steps that are multiples of 5000
    tick_step_df = df[df['Step'] % 10000 == 0]
    plt.xticks(tick_step_df['Step'].values)
    save_path = Path(save_folder) / 'loss_curve.png'
    plt.savefig(save_path)
    print(f"Loss curve saved at: {save_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path/to/training_logs.txt> <path/to/save/folder>")
        sys.exit(1)

    log_file = sys.argv[1]
    save_folder = sys.argv[2]
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    df= parse_training_logs(log_file)
    #print("Timestamps:", len(timestamps))
    #print("Losses:", len(losses))
    #print(f"{losses[0]}-->{losses[-1]}")
    plot_loss_curve(df, save_folder)
