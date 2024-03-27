import argparse
import matplotlib.pyplot as plt
from parse import compile
from pathlib import Path


def parse_log(log_file):
    epochs = []
    d_losses = []
    adv_losses = []
    g_losses = []
    cycle_losses = []
    identity_losses = []

    with open(log_file, 'r') as f:
        log_format = compile("[Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G loss: {}, adv: {}, cycle: {}, identity: {}] ETA: --")
        for line in f:
            line = line.strip()
            parsed = log_format.parse(line)
            if parsed is not None:
                epoch = int(parsed[0])
                d_loss = float(parsed[4])
                g_loss = float(parsed[5])
                adv_loss = float(parsed[6])
                cycle_loss = float(parsed[7])
                identity_loss = float(parsed[8])
                epochs.append(epoch)
                d_losses.append(d_loss)
                g_losses.append(g_loss)
                adv_losses.append(adv_loss)
                cycle_losses.append(cycle_loss)
                identity_losses.append(identity_loss)

    return epochs, d_losses, g_losses, adv_losses, cycle_losses, identity_losses


def plot_loss_curve(epochs, d_losses, g_losses, adv_losses, cycle_losses, identity_losses, output_dir):
    plt.plot(d_losses, label='D loss')
    plt.plot(g_losses, label='G loss')
    #plt.plot(epochs, adv_losses, label='Adv loss')
    #plt.plot(epochs, cycle_losses, label='Cycle loss')
    #plt.plot(epochs, identity_losses, label='Identity loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir + '/loss_curve.png')  # Save figure instead of showing it
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot loss curve from log file")
    parser.add_argument("log_file", help="Input log file containing loss information")
    parser.add_argument("output_dir", help="Output directory to save the figure")
    args = parser.parse_args()
    print(args)

    epochs, d_losses, g_losses, adv_losses, cycle_losses, identity_losses = parse_log(args.log_file)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    plot_loss_curve(epochs, d_losses, g_losses, adv_losses, cycle_losses, identity_losses, args.output_dir)


if __name__ == "__main__":
    main()
