import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import matplotlib.ticker as ticker

# Set up the argument parser
parser = argparse.ArgumentParser(description='Plot StyleGAN2 Training Loss Curves.')
parser.add_argument('--input_file', '-i', type=str, required=True,
                    help='Path to the log file containing JSON log entries.')
parser.add_argument('--output_dir', '-o', type=str, required=True,
                    help='Directory where the loss curve plot will be saved.')

# Parse the command-line arguments
args = parser.parse_args()

# Read the log file and parse each line as JSON
with open(args.input_file, 'r') as file:
    log_lines = file.readlines()

# Parse each JSON log entry and collect the Generator and Discriminator losses with kimgs
gen_losses = []
disc_losses = []
kimgs = []

for line in log_lines:
    # Parse the JSON line
    log_entry = json.loads(line)

    # Extract Generator and Discriminator losses and the kimg value
    gen_loss = log_entry["Loss/G/loss"]["mean"]
    disc_loss = log_entry["Loss/D/loss"]["mean"]
    kimg = log_entry["Progress/kimg"]["mean"]

    # Append to the lists
    gen_losses.append(gen_loss)
    disc_losses.append(disc_loss)
    kimgs.append(kimg)

# Create a DataFrame
df_losses = pd.DataFrame({
    'Kimg': kimgs,
    'Generator_Loss': gen_losses,
    'Discriminator_Loss': disc_losses
})

# Plot the loss curves
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_losses['Kimg'], df_losses['Generator_Loss'], label='Generator Loss')
ax.plot(df_losses['Kimg'], df_losses['Discriminator_Loss'], label='Discriminator Loss')
ax.set_title('Generator/Discriminator Loss Curve')
ax.set_xlabel('Kimg')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True)

# Set the x-axis major tick locator to enforce ticks every 5000 kimgs
ax.xaxis.set_major_locator(ticker.MultipleLocator(5000))

# Ensure the output directory exists
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

# Save the plot
output_file_path = Path(args.output_dir) / 'loss_curve.png'
plt.savefig(output_file_path)
print(f"Loss curve plot saved at: {output_file_path}")

# If you don't want to display the plot in a window, comment out plt.show()
# plt.show()
