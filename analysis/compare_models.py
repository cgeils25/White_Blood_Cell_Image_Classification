import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from neattime import neattime


def parse_args():
    parser = argparse.ArgumentParser(description="Compare models' performance.")
    parser.add_argument('--model_dirs', type=str, nargs='+', required=True, help='List of directories containing model training results')
    parser.add_argument('--model_names', type=str, nargs='+', required=True, help='List of model names corresponding to the directories')
    return parser.parse_args()

def get_test_loss(model_result_dir):
    loss_df = pd.read_csv(f"{model_result_dir}/losses.csv")

    test_loss = loss_df['Test'].iloc[-1]

    return test_loss

def save_comparison_plot(model_dirs, model_names):
    sns.set_style("darkgrid")

    plot_filename = "model_comparison_plot"
    test_losses = []

    for model_dir, model_name in zip(model_dirs, model_names):
        plot_filename += f"_{model_dir}"
        test_loss = get_test_loss(model_dir)
        test_losses.append(test_loss)
    
    plot_filename += ".png"

    ax = sns.barplot(x=model_names, y=test_losses)

    plt.title('Model Comparison: Performance of Validation-Selected Model on Test Set')
    plt.xlabel('Model Name')
    plt.ylabel('Cross-Entropy Loss')

    for container in ax.containers:
        ax.bar_label(container)

    plt.savefig(plot_filename, dpi=1000)
    print(f"Model comparison plot saved to {plot_filename}")


def main(args):
    save_comparison_plot(args.model_dirs, args.model_names)


if __name__ == "__main__":
    args = parse_args()

    main(args)