import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from neattime import neattime

from train import CLASS_MAP, CLASS_MAP_INV
CELL_TYPES = list(CLASS_MAP.keys()) # ensures they're in the same order that the labels are defined


def plot_train_valid_loss(model_result_dir, out_dir, model_name):
    sns.set_style(style="darkgrid")

    loss_df = pd.read_csv(f"{model_result_dir}/losses.csv")

    sns.lineplot(data=loss_df[['Train', 'Validation']], palette='hls', dashes=False)
    plt.ylabel('Cross-Entropy Loss')
    plt.xlabel('Epoch')
    plt.title(f'Training and Validation Loss for Leukocyte Image Classification \nUsing {model_name}')

    plt.axvline(x=np.argmin(loss_df.Validation), color='gray', linestyle='--', linewidth='0.8')
    plt.axhline(y=np.min(loss_df.Validation), color='gray', linestyle='--', linewidth='0.8')
    plt.text(x=np.argmin(loss_df.Validation)-.25, y=np.min(loss_df.Validation)+0.01, 
            s=f"Epoch: {np.argmin(loss_df.Validation)} \nValid. Loss: {round(np.min(loss_df.Validation), 3)}", 
            color='black', fontsize=7, ha='right')

    filename_train_and_valid=f"{out_dir}/train_and_val_loss_plot_{model_name}.png"
    plt.savefig(fname=filename_train_and_valid, dpi=1000)
    print(f"Training and validation loss plot saved to {filename_train_and_valid}.png")
    plt.close()


def plot_val_losses_per_class(model_result_dir, out_dir, model_name):
    sns.set_style(style="darkgrid")

    loss_df = pd.read_csv(f"{model_result_dir}/losses.csv")

    sns.lineplot(data=loss_df[CELL_TYPES], palette='hls', dashes=False)
    plt.ylabel('Cross-Entropy Loss')
    plt.xlabel('Epoch')
    plt.title(f'Validation Loss for Different Leukocyte Types \nUsing {model_name}')
    filename_diff_immune_cell=f"{out_dir}/val_losses_per_class_plot_{model_name}"
    plt.savefig(fname=filename_diff_immune_cell, dpi=1000)
    print(f"Validation losses per class plot saved to {filename_diff_immune_cell}.png")
    plt.close()


def get_test_predictions_df(model_result_dir):
    test_predictions_df = pd.read_csv(f"{model_result_dir}/test_predictions.csv")

    # add the prediction made by the model based on highest predicted probability
    test_predictions_df['Predictions'] = np.argmax(test_predictions_df[CELL_TYPES], axis=1)
    test_predictions_df['Predictions'] = test_predictions_df['Predictions'].map(CLASS_MAP_INV)

    # also replace the label with the actual class name
    test_predictions_df['Labels'] = test_predictions_df['Labels'].map(CLASS_MAP_INV)
    
    return test_predictions_df


def plot_confusion_matrix(model_result_dir, out_dir, model_name):
    test_predictions_df = get_test_predictions_df(model_result_dir=model_result_dir)
    
    # extract the labels and predictions for evaluation
    labels, predictions = test_predictions_df['Labels'].tolist(), test_predictions_df['Predictions'].tolist()

    # plot and save confusion matrix
    sns.set_style("dark")
    cm = confusion_matrix(labels, predictions, labels=CELL_TYPES, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CELL_TYPES)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for Leukocyte Image Classification with {model_name}')

    filename_confusion_matrix=f"{out_dir}/confusion_matrix_plot_{model_name}.png"
    plt.savefig(fname=filename_confusion_matrix, dpi=1000)
    print(f"Confusion matrix saved to {filename_confusion_matrix}")
    plt.close()


def save_classification_report(model_result_dir, out_dir, model_name):
    test_predictions_df = get_test_predictions_df(model_result_dir=model_result_dir)

    # extract the labels and predictions for evaluation
    labels, predictions = test_predictions_df['Labels'].tolist(), test_predictions_df['Predictions'].tolist()

    classification_report_output = classification_report(labels, predictions, target_names=CELL_TYPES, output_dict=True)
    
    classification_report_df = pd.DataFrame(classification_report_output)

    classification_report_filename = f"{out_dir}/classification_report_{model_name}.csv"
    classification_report_df.to_csv(classification_report_filename)
    print(f"Classification report saved to {classification_report_filename}")


def parse_args():
    parser = argparse.ArgumentParser(description = "Generate plots and tables to analyze model training data using the model result directory")
    parser.add_argument('--model_result_dir', type=str, required=True, help='Path to directory containing model training results')
    parser.add_argument('--model_name', type=str, default='ResNet-18', help='Name of the model used for training')
    return parser.parse_args()


def main(args):
    out_dir = args.model_result_dir + "/plots"
    os.makedirs(out_dir, exist_ok=True) 

    plot_train_valid_loss(args.model_result_dir, out_dir, args.model_name)
    plot_val_losses_per_class(args.model_result_dir, out_dir, args.model_name)
    plot_confusion_matrix(args.model_result_dir, out_dir, args.model_name)
    save_classification_report(args.model_result_dir, out_dir, args.model_name)


if __name__ == '__main__':
    args = parse_args()
    main(args)
