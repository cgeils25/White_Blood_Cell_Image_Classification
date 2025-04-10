# Classifying Leukocyte (White Blood Cell) Images with Deep Learning

This is a model training pipeline which includes a CLI to run local inference to classify microscopic white blood cell images. It was based on a pre-trained ResNet-18, which I fine-tuned with 16633 expert-labeled microscopic images of basophils, neutrophils, monocytes, eosinophils, and lymphocytes. 

![Immune Cell Image Classification Figure](immune_cell_image_classification_figure.png)


## Environment Setup

I managed my dependencies with uv. [View the instructions to install uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it on your system.

To build a suitable virtual environment, run:

```bash
uv sync
```

## The Data

The data include 16,633 expert-labelled microscopic images fo white blood cells, which are provided by [RaabinData](https://raabindata.com/).

To download all 16,633 images, run:

```bash
uv run python download_data.py
```

## Train the Model

You can reproduce my model by re-running my entire training pipeline with:

```bash
uv run python train.py --use_gpu --weight_loss --save_test_predictions_and_labels
```

For info on all command line arguments, run:

```bash
uv run python train.py --help
```

## Plot and Analyze Training Results

To plot and analyze the results of model training, use:

```bash
uv run python analysis/train_analysis.py --model_result_dir ... --model_name ...
```

And pass the name of the model result directory containing the outputs of `train.py`.


## To Run Inference on a New Image:

As an example:

```bash
uv run python classify_wbc.py --image_path random_eosinophil_image.jpg
```

Output:

```
The model is 99.91% confident that the image is of a(n) Eosinophil
```



