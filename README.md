This is a small command-line interface (CLI) to run local inference (with gpu-acceleration) on a model I built to classify microscopic white blood cell images. It was based on a pre-trained ResNet-18, to which I applied transfer learning with 16633 expert-labeled microscopic images of basophils, neutrophils, monocytes, eosinophils, and lymphocytes. 

![Immune Cell Image Classification Figure](immune_cell_image_classification_figure.png)

To make predictions on a single image, use the following command:

```bash
python classify_wbc.py --image_path <<image_path>>
```

Currently, this is only tested for .jpg and .png images. Also, my implementation will automatically resize the image, so don't worry about resolution. 

As an example, you can run this command on a randomly selected image of an eosinophil: 

```bash
python classify_wbc.py --image_path random_eosinophil_image.jpg
```

The output for this image should be the following:
<i>The model is 99.91% confident that the image is of a(n) Eosinophil</i>

Also, if you'd like to see the full workflow to train and evaluate this model, you can view that noteboook in ```white_blood_cell_image_classification.ipynb```

Lastly, I've included a full list of dependencies as a requirements.txt, so you can reconstruct my environment.

Thanks for reading!



