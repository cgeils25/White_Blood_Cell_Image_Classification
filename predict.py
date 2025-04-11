import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as v2
import time

from train import CLASS_MAP_INV, get_device, get_test_transform

logging=False

def load_model(model_path: str, device: torch.device = None) -> nn.Module:
    """Load a pytorch model from a given path

    Args:
        model_path (str): path to the pytorch model
        device (torch.device, optional): the device to load the model on. Defaults to None.

    Raises:
        ValueError: if the model path does not end in 'pt

    Returns:
        nn.Module: a pytorch model
    """

    print(f"Loading model from {model_path} on device {device}")
    
    if not model_path.endswith('.pt'):
        raise ValueError("Model path must end with .pt")

    model = torch.load(model_path, weights_only=False, map_location=device)
    print("Model loaded successfully")

    return model


def get_image_from_path(image_path: str, device: torch.device = None, image_transform: v2.Compose =None) -> torch.tensor:
    """Load an image as a tensor and prepare for input to model

    Args:
        image_path (str): path to the image
        device (torch.device, optional): the device to load the tensor onto. Defaults to None.
        image_transform (v2.Compose, optional): transformations to be applied to the data. Defaults to None.

    Returns:
        torch.tensor: the image as a tensor
    """
    img_as_tensor = torchvision.io.read_image(image_path)

    img_as_tensor = image_transform(img_as_tensor)

    img_as_tensor = img_as_tensor.to(device)

    return img_as_tensor


class PredictionResponse:
    """
    A response model for prediction results
    """
    def __init__(self, predictions: dict, predicted_class: str, status: str):
        self.predictions = predictions
        self.predicted_class = predicted_class
        self.status = status


def predict(model: nn.Module, image_path: str, device: torch.device, image_transform: v2.Compose) -> PredictionResponse:
    """Make a prediction on a given image

    Args:
        model (nn.Module): a pytorch model
        image_path (str): the path to the image
        device (torch.device): the device to load the model and tensor onto
        image_transform (v2.Compose): transformations to be applied to the data.

    Returns:
        PredictionResponse: predictions
    """

    try:
        image = get_image_from_path(image_path, device=device, image_transform=image_transform)

        logits = model(image.unsqueeze(0)).detach().cpu().squeeze(0)

        predictions = logits.softmax(dim=0)

        predictions_dict = {CLASS_MAP_INV[i]: predictions[i].item() for i in range(len(predictions))}

        predicted_class = max(predictions_dict, key=predictions_dict.get)

        return PredictionResponse(predictions_dict, predicted_class, "Success")
    
    except Exception as e:
        print(f"Prediction failed with eror: {e}")
        return PredictionResponse({}, "Unknown", "Failed")
   

def parse_args():
    parser = argparse.ArgumentParser(description="Predict class of leukocyte images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--model_name", type=str, help="Name of the model (optional)")
    parser.add_argument("--use_gpu", action='store_true', help="Whether to attempt to use GPU for inference")
    return parser.parse_args()


def main(args):
    print("="*150)

    if args.model_name:
        print(f"LEUKOCYTE IMAGE PREDICTION SUITE FOR MODEL \"{args.model_name}\" at {args.model_path}")
    
    else:
        print("LEUKOCYTE IMAGE PREDICTION SUITE FOR MODEL at {args.model_path}")

    print("="*150)

    device = get_device(use_gpu=args.use_gpu)

    model = load_model(args.model_path, device=device)
    
    image_transform = get_test_transform()

    print("Enter the path to the image: ")
    while True:
        user_input = input(">")

        if user_input.strip().lower() in ['exit', 'quit', 'q']:
            print("Exiting...")
            break
        
        start = time.monotonic()

        response = predict(model, user_input, device=device, image_transform=image_transform)

        end = time.monotonic()

        if response.status == "Success":
            print(f"Predicted class: {response.predicted_class}\n")
            
            print("Predicted probabilities:")
            for cls, prob in response.predictions.items():
                print(f"\t{cls}: {prob:.4f}")

            print(f"\nPrediction time: {end - start:.4f} seconds")
    
    print("Goodbye!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
