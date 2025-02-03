from src.model import MyConvNeXt
from src.dataset import MyDataset

from torchvision.models import ConvNeXt_Tiny_Weights
import matplotlib.pyplot as plt
from glob import glob
import torch
from PIL import Image
import torch.nn.functional as F

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transforms = ConvNeXt_Tiny_Weights.DEFAULT.transforms()
    checkpoint = torch.load("trained_model/best.pt")
    
    # Load images
    imgs_path = glob("images/*.jpg")
    images = [Image.open(img_path) for img_path in imgs_path]
    
    # Apply transforms to each image and stack them
    x = torch.stack([transforms(img) for img in images])
    
    # Load model
    model = MyConvNeXt(2).to(device)
    model.load_state_dict(checkpoint["model"]) # Load your trained model weights
    model.eval()
    
    # Perform inference
    with torch.no_grad():
        x = x.to(device)
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
    
    # Display results
    for i, img_path in enumerate(imgs_path):
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f"Predicted: {preds[i].item()}, Probability: {probs[i][preds[i]].item()}")
        plt.show()