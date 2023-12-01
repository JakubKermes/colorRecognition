import os
import torch
from torchvision import transforms
from PIL import Image
from main import ColorRecognitionModel  # Import the model class

model_path = 'model.pth'

# Create an instance of the ColorRecognitionModel
model = ColorRecognitionModel()

# Load the trained model parameters into the instance
model.load_state_dict(torch.load(model_path))
model.eval()

# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to predict the color of an input image
def predict_color(image_path):
    img = Image.open(image_path)
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(img)

    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# Replace 'path_to_your_image.jpg' with the actual path to your input image
print("input image path or 'end' to exit: ")
input_image_path = input()
if input_image_path == 'end':
    exit()
while not os.path.exists(input_image_path):
    print("input image path or 'end' to exit: ")
    input_image_path = input()
    if input_image_path == 'end':
        exit()

predicted_color_index = predict_color(input_image_path)

class_index_to_color = {
    0: "black",
    1: "blue",
    2: "brown",
    3: "gray",
    4: "green",
    5: "orange",
    6: "pink",
    7: "purple",
    8: "red",
    9: "yellow",
}
predicted_color = class_index_to_color[predicted_color_index]

# Print or use the predicted color index as needed
print(f"Predicted Color: {predicted_color}")
