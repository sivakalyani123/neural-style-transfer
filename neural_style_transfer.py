# neural_style_transfer.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import copy

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loading and preprocessing
def load_image(path, max_size=400):
    image = Image.open(path).convert("RGB")
    
    size = max_size if max(image.size) > max_size else max(image.size)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image.to(device)

# Function to display image
def imshow(tensor, title=None):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()

# Load content and style images
content = load_image("content.jpg")
style = load_image("style.jpg")

# Display input images
imshow(content, "Content Image")
imshow(style, "Style Image")

# Load pre-trained VGG19 model
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Content and style layers
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Helper to extract features
class FeatureExtractor(nn.Module):
    def __init__(self, model, content_layers, style_layers):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.layers = []
        self.name_map = {}
        self._register_layers()

    def _register_layers(self):
        i = 0
        for layer in self.model.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f"conv_{i}"
            elif isinstance(layer, nn.ReLU):
                name = f"relu_{i}"
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f"pool_{i}"
            elif isinstance(layer, nn.BatchNorm2d):
                name = f"bn_{i}"
            else:
                continue
            self.layers.append((name, layer))

    def forward(self, x):
        features = {}
        for name, layer in self.layers:
            x = layer(x)
            if name in self.content_layers + self.style_layers:
                features[name] = x
        return features

# Compute gram matrix (for style)
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram.div(b * c * h * w)

# Style transfer
target = content.clone().requires_grad_(True)

optimizer = torch.optim.Adam([target], lr=0.003)
model = FeatureExtractor(vgg, content_layers, style_layers).to(device)

style_features = model(style)
content_features = model(content)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}

# Weights
style_weight = 1e6
content_weight = 1

# Optimization loop
for step in range(1, 501):
    target_features = model(target)
    
    content_loss = torch.mean((target_features['conv_4'] - content_features['conv_4'])**2)
    
    style_loss = 0
    for layer in style_layers:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        style_loss += torch.mean((target_gram - style_gram) ** 2)
    
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    optimizer.zero_grad()
    total_loss.backward(retain_graph=True)

    optimizer.step()

    if step % 100 == 0:
        print(f"Step [{step}/500], Loss: {total_loss.item():.4f}")

# Show result
imshow(target, "Styled Image")
# Save the result
output = target.clone().detach().cpu().squeeze(0)
output_image = transforms.ToPILImage()(output)
output_image.save("styled_output.jpg")
