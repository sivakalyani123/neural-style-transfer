## Neural Style Transfer

This project demonstrates **Neural Style Transfer (NST)** using PyTorch. The objective is to take a content image and a style image, and generate a new image that preserves the content of the first but adopts the artistic style of the second.

## 📌 Objective

Implement and run Neural Style Transfer using the **VGG19** network pretrained on ImageNet, applying the style from one image onto the content of another.

---

## 🧠 Key Concepts

- **Content Image**: The image whose structure you want to retain.
- **Style Image**: The image whose artistic features you want to apply.
- **Output Image**: The generated image with the content of the first and style of the second.
- **VGG19**: A convolutional neural network used to extract features from both content and style images.

---

## 🧰 Requirements

- Python 3.8 or above
- PyTorch
- Torchvision
- Pillow
- Matplotlib
- OpenCV (optional for image processing)

Install the dependencies using:

pip install torch torchvision pillow matplotlib opencv-python

📂 Folder Structure
'''
neural-style-transfer/
│
├── content.jpg                # Content image
├── style.jpg                  # Style image
├── output.png                 # Final generated image (after running the script)
├── neural_style_transfer.py   # Main Python script
└── README.md                  # This file
'''
▶️ How to Run
Place your content and style images in the project directory.

Rename them (or edit the code) to content.jpg and style.jpg.

Run the script:
python neural_style_transfer.py
The result will be saved as output.png.
