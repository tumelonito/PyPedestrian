# Pedestrian Detection on CPU 🚶‍♂️

This repository contains the code for my Deep Learning for Computer Vision project (Track: Real-world / Embedded / Mobile Vision).

The goal of this project is to build a lightweight pedestrian detector that can run efficiently on a standard laptop CPU without needing a powerful GPU. I used a Faster R-CNN model with a MobileNetV3 backbone and applied dynamic quantization to optimize inference speed.

## 📂 Repository Structure

* `PyPedestrian.ipynb`: The main Google Colab / Jupyter Notebook containing the entire pipeline (data loading, training, CPU benchmarking, quantization, and inference).
* `requirements.txt`: List of Python dependencies needed to run the notebook locally.

## 🛠️ Setup and Dependencies

If you want to run this notebook locally, you need Python 3.10+ and the following main libraries:

* `torch` and `torchvision`
* `matplotlib` (for drawing bounding boxes)
* `Pillow` (PIL)
* `torchmetrics` (for evaluation)

**To install the dependencies:**
```bash
pip install -r requirements.txt
```

_Note: If you are on osx64, I highly recommend using a Conda environment to install PyTorch to avoid binary compatibility issues._

## 🚀 How to Run

1. Clone this repository. 
2. Open PyPedestrian.ipynb in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run the cells sequentially.
4. The notebook is set up to automatically download the Penn-Fudan Pedestrian Dataset if it's not already in the folder.

_Note: If you want to skip training and just test the inference, you can download my pre-trained quantized model here: [[Google Drive Link]](https://drive.google.com/file/d/1iAe0hzdonNeel88ae5T9IYmw941id1Au/view?usp=sharing). Place it in the root folder._

## 📊 Current Progress

* Model: Faster R-CNN with MobileNetV3-Large FPN.
* Environment: Trained on Colab GPU, benchmarked on local CPU.
* Quantization: Used torch.quantization.quantize_dynamic (qnnpack backend) to compress the model.
* Results: The unquantized model runs at ~9-10 FPS on my CPU. The quantized model shows a speedup (~11-12 FPS) and a ~55% reduction in file size (from ~76MB to ~34MB) with minimal accuracy drop.

## 📝 Future Plans

* Add data augmentation to the training pipeline.
* Create a simple OpenCV script for webcam inference.