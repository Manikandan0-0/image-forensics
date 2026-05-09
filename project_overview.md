# Adversarial Attack Explorer: Project Documentation

## 🎯 Project Vision
The **Adversarial Attack Explorer** is an interactive full-stack application designed to visualize how deep learning models can be manipulated. By adding carefully calculated, human-imperceptible noise to an image, the tool demonstrates how a state-of-the-art vision model (ResNet-18) can be coerced into misclassifying objects with high confidence.

## 🧠 Core Concept: The Targeted Attack
This project implements the **Iterative Target Class Method** (a variation of the Basic Iterative Method/PGD). 

Unlike a "non-targeted" attack which simply aims to make the model wrong, this **Targeted Attack** calculates specific perturbations that shift the image across the model's decision boundary into a *specific user-selected class*.

### Key Parameters:
- **Epsilon (ε)**: The "Noise Ceiling." It defines the maximum amount any single pixel can be changed from its original value. Lower epsilon = more stealthy attack.
- **Iterations**: The number of optimization steps. More steps allow the attack to find a more "convincing" path to the target class.

---

## 🏗️ Technical Architecture

### Backend (Python/FastAPI)
The backend serves as the mathematical engine, leveraging **PyTorch** for gradient calculation.
- **`main.py`**: The API layer handling the `/attack` and `/classes` endpoints.
- **`src/add_adversarial_noise.py`**: Contains the logic for the optimization loop. It uses the model's gradients with respect to the input image to "nudge" the pixels.
- **`src/utils.py`**: Handles ImageNet normalization logic and pre-trained model initialization.

### Frontend (React/Vite)
A modern dashboard built with **Tailwind CSS** that focuses on data visualization and educational clarity.
- **Interactive Controls**: Range sliders for ε and iterations, and a searchable dropdown for all 1,000 ImageNet classes.
- **Visual Comparison**: Side-by-side view of Original vs. Adversarial images.
- **Noise Analysis**: Visualizes the amplified "perturbation mask" to show exactly what the AI is seeing.
- **Confidence Tracking**: Dynamic bar charts showing the top-5 model predictions before and after the attack.

---

## 📁 Repository Structure

```text
adversarial-attack/
├── main.py                 # FastAPI Entry Point
├── src/                    # Core Python Logic
│   ├── add_adversarial_noise.py   # Attack implementation
│   ├── utils.py                   # Data/Model utilities
│   └── imagenet_dataset/          # Mapping for 1000 classes
├── frontend/               # React Dashboard
│   ├── src/
│   │   ├── App.tsx         # Main UI Logic
│   │   └── index.css       # Tailwind & Glassmorphism styles
│   └── vite.config.ts      # Frontend build config
└── requirements.txt        # Python dependencies
```

---

## 🛠️ Tech Stack
- **Model**: ResNet-18 (Pre-trained on ImageNet)
- **Frameworks**: PyTorch, FastAPI, React
- **Styling**: Tailwind CSS (Modern Dark/Light UI)
- **Language**: Python 3.9+, TypeScript
