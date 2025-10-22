
## Neural Digit Recognizer 🎯


---

## 🌟 Project Overview

This intelligent web application leverages cutting-edge deep learning to instantly recognize handwritten digits. Built on a sophisticated Convolutional Neural Network architecture powered by PyTorch, the system delivers exceptional accuracy (98%+) through training on the comprehensive MNIST dataset. Experience seamless digit recognition through an elegant, user-friendly interface.

---

## 🧠 Project Preview

<div align="center">

  <!-- Live Demo Button -->
  <h1>🚀 <a href="https://neural-digit-recognizer-t47d.onrender.com/" target="_blank" style="text-decoration:none;">Live Demo</a></h1>

  <br>

  <!-- Divider Line -->
  <hr style="width:60%; border: 1px solid #d3d3d3;">

  <!-- Preview Section -->
  <h2>🎯 Preview</h2>
  <img src="https://github.com/dineshbarri/Neural_Digit_Recognizer/blob/main/images/preview.gif" alt="Digit Recognizer Demo" width="70%" style="border-radius:12px; border:2px solid #ccc;">

  <br><br>
  <em>🧠 Watch the Neural Digit Recognizer accurately classify handwritten digits in real-time!</em>

</div>

---

## ✨ Core Capabilities

### 🧠 **Advanced Neural Architecture**
- Deep CNN implementation with optimized layer configurations
- Comprehensive training pipeline with real-time performance monitoring
- Detailed analytics including confusion matrices and performance metrics

### 🖼️ **Intelligent Image Processing**
- Adaptive resizing and normalization algorithms
- Automatic color space conversion and inversion
- Sophisticated data augmentation for enhanced model resilience

### 🎨 **Modern Web Interface**
- **Live Canvas Drawing:** Sketch digits directly on an interactive canvas with instant predictions
- **File Upload Support:** Seamlessly process images of handwritten digits
- **Confidence Visualization:** View top-3 predictions with detailed probability scores
- **Responsive Design:** Sleek, minimalist UI optimized for all devices

### ⚙️ **Flexible Configuration**
- Centralized hyperparameter management via `config.py`
- Easy customization of model and application settings

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

**1. Clone the Repository**
```bash
git clone https://github.com/dineshbarri/Neural_Digit_Recognizer
cd Neural_Digit_Neural_Digit_Recognizer
```

**2. Install Required Dependencies**
```bash
pip install -r requirements.txt
```
*All essential packages including PyTorch, Flask, and image processing libraries will be installed automatically.*

**3. Train the Neural Network** *(Recommended for optimal performance)*
```bash
python train_model.py
```
*This process will:*
- Train the CNN on MNIST dataset
- Generate `mnist_model.pth` with optimized weights
- Create performance analytics (training/validation metrics, confusion matrix)

**4. Launch the Application**
```bash
python flask_app.py
```
*Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to start recognizing digits!*

---

## 🐳 Docker Deployment

### Containerized Setup for Production

**Build the Docker Image:**
```bash
docker build -t -neural_digit_Recognizer .
```

**Run the Container:**
```bash
docker run -p 5000:5000 neural-digit-recognizer
```

Access the application at `http://localhost:5000`

---

## 📁 Architecture & Structure

```
Neural_Digit_Recognizer/
│
├── 🌐 flask_app.py              # Web server and API endpoints
├── 🎓 train_model.py             # Neural network training pipeline
├── 🏗️  model.py                  # CNN architecture definition
├── ⚙️  config.py                 # Configuration management
├── 🧠 mnist_model.pth            # Trained model checkpoint
├── 📦 requirements.txt           # Python dependencies
├── 📖 README.md                  # Documentation
├── 🐳 Dockerfile                 # Container specification
├── 🚫 .dockerignore              # Docker build exclusions
│
├── 📊 Analytics Output
│   ├── confusion_matrix.csv     # Model performance matrix
│   ├── train_metrics.csv        # Training history
│   └── val_metrics.csv          # Validation metrics
├──     Procfile                 # tell the deployment platform(render) how to start your web application server
├──     images/
|          └── preview.gif       # output preview
├── 🎨 templates/
│   └── index.html               # Main web interface
│
└── 📂 static/
    ├── css/
    │   └── style.css            # Styling and animations
    └── js/
        └── canvas.js            # Interactive canvas logic
```

---

## 🎯 Performance Metrics

- **Validation Accuracy:** ~98%
- **Model Architecture:** Multi-layer CNN with dropout regularization
- **Training Dataset:** 60,000 MNIST images
- **Inference Speed:** Real-time prediction (<100ms)

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Deep Learning** | PyTorch |
| **Web Framework** | Flask |
| **Image Processing** | PIL, OpenCV |
| **Frontend** | HTML5 Canvas, Vanilla JS |
| **Containerization** | Docker |

---

## 📈 Future Enhancements

- [ ] Multi-language digit recognition
- [ ] REST API with authentication
- [ ] Model versioning and A/B testing
- [ ] Enhanced visualization dashboard
- [ ] Mobile app deployment

---

## 👤 Author

 #### &nbsp; Dinesh Barri

-  [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/dineshbarri)
-  [![Linkedin](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/dinesh-barri-7654b010b)


---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to enhance the project.

---

## 📝 License

This project is open source and available for educational and commercial use.

---

**Built with ❤️ using PyTorch and Flask**









