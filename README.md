# Planar Data Classification from Scratch (Shallow vs Deep Neural Networks)

This repository explores **binary classification on a non-linearly separable planar dataset**, implemented **from scratch using NumPy**.  
The goal is to understand **why depth matters in neural networks** by comparing a **shallow neural network** with a **deeper architecture**, without relying on high-level ML frameworks.

---

## Project Motivation

Linear models fail on non-linear decision boundaries.  
This project investigates:

- How neural networks learn **non-linear representations**
- The limitations of shallow models
- Why **adding depth improves expressiveness**
- How forward propagation, backpropagation, and gradient descent work internally

All models are built **step-by-step from first principles** to avoid treating neural networks as black boxes.

---

## Repository Contents

├── planar_shallow.ipynb # 1-hidden-layer neural network from scratch <br>
├── planar_deep.ipynb # Multi-layer neural network from scratch <br>
└── README.md

---

## Dataset

- **Synthetic planar dataset**
- Binary classification task
- Non-linearly separable
- Ideal for visualizing decision boundaries

The dataset is intentionally simple so the focus remains on **model behavior**, not data complexity.

---

## Models Implemented

### 1. Shallow Neural Network (`planar_shallow.ipynb`)
- Architecture:  
  **Input → Hidden Layer → Output**
- Activation: `tanh` (hidden), `sigmoid` (output) (also experimented with different activations and other hyperparameters)
- Trained using:
  - Forward propagation
  - Binary Cross-Entropy loss
  - Manual backpropagation
  - Gradient descent

**Key observation:**  
The shallow network struggles to form complex decision boundaries.

---

### 2. Deep Neural Network (`planar_deep.ipynb`)
- Architecture:  
  **Input → Multiple Hidden Layers → Output**
- Non-linear activations across layers
- Same training principles as the shallow model

**Key observation:**  
Adding depth allows the model to learn richer hierarchical representations and significantly improves classification performance.

---

## Training Pipeline (Both Models)

1. Parameter initialization  
2. Forward propagation  
3. Loss computation (Binary Cross-Entropy)  
4. Backpropagation (manual gradients)  
5. Parameter updates using gradient descent  
6. Visualization of decision boundary  

Every step is implemented explicitly to emphasize **how learning actually happens**.

---

## Results & Insights

- Shallow networks are limited in representational power
- Deep networks learn complex boundaries more effectively
- Depth matters even on small, synthetic datasets
- Understanding internals builds stronger intuition than using prebuilt APIs

---

## Tech Stack

- Python
- NumPy
- Matplotlib
- Jupyter Notebook

_No high-level ML libraries (TensorFlow / PyTorch / Scikit-learn models) were used for training._

---

## Learning Outcomes

- Solid intuition for neural network depth
- Hands-on understanding of backpropagation
- Clear mental model of decision boundaries
- Strong foundation for CNNs and deeper architectures

---

## Future Improvements

- Add regularization (L2, Dropout)
- Compare with logistic regression baseline
- Experiment with different activation functions
- Extend to multi-class classification

---

## Author

**Abhi**  
Aspiring Machine Learning & Deep Learning Practitioner  
Focused on learning by building models **from scratch** and understanding systems end-to-end.
