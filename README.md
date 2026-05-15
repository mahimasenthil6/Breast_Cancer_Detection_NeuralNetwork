# Breast Cancer Classification System (Artificial Neural Network)

An end-to-end deep learning framework built to classify breast tumor cell masses as either **Malignant** or **Benign** based on cellular nuclei measurements [INDEX]. This project includes a standardized data processing system, a trained sequential neural network optimized on Apple Silicon, and a live web deployment pipeline [INDEX].

## 📊 Live Web Application Link
The predictive system is deployed globally and accessible for real-time diagnostic checks:
👉 **[Click Here to Open the Live Web Application](https://huggingface.co/spaces/Mahima6Teju/Breast_Cancer_NN)**

---

## 🔬 How the Project Works

### 1. Data Cleaning & Standardisation
* **The Problem:** Raw medical data has huge variance (e.g., cell area can be larger than 1,000, while cell smoothness is a tiny decimal like 0.1) [INDEX]. If fed unscaled, the model gets confused and fails to learn [INDEX].
* **The Solution:** Implemented Scikit-Learn's `StandardScaler` to shift and shrink all 30 features into a uniform scale between -3 and +3 [INDEX], allowing the model to train smoothly [INDEX].

### 2. Neural Network Brain (ANN)
* **Structure:** Built as a multi-layer fully connected sequential network (`keras.Sequential`) [INDEX].
* **Hidden Layer:** Features 20 artificial neurons using a **ReLU activation function** to catch complex data patterns [INDEX].
* **Output Layer:** Features 2 neurons backed by a **Sigmoid activation function** to calculate exact diagnosis percentages [INDEX].
* **Optimizer:** Powered by the **Adam optimizer** tied to a **Sparse Categorical Cross-Entropy loss function** for fast, smart error corrections [INDEX].

### 3. Cloud Deployment
* Designed a clean 3-column user dashboard using the **Streamlit web framework** [INDEX].
* Deployed the entire pipeline container securely on **Hugging Face Spaces** running on a stable Python 3.10 cloud server [INDEX].

---
*Disclaimer: This system is an educational machine learning prototype. It is not an alternative to licensed clinical diagnosis or professional medical consultation [INDEX].*
