# 🧠 Neural Architecture Comparison: Manual Core vs. Optimized API
### *Fashion-MNIST Classification: A Deep Dive into Optimization and Topology*

---

## 📊 Performance Summary
This project demonstrates the impact of **Bayesian Hyperparameter Optimization** over a standard fixed-architecture neural network. By comparing a manual NumPy implementation with an Optuna-tuned Scikit-Learn pipeline, we observe a significant leap in both accuracy and classification clarity.

| Metric | Custom Core (Manual) | API + Optuna (Optimized) |
| :--- | :--- | :--- |
| **Accuracy** | 81.75% | **89.62%** |
| **Activation** | Sigmoid (Fixed) | **Tanh (Optimized)** |
| **Hidden Nodes** | 200 (Static) | **113 (Optimized)** |
| **Logic** | Baseline Math | Bayesian Intelligence |

---

## 🗺 3D Topological Analysis
To analyze the results, I transformed the **Confusion Matrix** into a 3D surface. This visualizes the **Inter-class Confusion** of the networks.

* **Sharp Diagonal Peaks:** Represent high precision and recall. In the API model, these peaks are significantly higher, indicating a more robust separation of feature manifolds.
* **Off-Diagonal Noise (The "Hills"):** These represent systematic misclassifications. In the Core model, these "hills" are more prominent between similar categories (e.g., T-shirt vs. Shirt), showing where the model's feature extraction logic fails to differentiate textures.

> **🕹 Interactive Experience:** Click on the images below to open the interactive 3D landscapes in your browser.

| **Custom Core Implementation** | **Optimized API Pipeline** |
| :---: | :---: |
| [![Core](results/topologies/core_fashion_topology.png)](https://dalliya.github.io/core-vs-api-neural-networks/results/topologies/core_fashion_topology.html) | [![API](results/topologies/api_fashion_topology.png)](https://dalliya.github.io/core-vs-api-neural-networks/results/topologies/api_fashion_topology.html) |
| *81.75% Accuracy. Note the broader "noise" in the valleys.* | *89.62% Accuracy. Note the sharper, more distinct peaks.* |

---

## 🔬 The Core vs. API Battle

### 1. Manual Core (Baseline)
* **Architecture:** 3-layer MLP based on Tariq Rashid's methodology.
* **Observation:** The model reached **81.75%**. While robust, the fixed sigmoid activation and manual learning rate limit its ability to capture the complex textures of the Fashion-MNIST dataset.
* **Topology:** The 3D landscape shows broader **"error valleys"**, indicating more frequent confusion between similar categories (e.g., shirts vs. coats).

### 2. API + Optuna (Optimized)
* **Architecture:** Scikit-Learn `MLPClassifier`.
* **Strategy:** **Optuna** executed 10 trials to find the mathematical "sweet spot" using a TPE Sampler.
* **Discovery:** It found that **113 nodes** and **Tanh** activation outperformed the larger 200-node baseline.
* **Result:** **89.62% Accuracy**. This proves that "more neurons" doesn't always mean a "better model"—the right *balance* does.

---

## 🛠 Engineering Takeaways
* **Hyperparameter Power:** The choice of activation function (Tanh vs Sigmoid) and precise hidden layer sizing—automated by Optuna—accounted for a **~8% increase** in performance.
* **Scalability:** Modern API frameworks combined with HPO (Hyperparameter Optimization) provide a more scalable and accurate path for complex image recognition tasks than static manual implementations.
* **Visual Debugging:** Using 3D topologies helps in identifying **"systemic confusion"** between specific categories that a simple accuracy percentage might hide.

---

## 📂 Project Setup
* **Core Engine:** `/core/custom_network.py`
* **API Pipeline:** `/api/sklearn_pipeline.py`
* **Visualization Engine:** `/visualizations/plotter.py`
* **Data Loader:** `/utils/data_loader.py`

---

## 👩‍💻 About the Author & Acknowledgments
**Developed by Dariia Zhdanova (@Dalliya)** | *ML Explorer & Architect of the 3D Neural Topology Landscapes.*

> "In this study, I transitioned from manual mathematical foundations to automated Bayesian optimization, proving that even a 7.87% increase in accuracy is a journey of thousands of hidden neural connections."