# 🧠 ClusterVision  
### Customer Segmentation Visualizer (K-Means + Interactive Visuals)

ClusterVision is an open-source Streamlit tool for interactive unsupervised customer segmentation using K-Means.  
It supports:

✔ CSV data upload  
✔ Preprocessing (missing values, scaling, one-hot encoding)  
✔ Elbow & Silhouette evaluation  
✔ 2D/3D cluster visualizations  
✔ Export results (CSV & PNG)  

---

# 🚀 Features

- Upload customer datasets (CSV)
- Automatic detection of numeric & categorical columns
- One-hot encoding for categorical variables
- Scaling (StandardScaler / MinMax / None)
- Elbow method (k vs inertia)
- Silhouette analysis (k vs score)
- 2D & 3D cluster plots using Plotly
- Export clustered data and visualization

---

# 📦 Installation

### Option 1 — Local Setup

git clone https://github.com/your-username/clustervision.git
cd clustervision
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

Option 2 — Docker

Build image:

docker build -t clustervision .


Run container:

docker run -p 8501:8501 clustervision

🔧 Project Structure
clustervision/
├── app.py
├── modules/
│   ├── data_loader.py
│   ├── clustering_engine.py
│   ├── visualizer.py
│   ├── exporter.py
├── requirements.txt
├── Dockerfile
└── README.md

