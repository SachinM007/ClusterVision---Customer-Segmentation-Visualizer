import streamlit as st
import pandas as pd

from src.data_loader import DataLoader
from src.clustering_engine import ClusteringEngine
from src.visualizer import Visualizer
from src.exporter import Exporter

st.set_page_config(page_title="ClusterVision", layout="wide")

# ---------------------------------------------
# Initialize Session State
# ---------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "processed_df" not in st.session_state:
    st.session_state.processed_df = None

if "labels" not in st.session_state:
    st.session_state.labels = None

# ---------------------------------------------
# Sidebar Controls
# ---------------------------------------------
st.sidebar.header("🔧 Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])

scaling_method = st.sidebar.selectbox(
    "Scaling Method",
    ["standard", "minmax", "none"],
    index=0
)

missing_strategy = st.sidebar.selectbox(
    "Missing Value Handling",
    ["drop rows", "impute"],
    index=0
)

st.sidebar.markdown("---")

compute_metrics_btn = st.sidebar.button("Compute Elbow & Silhouette")
run_cluster_btn = st.sidebar.button("Run Clustering")

st.sidebar.markdown("---")
st.sidebar.write("Visualization Options")

plot_type = st.sidebar.radio("Plot Type", ["2D", "3D"])

# ---------------------------------------------
# Main Application
# ---------------------------------------------
st.title("🧠 ClusterVision - Customer Segmentation Visualizer")

# 1. Load Data
if uploaded_file:
    st.session_state.df = DataLoader().load_data(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(st.session_state.df.head())

    st.markdown("---")

    # 2. Preprocessing
    st.subheader("⚙️ Preprocessing")

    dl = DataLoader()

    st.write("Detected Columns:")
    st.write("Numeric:", dl.numeric_cols)
    st.write("Categorical:", dl.categorical_cols)

    if scaling_method == "none":
        scaling = None
    else:
        scaling = scaling_method

    drop_missing = (missing_strategy == "drop rows")

    st.session_state.processed_df = dl.preprocess(
        st.session_state.df,
        scaling_method=scaling,
        drop_missing=drop_missing
    )

    st.success("Data successfully preprocessed!")

    # 3. Feature selection
    st.subheader("🎯 Feature Selection")
    all_columns = st.session_state.processed_df.columns.tolist()

    selected_features = st.multiselect(
        "Select features for clustering:",
        options=all_columns,
        default=all_columns
    )

    if len(selected_features) < 2:
        st.warning("Please select at least 2 features.")
        st.stop()

    df_selected = st.session_state.processed_df[selected_features]

    # 4. Evaluation: Elbow & Silhouette
    st.markdown("---")
    st.subheader("📉 Elbow & Silhouette Evaluation")

    if compute_metrics_btn:
        ce = ClusteringEngine()
        metrics = ce.compute_metrics(df_selected, max_k=10)

        k_values = list(range(2, 11))

        # Elbow Plot
        st.plotly_chart(
            Visualizer.plot_elbow(k_values, metrics["inertia"]),
            use_container_width=True
        )

        # Silhouette Plot
        st.plotly_chart(
            Visualizer.plot_silhouette(k_values, metrics["silhouette"]),
            use_container_width=True
        )

    # 5. Run Clustering
    st.markdown("---")
    st.subheader("🔮 Run K-Means Clustering")

    k_value = st.slider("Select number of clusters (k):", 2, 10, 3)

    if run_cluster_btn:
        ce = ClusteringEngine()
        labels = ce.fit(df_selected, n_clusters=k_value)
        st.session_state.labels = labels

        st.subheader("Clustered Data")
        result_df = st.session_state.df.copy()
        result_df["cluster_label"] = labels
        st.dataframe(result_df.head())

        st.success("Clustering complete!")

    # 6. Visualization
    if st.session_state.labels is not None:
        st.markdown("---")
        st.subheader("Cluster Visualization")

        if plot_type == "2D":
            x_axis = st.selectbox("X-axis:", selected_features)
            y_axis = st.selectbox("Y-axis:", selected_features)

            fig = Visualizer.plot_clusters_2d(
                df_selected, st.session_state.labels, x_axis, y_axis
            )
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "3D":
            x_axis = st.selectbox("X-axis:", selected_features)
            y_axis = st.selectbox("Y-axis:", selected_features)
            z_axis = st.selectbox("Z-axis:", selected_features)

            fig = Visualizer.plot_clusters_3d(
                df_selected, st.session_state.labels, x_axis, y_axis, z_axis
            )
            st.plotly_chart(fig, use_container_width=True)

        # 7. Export
        st.markdown("---")
        st.subheader("📥 Export Results")

        # Export CSV
        export_df = st.session_state.df.copy()
        export_df["cluster_label"] = st.session_state.labels

        csv_bytes = Exporter.export_csv(export_df)
        st.download_button(
            label="Download Clustered Data (CSV)",
            data=csv_bytes,
            file_name="clustered_data.csv",
            mime="text/csv"
        )

        # Export PNG
        fig_png = Exporter.export_png(fig)
        st.download_button(
            label="Download Visualization (PNG)",
            data=fig_png,
            file_name="cluster_plot.png",
            mime="image/png"
        )
