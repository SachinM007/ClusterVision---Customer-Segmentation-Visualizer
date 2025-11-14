import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class Visualizer:
    """
    Generates interactive cluster visualizations and evaluation plots
    """
    @staticmethod
    def plot_clusters_2d(df: pd.DataFrame, labels, x_col: str, y_col: str):
        """
        Creates a 2D interactive scatter plot for cluster visualization
        """
        df_plot = df.copy()
        df_plot['cluster'] = labels

        fig = px.scatter(
            df_plot,
            x=x_col,
            y=y_col,
            color=df_plot['cluster'].astype(str),
            title="2D Cluster Visualization",
            hover_data=df_plot.columns
        )

        return fig
    
    @staticmethod
    def plot_clusters_3d(df: pd.DataFrame, labels, x_col: str, y_col: str, z_col: str):
        """
        Creates a 3D interactive cluster plot for cluster visualization
        """
        df_plot = df.copy()
        df_plot['cluster'] = labels

        fig = px.scatter_3d(
            df_plot,
            x=x_col,
            y=y_col,
            title="3D Cluster Visualization",
            hover_data=df_plot.columns
        )

        return fig

    @staticmethod
    def plot_elbow(k_values, inertia_values):
        """
        Creates an Elbow Method plot (k vs inertia)
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=k_values,
            y=inertia_values,
            mode='lines+markers',
            line=dict(color='blue'),
            marker=dict(size=8),
            name="Inertia"
        ))

        fig.update_layout(
            title="Elbow Method (k vs Inertia)",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia",
            template="plotly_white"
        )

        return fig
    
    @staticmethod
    def plot_silhouette(k_values, silhouette_values):
        """
        Creates silhoutte score plot (k vs silhoutte score)
        """

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=k_values,
            y=silhouette_values,
            mode='lines+markers',
            line=dict(color='green'),
            marker=dict(size=8),
            name="Silhouette Score"
        ))

        fig.update_layout(
            title="Silhouette Analysis (k vs Score)",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Silhouette Score",
            template="plotly_white"
        )

        return fig