from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

class ClusteringEngine:
    """
    Performs K-Means clustering and computes evaluation metrics
    """
    def fit(self, data:pd.DataFrame, n_clusters: int):
        try:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = model.fit_predict(data)
            return pd.Series(cluster_labels, name="Cluster_label")
        except Exception as e:
            raise RuntimeError(f"K-Means failed: {e}")
        
    def compute_metric(self, data: pd.DataFrame, max_k: int = 10):
        """
        Compute inertia (Elbow method) and Silhoutte scores

        max_k: maximum k value to evaluate
        """
        inertia_list = []
        silhouette_list = []
        
        for k in range(2, max_k+1):
            try:
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                cluster_labels = km.fit_predict(data)
                inertia_list.append(km.inertia_)
                silhouette_list.append(silhouette_score(data, cluster_labels))
            except Exception as e:
                inertia_list.append(None)
                silhouette_list.append(None)
                print(f"Warning K-Means failed for k = {k}: {e}")

            metrics_dict = {"inertia": inertia_list, "silhouette": silhouette_list}

            return metrics_dict
