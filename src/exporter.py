import io
import pandas as pd

class Exporter:
    """
    Handles exporting of clustered data and visualizations
    Supports:
        - CSV export
        - PNG export from Plotly figures
    """

    @staticmethod
    def export_csv(df:pd.DataFrame) -> bytes:
        """
        Export DataFrame as CSV bytes for Streamlit download
        """
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            return csv_buffer.getvalue().encode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to export CSV: {e}")
        
    @staticmethod
    def export_png(fig) -> bytes:
        """
        Export a Plotly figure as PNG bytes
        """
        try:
            png_bytes = fig.to_image(format="png")
            return png_bytes
        except Exception as e:
            raise RuntimeError(
                f"Failed to export PNG"
                f"Ensure kaleido is installed ('pip install -U kaleido). Error: {e}"
            )