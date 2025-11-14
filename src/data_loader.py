import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path

class DataLoader():
    """
    Handles data loading and preprocessing for ClusterVision
    """
    def load_data(self, file):
        """
        Parameters: 
            file: Streamlit uploaded file object
        Returns:
            pd.DataFrame
        """

        try:
            df = pd.read_csv(file)
            if df.empty():
                raise ValueError("The uploaded file is empty")
            return df
        except Exception as e:
            raise ValueError(f"Failed to load file: {str(e)}")


    def process(self, df: pd.DataFrame, scaling_method="standard", drop_missing=True):
        """
        Preprocess dataset by:
            -Handling missing values
            -Onehot encoding categorical columns
            -Scaling numerical columns
        Parameters:
            df: Pandas DataFrame
            scaling_method: "standard", "minmax", or None
            drop_missing: Whether to drop rows with missing values

        Returns:
            Preprocessed pd.DataFrame    
        """

        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if drop_missing:
            df.dropna(inplace=True)
        else:
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
            df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

        #One hot encoding
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        if scaling_method == 'standard':
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        elif scaling_method == 'minmax':
            sclaer = MinMaxScaler()
            df[numerical_cols] = sclaer.fit_transform(df[numerical_cols])
        
        elif scaling_method is None:
            pass
        
        else:
            raise ValueError("Invalid scaling method: choose 'standard', 'minmax' or None ")
        
        if df.empty:
            raise ValueError("No usable features found after preprocessing")
        
        return df

