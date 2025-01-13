import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class OutlierDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.lof = LocalOutlierFactor(contamination=0.1, novelty=True)
        
    def iqr_outliers(self, data, column, threshold=1.5):
        """
        Detect outliers using the Interquartile Range (IQR) method
        
        Parameters:
        data (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        threshold (float): IQR multiplier for outlier detection
        
        Returns:
        pd.Series: Boolean series indicating outliers
        """
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
        
        # Add bounds to class attributes for reporting
        self.iqr_bounds = {
            'lower': lower_bound,
            'upper': upper_bound,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR
        }
        
        return outliers
    
    def zscore_outliers(self, data, column, threshold=3):
        """
        Detect outliers using Z-score method
        
        Parameters:
        data (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        threshold (float): Number of standard deviations for outlier detection
        
        Returns:
        pd.Series: Boolean series indicating outliers
        """
        z_scores = np.abs(stats.zscore(data[column], nan_policy='omit'))
        return pd.Series(z_scores > threshold, index=data.index)
    
    def isolation_forest_outliers(self, data, columns):
        """
        Detect outliers using Isolation Forest
        
        Parameters:
        data (pd.DataFrame): Input DataFrame
        columns (list): List of columns to use for outlier detection
        
        Returns:
        pd.Series: Boolean series indicating outliers
        """
        # Fit and predict
        self.isolation_forest.fit(data[columns])
        predictions = self.isolation_forest.predict(data[columns])
        
        # Convert to boolean (True for outliers)
        return pd.Series(predictions == -1, index=data.index)
    
    def lof_outliers(self, data, columns):
        """
        Detect outliers using Local Outlier Factor
        
        Parameters:
        data (pd.DataFrame): Input DataFrame
        columns (list): List of columns to use for outlier detection
        
        Returns:
        pd.Series: Boolean series indicating outliers
        """
        # Fit and predict
        self.lof.fit(data[columns])
        predictions = self.lof.predict(data[columns])
        
        # Convert to boolean (True for outliers)
        return pd.Series(predictions == -1, index=data.index)
    
    def mahalanobis_outliers(self, data, columns, threshold=0.975):
        """
        Detect outliers using Mahalanobis distance
        
        Parameters:
        data (pd.DataFrame): Input DataFrame
        columns (list): List of columns to use for outlier detection
        threshold (float): Chi-square distribution threshold
        
        Returns:
        pd.Series: Boolean series indicating outliers
        """
        # Calculate mean and covariance
        X = data[columns]
        mean = np.mean(X, axis=0)
        cov = np.cov(X.T)
        
        # Calculate Mahalanobis distance
        inv_covmat = np.linalg.inv(cov)
        center_scaled = X - mean
        distances = np.sqrt(np.sum(np.dot(center_scaled, inv_covmat) * center_scaled, axis=1))
        
        # Get threshold from chi-square distribution
        cutoff = stats.chi2.ppf(threshold, df=len(columns))
        
        return pd.Series(distances > cutoff, index=data.index)
    
    def ensemble_outliers(self, data, columns, threshold=0.5):
        """
        Combine multiple outlier detection methods
        
        Parameters:
        data (pd.DataFrame): Input DataFrame
        columns (list): List of columns to use for outlier detection
        threshold (float): Proportion of methods that must agree for outlier classification
        
        Returns:
        pd.DataFrame: DataFrame with outlier flags and ensemble result
        """
        results = pd.DataFrame(index=data.index)
        
        # Apply each method
        for column in columns:
            results[f'iqr_{column}'] = self.iqr_outliers(data, column)
            results[f'zscore_{column}'] = self.zscore_outliers(data, column)
        
        results['isolation_forest'] = self.isolation_forest_outliers(data, columns)
        results['lof'] = self.lof_outliers(data, columns)
        results['mahalanobis'] = self.mahalanobis_outliers(data, columns)
        
        # Calculate ensemble result
        results['outlier_score'] = results.mean(axis=1)
        results['is_outlier'] = results['outlier_score'] > threshold
        
        return results
    
    def get_outlier_summary(self, data, columns):
        """
        Generate summary statistics for outliers
        
        Parameters:
        data (pd.DataFrame): Input DataFrame
        columns (list): List of columns to analyze
        
        Returns:
        dict: Summary statistics for outliers
        """
        results = self.ensemble_outliers(data, columns)
        
        summary = {
            'total_records': len(data),
            'total_outliers': results['is_outlier'].sum(),
            'outlier_percentage': (results['is_outlier'].mean() * 100),
            'method_agreement': {
                'high_agreement': (results['outlier_score'] > 0.8).sum(),
                'medium_agreement': ((results['outlier_score'] > 0.5) & (results['outlier_score'] <= 0.8)).sum(),
                'low_agreement': ((results['outlier_score'] > 0.2) & (results['outlier_score'] <= 0.5)).sum()
            }
        }
        
        return summary
