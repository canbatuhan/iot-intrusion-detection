import os
import kagglehub
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

def get_dataset(source):
    path = kagglehub.dataset_download(source)
    return path

def integration(input_dir, output_file):
    all_data = []
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for file in filenames:
            if file.endswith('.csv'):
                file_path = os.path.join(dirpath, file)
                chunk_iter = pd.read_csv(file_path, chunksize=1000)
                for chunk in chunk_iter:
                    all_data.append(chunk)
    pd.concat(all_data, ignore_index=True).to_csv(output_file)
    
class DataMiningPipeline:
    def __init__(self, path, sample_size=None):
        self.data = pd.read_csv(path)
        if sample_size != None:
            n_samples = int(self.data.shape[0]*sample_size)
            self.data = self.data.sample(n_samples)

    def drop_unnecessary_columns(self, columns):
        self.data.drop(columns, axis=1, inplace=True)

    def data_cleaning(self):
        self.data.drop_duplicates(subset=None, keep="first", inplace=True)
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.data[col].fillna(
                    self.data[col].mode()[0], inplace=True)
            else:
                self.data[col].fillna(
                    self.data[col].mean(), inplace=True)

    def data_transformation(self, label_name):
        for col in self.data.columns:
            if self.data[col].dtype == 'object' and col != label_name:
                self.data[col] = self.data[col].astype(str)
                self.data[col] = LabelEncoder().fit_transform(self.data[col])
                
        scaler = StandardScaler()
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns
        self.data[numerical_columns] = scaler.fit_transform(self.data[numerical_columns])

    def data_reduction(self, n_components=10):
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(self.data[numerical_columns])
        reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(n_components)])
        non_numerical_columns = self.data.select_dtypes(exclude=[np.number])
        self.data = pd.concat([reduced_df, non_numerical_columns.reset_index(drop=True)], axis=1)

    def get_processed_data(self):
        return self.data