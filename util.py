import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from PIL import Image

#ML bs 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,matthews_corrcoef
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_sensor_file(file_path, set_name, image_map, wear_map):
    """
    Loads a single sensor CSV, attaches metadata, and links it to an image.

    Args:
        file_path (str): Full path to the sensor .csv file.
        set_name (str): The identifier for the set (e.g., 'Set1').
        image_map (dict): Mapping of {filename: image_path}.
        wear_map (dict): Mapping of {filename: wear_value}.

    Returns:
        pd.DataFrame: A dataframe containing sensor readings + metadata.
        None: If the file is empty or cannot be read.
    """
    try:
        # Load Raw Data
        df = pd.read_csv(file_path, header=None, 
                         names=['accel', 'acoustic', 'force_x', 'force_y', 'force_z', 'timestamp'])
        
        filename = os.path.basename(file_path)
        
        raw_image_path = image_map.get(filename, None)
        if raw_image_path and isinstance(raw_image_path, str):
            df['image_path'] = raw_image_path.replace('MATWI', 'data')
        else:
            df['image_path'] = None

        # Add Metadata
        df['wear'] = wear_map.get(filename, None)
        df['source_file'] = filename
        df['set_id'] = set_name
        
        return df

    except pd.errors.EmptyDataError:
        return None
    


def process_set_batch(folder_path, set_name, image_map, wear_map, output_dir):
    """
    Aggregates all sensor files in a folder into a single DataFrame and pickles it.

    Args:
        folder_path (str): Path to the 'sensordata' folder.
        set_name (str): Name of the set (e.g., 'Set1').
        image_map (dict): Lookup table for images.
        wear_map (dict): Lookup table for wear labels.
        output_dir (str): Where to save the final .pkl file.

    Returns:
        bool: True if data was found and saved, False otherwise.
    """
    sensor_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    set_data = []
    
    print(f"[{set_name}] Found {len(sensor_files)} files.")

    for file in sensor_files:
        full_path = os.path.join(folder_path, file)
        df = load_sensor_file(full_path, set_name, image_map, wear_map)
        
        if df is not None:
            set_data.append(df)
            
    if set_data:
        # Merge and Save
        final_df = pd.concat(set_data, ignore_index=True)
        output_path = os.path.join(output_dir, f"{set_name}_linked.pkl")
        final_df.to_pickle(output_path)
        
        print(f"[{set_name}] Success. Saved {len(final_df)} rows to {output_path}")
        
        # Return True to signal success
        return True
    
    return False


def timeseries(dataset):
    """
    Plots the 5 sensor channels (accel, acoustic, force_x, force_y, force_z)
    from the provided DataFrame as vertically stacked subplots.
    """
    sensors = ['accel', 'acoustic', 'force_x', 'force_y', 'force_z']
    fig, axes = plt.subplots(len(sensors), 1, figsize=(12, 10), sharex=True)
    
    for ax, sensor in zip(axes, sensors):
        # Plot data; using .values avoids potential index issues if not sorted
        ax.plot(dataset[sensor].values) 
        ax.set_title(f"Sensor: {sensor}")
        ax.set_ylabel('Value')
        ax.grid(True)
        
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.show()