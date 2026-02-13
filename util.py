import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from PIL import Image, ImageOps

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


def compute_LocalBinaryPattern(image):
    """
    Extract Local Binary Pattern (LBP) transformation of an image.
    
    LBP encodes local texture by comparing each pixel with its 8 neighbors,
    creating a binary pattern that's converted to a decimal value (0-255).
    
    Args:
        image: A PIL Image object (will be converted to grayscale)
        
    Returns:
        numpy.ndarray: LBP-transformed image of shape (height, width) with uint8 values
    """

    def get_value(image, center_value, x_, y_):
    	"""
		Returns a binary value for each neighbour of the current "center" pixel
    	"""

        try:
            if image[x_, y_] >= center_value:
                return 1

            else:
                return 0

        except:
            return 0


    def lpg(image, x, y):
    	"""
    	Compute LBP value for a single pixel by comparing with 8 neighbors.
    	"""

        values = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if not (i == 0 and j == 0):
                    v = get_value(image, image[x][y], x+i, y+j)
                    values.append(v)

        # Convert binary to decimal
        power = [1,2,4,8,16,32,64,128]
        value = 0
        for k in range(len(values)):
            value += values[k] * power[k]

        return value


    width, height  = image.size
    im = ImageOps.grayscale(image)
    im = np.asarray(im)
    img_lbp = np.zeros((height, width),np.uint8)

    for x in range(height-1):
        for y in range(width-1):
            img_lbp[x, y] = lpg(im, x, y)
            
    return img_lbp