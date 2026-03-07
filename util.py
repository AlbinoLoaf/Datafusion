import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import pywt
from skimage.feature import hog, local_binary_pattern
import copy

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
    # Cropping the image to zoom in the point of the drill
    cropped = image.crop((2000, 500, 5496, 3672))

    resized = cropped.resize((512, 512))
    gray_image = ImageOps.grayscale(resized)
    gray_array = np.array(gray_image)

    radius = 1
    n_points = 8 * radius

    lbp = local_binary_pattern(gray_array, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
    features = hist.astype(float) / (hist.sum() + 1e-7)

    return features


def get_HOG(image):
    """
    Compute Histogram of Oriented Gradients (HOG) for a given image.
    
    HOG captures edge and gradient structure by computing gradients, 
    creating histograms of gradient orientations, and normalizing them.
    
    Args:
        image: A PIL Image object (expected size: 5496x3672)

    Return:
        fd: vector of size 7200 
    """

    if image.size != (5496, 3672):
        raise ValueError(f"Expected image size (5496, 3672), got {image.size}")
    
    else:
        # Cropping the image to zoom in the point of the drill
        cropped = image.crop((2000, 500, 5496, 3672))
        # Rezising the image to speed up computation and reduce the size of output features
        resized = cropped.resize((512, 512))

        fd = hog(
            resized,
            orientations=8,
            pixels_per_cell=(32, 32),
            cells_per_block=(2, 2),
            channel_axis=-1,
            feature_vector=True
        )

    return fd


def process_set_images(set_path):
    """
    Loads the pickled DataFrame for a set, extracts unique image paths and wear values,
    computes HOG features for each unique image, and returns a new DataFrame with these features.
    """

    set = pd.read_pickle(set_path)
    
    # Select unique image paths and their corresponding wear values, dropping any rows with missing data
    unique_image_set = (
        set
        .drop_duplicates("image_path")
        .loc[:, ["image_path", "wear"]]
        .dropna()   
        .reset_index(drop=True)
    )

    hog_features = []
    for image_path in unique_image_set["image_path"]:
        with Image.open(image_path) as im:
            hog_feature = get_HOG(im) 
            hog_features.append(hog_feature)

    unique_image_set['hog_features'] = hog_features

    return unique_image_set


def extract_wavelet_packet_features(signal, wavelet='db4', level=3):
    """
    Extract Wavelet Packet Transform features.
    Provides finer frequency resolution than standard DWT.
    Good for detailed frequency analysis of 50-second signals.
    
    Args:
        signal: 1D numpy array
        wavelet: Wavelet type
        level: Decomposition level (3-4 for 50-second data)
        
    Returns:
        numpy.ndarray: Wavelet packet energy features
    """
    if len(signal) < 2**level:
        level = int(np.log2(len(signal))) - 1
        if level < 1:
            level = 1
    
    # Wavelet packet decomposition
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, maxlevel=level)
    
    # Get all nodes at the deepest level
    packet_names = [node.path for node in wp.get_level(level, 'natural')]
    
    # Extract energy from each frequency band
    energies = []
    for packet_name in packet_names:
        packet = wp[packet_name].data
        energy = np.sum(packet**2)
        energies.append(energy)
    
    # Normalize energies
    energies = np.array(energies)
    total_energy = np.sum(energies) + 1e-7
    normalized_energies = energies / total_energy
    
    return normalized_energies


def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    early_stopping_patience = 5
    best_valid_loss = float('inf')
    epochs_without_improvement = 0
    best_model_weights = None

    for epoch in range(1,epochs+1):
        print(f'- Epoch {epoch}')
        model.train()
        batch_losses=[]

        for _, data in enumerate(train_loader):
            x, y = data

            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            y = torch.reshape(y, (y.shape[0], 1))

            pred = model(x)

            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()

            batch_losses.append(loss.item())
            optimizer.step()
            
        train_losses.append(np.mean(batch_losses))
        print(f'-- Train-Loss : {train_losses[-1]}')

        model.eval()
        batch_losses=[]

        for i, data in enumerate(valid_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            y = torch.reshape(y, (y.shape[0], 1))

            pred = model(x)
            mse = loss_fn(pred, y)
            batch_losses.append(mse.item())

        valid_losses.append(np.mean(batch_losses))
        scheduler.step(valid_losses[-1])

        if valid_losses[-1] < best_valid_loss:
            best_valid_loss = valid_losses[-1]
            epochs_without_improvement = 0
            best_model_weights = copy.deepcopy(model.state_dict()) 
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            model.load_state_dict(best_model_weights)
            break

        print(f'-- Valid Loss : {valid_losses[-1]}\n')
        
    return valid_losses