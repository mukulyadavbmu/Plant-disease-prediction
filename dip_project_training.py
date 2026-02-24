# Core libraries for data handling and modeling
import os
os.makedirs('output_plots', exist_ok=True)  # Create directory for plots

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import json
import pandas as pd

# Libraries for Digital Image Processing
import cv2

# Libraries for visualization and evaluation
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Kaggle-specific library for dataset download
import kagglehub

print("TensorFlow Version:", tf.__version__)
print("OpenCV Version:", cv2.__version__)

"""Download and Load Dataset"""

# --- 1. Foundational Setup and Data Exploration ---

# Define image size and batch size
IMG_SIZE = 128
BATCH_SIZE = 32

# Download the dataset using KaggleHub
dataset_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print(f"Dataset downloaded to: {dataset_path}")

# Construct the correct paths
base_dir = os.path.join(dataset_path, 'New Plant Diseases Dataset(Augmented)', 'New Plant Diseases Dataset(Augmented)')
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')

# Setup ImageDataGenerator for training with augmentation
# This will handle resizing, normalization, and all the augmentations you requested.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Validation data should only be rescaled, not augmented
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
# These will feed data from the directory to our model in batches.
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Keep shuffle false for consistent evaluation
)

# Get class names and number of classes
class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)
print(f"\nFound {num_classes} classes.")

"""Exploratory Data Analysis (EDA) 📊"""

# --- Graph 10: Visual Overview of All Classes ---

plt.figure(figsize=(20, 35))
plt.suptitle("Sample Image from Each of the 38 Classes", fontsize=24, y=0.93)

# Since we have 38 classes, a 10x4 grid works well
for i, class_name in enumerate(class_names):
    ax = plt.subplot(10, 4, i + 1)

    # Find the first image in the class directory
    class_dir = os.path.join(train_dir, class_name)
    img_file = os.listdir(class_dir)[0]
    img = cv2.imread(os.path.join(class_dir, img_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.title(class_name.replace("_", " "), fontsize=10)
    plt.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.91])
plt.savefig('output_plots/01_all_classes_overview.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: output_plots/01_all_classes_overview.png")

# --- 1. (Continued) Exploratory Data Analysis ---

# Let's get the number of images for each class
class_counts = {}
for class_name in class_names:
    class_dir = os.path.join(train_dir, class_name)
    class_counts[class_name] = len(os.listdir(class_dir))

# Convert to a Pandas DataFrame for easier plotting
df_class_counts = pd.DataFrame(class_counts.items(), columns=['Class', 'Count']).sort_values('Count', ascending=False)

# --- Graph 1: Bar Plot of Class Distribution ---
plt.figure(figsize=(12, 15))
sns.barplot(x='Count', y='Class', data=df_class_counts, palette='viridis')
plt.title('Number of Images per Class in the Training Set', fontsize=16)
plt.xlabel('Number of Images', fontsize=12)
plt.ylabel('Disease Class', fontsize=12)
plt.savefig('output_plots/02_class_distribution.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: output_plots/02_class_distribution.png")

# --- FIX: Find sample images automatically ---
healthy_class_dir = os.path.join(train_dir, 'Apple___healthy')
diseased_class_dir = os.path.join(train_dir, 'Apple___Cedar_apple_rust')

# Get lists of all images in those directories
healthy_files = os.listdir(healthy_class_dir)
diseased_files = os.listdir(diseased_class_dir)

# Check if both directories contain images before proceeding
if healthy_files and diseased_files:
    # Build full paths to the first image found in each directory
    healthy_path = os.path.join(healthy_class_dir, healthy_files[0])
    diseased_path = os.path.join(diseased_class_dir, diseased_files[0])

    healthy_img = cv2.cvtColor(cv2.imread(healthy_path), cv2.COLOR_BGR2RGB)
    diseased_img = cv2.cvtColor(cv2.imread(diseased_path), cv2.COLOR_BGR2RGB)

    # --- Graph 5 & 6: RGB Channel Histograms ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ('r', 'g', 'b')

    # Healthy Leaf
    axes[0, 0].imshow(healthy_img)
    axes[0, 0].set_title('Healthy Leaf')
    axes[0, 0].axis('off')

    for i, color in enumerate(colors):
        hist = cv2.calcHist([healthy_img], [i], None, [256], [0, 256])
        axes[0, 1].plot(hist, color=color, label=f'{color.upper()} channel')
    axes[0, 1].set_title('Healthy Leaf RGB Histograms')
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    # Diseased Leaf
    axes[1, 0].imshow(diseased_img)
    axes[1, 0].set_title('Diseased Leaf')
    axes[1, 0].axis('off')

    for i, color in enumerate(colors):
        hist = cv2.calcHist([diseased_img], [i], None, [256], [0, 256])
        axes[1, 1].plot(hist, color=color, label=f'{color.upper()} channel')
    axes[1, 1].set_title('Diseased Leaf RGB Histograms')
    axes[1, 1].set_xlabel('Pixel Intensity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('output_plots/03_rgb_histograms.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: output_plots/03_rgb_histograms.png")

else:
    print("Error: Could not find images in one or both of the specified directories.")
    print(f"Files in healthy dir: {len(healthy_files)}")
    print(f"Files in diseased dir: {len(diseased_files)}")

# --- FIX: Load the specific images needed for the Hue plots in this cell ---
# We need one healthy and one diseased image for the second part of this cell.
try:
    healthy_class_dir = os.path.join(train_dir, 'Apple___healthy')
    diseased_class_dir = os.path.join(train_dir, 'Grape___Black_rot') # Using one of the classes from above

    healthy_files = os.listdir(healthy_class_dir)
    diseased_files = os.listdir(diseased_class_dir)

    healthy_img = cv2.cvtColor(cv2.imread(os.path.join(healthy_class_dir, healthy_files[0])), cv2.COLOR_BGR2RGB)
    diseased_img = cv2.cvtColor(cv2.imread(os.path.join(diseased_class_dir, diseased_files[0])), cv2.COLOR_BGR2RGB)

    # Set a flag that images were loaded successfully
    images_loaded_for_hue = True
except Exception as e:
    print(f"Could not load images for Hue plot, skipping this visualization. Error: {e}")
    images_loaded_for_hue = False

# --- Graph 7: Box Plot of Average Brightness ---
def get_avg_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.mean()

# Sample brightness from a few classes
brightness_data = {'Class': [], 'Brightness': []}
classes_to_check = ['Apple___healthy', 'Grape___Black_rot', 'Potato___Late_blight']

for class_name in classes_to_check:
    class_dir = os.path.join(train_dir, class_name)
    for img_file in os.listdir(class_dir)[:50]: # Sample 50 images
        img = cv2.imread(os.path.join(class_dir, img_file))
        if img is not None:
            brightness_data['Class'].append(class_name)
            brightness_data['Brightness'].append(get_avg_brightness(img))

df_brightness = pd.DataFrame(brightness_data)

plt.figure(figsize=(12, 6))
sns.boxplot(x='Class', y='Brightness', data=df_brightness)
plt.title('Average Image Brightness Distribution by Class', fontsize=16)
plt.xticks(rotation=15)
plt.savefig('output_plots/04_brightness_distribution.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: output_plots/04_brightness_distribution.png")

# --- Graph 8 & 9: Hue Channel Distribution ---
# Only run this part if the images were loaded correctly
if images_loaded_for_hue:
    # Convert to HSV
    healthy_hsv = cv2.cvtColor(healthy_img, cv2.COLOR_RGB2HSV)
    diseased_hsv = cv2.cvtColor(diseased_img, cv2.COLOR_RGB2HSV)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(healthy_hsv[:, :, 0].flatten(), bins=50, kde=True, color='green')
    plt.title('Hue Distribution for Healthy Leaf')

    plt.subplot(1, 2, 2)
    sns.histplot(diseased_hsv[:, :, 0].flatten(), bins=50, kde=True, color='brown')
    plt.title('Hue Distribution for Diseased Leaf')
    plt.savefig('output_plots/05_hue_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: output_plots/05_hue_distribution.png")

"""Image Processing & Segmentation Showcase"""

# Helper function to display multiple images for comparison
def display_images(images, titles, rows, cols):
    plt.figure(figsize=(15, 4 * rows))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        # Use a grayscale colormap for single-channel images
        cmap = 'gray' if len(image.shape) == 2 else None
        plt.imshow(image, cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    # Generate filename from function call
    import hashlib
    filename = f"output_plots/processing_{hashlib.md5(str(titles).encode()).hexdigest()[:8]}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# Load a consistent sample image for all demonstrations
# Ensure the path is correct by finding an image programmatically
sample_class_dir = os.path.join(train_dir, 'Apple___Apple_scab')
image_files = os.listdir(sample_class_dir)
if image_files:
    sample_path = os.path.join(sample_class_dir, image_files[0])
    sample_img_bgr = cv2.imread(sample_path)
    sample_img_rgb = cv2.cvtColor(sample_img_bgr, cv2.COLOR_BGR2RGB)
    print("Sample image loaded successfully for preprocessing showcase.")
else:
    print("Error: Could not load a sample image.")

# --- Advanced Filtering and Sharpening ---

# 1. Bilateral Filter (Edge-Preserving Denoising)
bilateral_filtered = cv2.bilateralFilter(sample_img_rgb, d=9, sigmaColor=75, sigmaSpace=75)

# 2. Non-Local Means Denoising
# This is computationally more intensive but very effective.
nl_means_denoised = cv2.fastNlMeansDenoisingColored(sample_img_rgb, None, 10, 10, 7, 21)

# 3. Unsharp Masking (Sharpening)
gaussian_blur = cv2.GaussianBlur(sample_img_rgb, (9, 9), 10.0)
unsharp_masked = cv2.addWeighted(sample_img_rgb, 1.5, gaussian_blur, -0.5, 0)

# Display the results
display_images(
    [sample_img_rgb, bilateral_filtered, nl_means_denoised, unsharp_masked],
    ['Original', 'Bilateral Filter', 'Non-Local Means Denoising', 'Unsharp Masking (Sharpened)'],
    1, 4
)

# --- Advanced Color Space Transforms ---

# 4. L*a*b* Color Space
lab_img = cv2.cvtColor(sample_img_rgb, cv2.COLOR_RGB2LAB)
l_channel, a_channel, b_channel = cv2.split(lab_img)

# 5. YCrCb Color Space
ycrcb_img = cv2.cvtColor(sample_img_rgb, cv2.COLOR_RGB2YCrCb)
y_channel, cr_channel, cb_channel = cv2.split(ycrcb_img)

# 6. Simple White Balancing (Gray World Assumption)
# Calculate the average of each channel
r, g, b = cv2.split(sample_img_rgb.astype("float32"))
r_avg, g_avg, b_avg = cv2.mean(r)[0], cv2.mean(g)[0], cv2.mean(b)[0]
# Calculate the overall average
gray_avg = (r_avg + g_avg + b_avg) / 3
# Apply the correction factors
r = cv2.multiply(r, gray_avg / r_avg)
g = cv2.multiply(g, gray_avg / g_avg)
b = cv2.multiply(b, gray_avg / b_avg)
white_balanced = cv2.merge([r, g, b]).astype("uint8")


# Display the results
display_images(
    [sample_img_rgb, l_channel, a_channel, b_channel, cr_channel, cb_channel, white_balanced],
    ['Original', 'L* (Lightness)', 'a* (Green-Red)', 'b* (Blue-Yellow)',
     'Cr (Red Chroma)', 'Cb (Blue Chroma)', 'Simple White Balance'],
    2, 4
)

# --- Advanced Edge Detection and Thresholding ---
gray_img = cv2.cvtColor(sample_img_rgb, cv2.COLOR_RGB2GRAY)
blurred_gray = cv2.GaussianBlur(gray_img, (5, 5), 0)

# 7. Sobel Edge Detection
sobel_x = cv2.Sobel(blurred_gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(blurred_gray, cv2.CV_64F, 0, 1, ksize=5)
sobel_combined = cv2.magnitude(sobel_x, sobel_y).astype(np.uint8)

# 8. Laplacian of Gaussian
laplacian = cv2.Laplacian(blurred_gray, cv2.CV_64F).astype(np.uint8)

# 9. Adaptive Thresholding
adaptive_thresh = cv2.adaptiveThreshold(blurred_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)

display_images(
    [gray_img, sobel_combined, laplacian, adaptive_thresh],
    ['Grayscale', 'Sobel Edges', 'Laplacian Edges', 'Adaptive Thresholding'],
    1, 4
)

# --- Morphological Feature Extraction ---
gray_img = cv2.cvtColor(sample_img_rgb, cv2.COLOR_RGB2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

# 10. Top-Hat Transform (Finds bright spots)
top_hat = cv2.morphologyEx(gray_img, cv2.MORPH_TOPHAT, kernel)

# 11. Black-Hat Transform (Finds dark spots)
black_hat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)

# 12. Morphological Gradient (Finds outlines)
gradient = cv2.morphologyEx(gray_img, cv2.MORPH_GRADIENT, kernel)


display_images(
    [gray_img, top_hat, black_hat, gradient],
    ['Grayscale', 'Top-Hat (Highlights Bright Lesions)', 'Black-Hat (Highlights Dark Lesions)', 'Morphological Gradient (Outlines)'],
    1, 4
)

# --- Texture Feature Visualization ---
from skimage.feature import local_binary_pattern, hog
from skimage import exposure

gray_img = cv2.cvtColor(sample_img_rgb, cv2.COLOR_RGB2GRAY)
IMG_SIZE_HOG = 64 # HOG works best on smaller, consistent image sizes

# 13. Local Binary Patterns (LBP)
radius = 3
n_points = 8 * radius
lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')

# 14. Histogram of Oriented Gradients (HOG)
# We need to resize the image for HOG
resized_img = cv2.resize(gray_img, (IMG_SIZE_HOG, IMG_SIZE_HOG))
hog_features, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True, transform_sqrt=True)

# Rescale HOG image for better viewing
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

display_images(
    [gray_img, lbp, hog_image_rescaled],
    ['Grayscale Original', 'Local Binary Patterns (LBP) Image', 'Histogram of Oriented Gradients (HOG)'],
    1, 3
)

"""Defining the Best Preprocessing Pipeline"""

# --- Defining the Best Preprocessing Pipeline ---

def advanced_preprocessing(image):
    """
    Applies a series of advanced DIP techniques to an input image.
    This function will be used by our ImageDataGenerator.

    Args:
        image: A NumPy array representing an image.

    Returns:
        A processed NumPy array of the same type and shape.
    """
    # 1. Bilateral Filter for edge-preserving noise reduction
    # The image from the generator is float32, so we convert to uint8 for cv2
    image_uint8 = tf.cast(image, tf.uint8).numpy()
    bilateral_filtered = cv2.bilateralFilter(image_uint8, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # We apply CLAHE to the L channel of the L*a*b* color space to avoid color distortion
    lab_img = cv2.cvtColor(bilateral_filtered, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_l = clahe.apply(l_channel)
    clahe_lab_img = cv2.merge((clahe_l, a_channel, b_channel))
    clahe_rgb = cv2.cvtColor(clahe_lab_img, cv2.COLOR_LAB2RGB)

    # 3. Unsharp Masking for Sharpening
    gaussian_blur = cv2.GaussianBlur(clahe_rgb, (5, 5), 10.0)
    # Convert to float to avoid overflow issues with addWeighted
    unsharp_masked = cv2.addWeighted(clahe_rgb.astype('float32'), 1.5, gaussian_blur.astype('float32'), -0.5, 0)

    # Clip values to be in the valid range [0, 255] and convert back to uint8
    unsharp_masked = np.clip(unsharp_masked, 0, 255)

    return unsharp_masked


# --- Setup Data Generators with the Preprocessing Pipeline ---

# The generator for the advanced models will now use our custom function.
# Standard augmentations are still applied AFTER our preprocessing function.
train_datagen_advanced = ImageDataGenerator(
    preprocessing_function=advanced_preprocessing,
    rescale=1./255, # Rescale after preprocessing
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation generator also needs the preprocessing, but no augmentation
validation_datagen_advanced = ImageDataGenerator(
    preprocessing_function=advanced_preprocessing,
    rescale=1./255
)


# --- Create the new generators ---
train_generator_advanced = train_datagen_advanced.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator_advanced = validation_datagen_advanced.flow_from_directory(
    valid_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Advanced preprocessing pipeline and data generators are ready.")

""" Visualize the Preprocessing Pipeline Output"""

# --- Visualize a batch from the advanced generator ---

print("--- Sample images after Advanced Preprocessing & Augmentation ---")
x_batch_adv, y_batch_adv = next(train_generator_advanced)

plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    # The output is already rescaled to [0,1], so we can show it directly
    plt.imshow(x_batch_adv[i])
    plt.title(class_names[np.argmax(y_batch_adv[i])])
    plt.axis('off')
plt.tight_layout()
plt.savefig('output_plots/06_preprocessed_samples.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: output_plots/06_preprocessed_samples.png")

"""Model 1 Training - Baseline Simple CNN"""

# --- 3. Structured Model Training and Evaluation ---

# We need the basic generator for the baseline model (rescale only, no augmentation)
train_datagen_basic = ImageDataGenerator(rescale=1./255)
validation_datagen_basic = ImageDataGenerator(rescale=1./255)

train_generator_basic = train_datagen_basic.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)
validation_generator_basic = validation_datagen_basic.flow_from_directory(
    valid_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


# --- Model 1: Baseline Simple CNN Training ---
print("--- Training Model 1: Baseline CNN ---")
model_1 = tf.keras.models.Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_1.summary()

history_1 = model_1.fit(
    train_generator_basic,
    epochs=10, # Train for 10 epochs
    validation_data=validation_generator_basic
)

# Save the model
model_save_path = 'baseline_cnn_model.keras'
model_1.save(model_save_path)
print(f"Baseline CNN model saved to: {model_save_path}")

# Also save class names for later use
class_indices_path = 'class_indices.json'
with open(class_indices_path, 'w') as f:
    json.dump(train_generator_basic.class_indices, f)
print(f"Class indices saved to: {class_indices_path}")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Run 'python dip_project_model_testing.py' to evaluate the model")
print("2. Or run 'streamlit run app.py' for interactive web demo")
print("="*80)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# --- Core Libraries ---
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- Kaggle-Specific Library ---
import kagglehub

# ==============================================================================
# 1. DOWNLOAD AND SETUP DATASET
# ==============================================================================
print("--- Step 1: Downloading and setting up the dataset ---")

# Download the dataset using KaggleHub
dataset_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print(f"Dataset downloaded to: {dataset_path}")

# Construct the correct paths
base_dir = os.path.join(dataset_path, 'New Plant Diseases Dataset(Augmented)', 'New Plant Diseases Dataset(Augmented)')
train_dir = os.path.join(base_dir, 'train')
# We will use the 'valid' directory as our test set for this evaluation
test_dir = os.path.join(base_dir, 'valid')

# Define image size and batch size
IMG_SIZE = 128
BATCH_SIZE = 32

# Get class names from the folder names in the training directory
class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)
print(f"\nFound {num_classes} classes.")

# ==============================================================================
# 2. LOAD THE TRAINED MODEL
# ==============================================================================
print("\n--- Step 2: Loading the pre-trained custom CNN model ---")
# This is the path where your training script saved the model
model_path = '/kaggle/input/plant_disease/keras/default/1/baseline_cnn_model.keras'

# Check if the model file exists before trying to load it
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    model.summary()
else:
    print(f"ERROR: Model file not found at {model_path}")
    print("Please make sure you have run the training script first and the model was saved correctly.")
    # Exit the script if the model isn't found
    exit()

# ==============================================================================
# 3. PREPARE THE TEST DATASET
# ==============================================================================
print("\n--- Step 3: Preparing the test data generator ---")
# For evaluation, we only need to rescale the images. No augmentation is applied.
# It is CRITICAL to set shuffle=False to ensure that the order of predictions
# matches the order of the true labels.
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Do not shuffle the data
)

# ==============================================================================
# 4. EVALUATE AND PREDICT
# ==============================================================================
print("\n--- Step 4: Evaluating the model on the test data ---")
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

print("\nGenerating predictions...")
# Get the predicted probabilities for each class
y_pred_probs = model.predict(test_generator)
# Get the predicted class index by finding the index with the highest probability
y_pred = np.argmax(y_pred_probs, axis=1)
# Get the true class indices
y_true = test_generator.classes

# ==============================================================================
# 5. GENERATE VISUALIZATIONS
# ==============================================================================
print("\n--- Step 5: Generating performance visualizations ---")

# --- Visualization 1: Classification Report ---
# This provides a detailed breakdown of precision, recall, and F1-score for each class.
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))

# --- Visualization 2: Confusion Matrix Heatmap ---
# This is the best way to visualize which classes the model is confusing.
print("\n--- Generating Confusion Matrix ---")
conf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(22, 22))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix for Custom CNN', fontsize=26, pad=20)
plt.ylabel('Actual Class', fontsize=22)
plt.xlabel('Predicted Class', fontsize=22)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()


# --- Visualization 3: Visualize Misclassified Images ---
# Seeing where the model fails can provide valuable insights.
print("\n--- Visualizing Misclassified Images ---")
# Find the indices of all misclassified images
misclassified_indices = np.where(y_pred != y_true)[0]

# Randomly select a few misclassified images to display (e.g., 9)
if len(misclassified_indices) > 0:
    random_indices = np.random.choice(misclassified_indices, size=min(9, len(misclassified_indices)), replace=False)

    plt.figure(figsize=(15, 15))
    plt.suptitle("Sample Misclassified Images", fontsize=24, y=0.97)

    for i, idx in enumerate(random_indices):
        plt.subplot(3, 3, i + 1)

        # Get the corresponding image file path
        image_path = test_generator.filepaths[idx]
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))

        # Get the true and predicted labels
        true_label = class_names[y_true[idx]].replace("_", " ")
        predicted_label = class_names[y_pred[idx]].replace("_", " ")

        plt.imshow(img)
        plt.title(f"True: {true_label}\nPredicted: {predicted_label}", fontsize=12, color='darkred')
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
else:
    print("No misclassified images found! The model is perfect on this test set.")

