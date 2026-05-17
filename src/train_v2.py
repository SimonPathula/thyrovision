import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

# ── FIX 1: Enable memory growth BEFORE any other TF ops ──────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ── FIX 2: Disable XLA JIT (causes CUDNN_STATUS_EXECUTION_FAILED on 1650 Ti) ─
tf.config.optimizer.set_jit(False)

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model

BASE_PATH = "/mnt/d/projects/thyroid_cancer_detection/archive"
DATASET_PATH = os.path.join(BASE_PATH, "Thyroid Data")

categories = [0, 1]

image_paths = []
labels = []

for cat in categories:
    labelled_path = os.path.join(DATASET_PATH, str(cat))
    for img_name in os.listdir(labelled_path):
        if img_name.lower().endswith((".png", ".jpeg", ".jpg")):
            image_path = os.path.join(labelled_path, img_name)
            image_paths.append(image_path)
            labels.append(int(cat))

df = pd.DataFrame({"img_path": image_paths, "label": labels})

majority_class = df[df["label"] == 0]
minority_class = df[df["label"] == 1]

minority_oversampled = minority_class.sample(n=len(majority_class), replace=True, random_state=42)

df_resampled = pd.concat([majority_class, minority_oversampled]).sample(frac=1, random_state=42).reset_index(drop=True)

encoder = LabelEncoder()
df_resampled["category_encoded"] = encoder.fit_transform(df_resampled["label"])

train_df, temp_df = train_test_split(
    df_resampled,
    test_size=0.2,
    shuffle=True,
    random_state=42,
    stratify=df_resampled["category_encoded"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["category_encoded"],
    shuffle=True,
    random_state=42
)

train_df["category_encoded"] = train_df["category_encoded"].astype(str)
val_df["category_encoded"] = val_df["category_encoded"].astype(str)
test_df["category_encoded"] = test_df["category_encoded"].astype(str)

# ── FIX 3: Reduce batch_size to 2 for 2240MB VRAM ────────────────────────────
batch_size = 2
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_gen_new = train_gen.flow_from_dataframe(
    train_df,
    x_col="img_path",
    y_col="category_encoded",
    target_size=img_size,
    class_mode="binary",
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size
)
valid_gen_new = test_gen.flow_from_dataframe(
    val_df,
    x_col="img_path",
    y_col="category_encoded",
    target_size=img_size,
    class_mode="binary",
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size
)
test_gen_new = test_gen.flow_from_dataframe(
    test_df,
    x_col="img_path",
    y_col="category_encoded",
    target_size=img_size,
    class_mode="binary",
    color_mode="rgb",
    shuffle=False,
    batch_size=batch_size
)


import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Concatenate, Dense, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, Dropout

def fibonacci_filters(n):
    fib = [21, 34]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    return fib  # [21, 34, 55, 89, 144, 233, 377] for n=7
 
 

class FibonacciConvBlock(layers.Layer):
 
    def __init__(self, filters):
        super().__init__()
 
        self.conv = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer="he_normal"
        )
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.pool = MaxPooling2D((2, 2))
 
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.pool(x)
        return x
 
class Avg2MaxPooling(layers.Layer):
 
    def __init__(self):
        super().__init__()
 
        self.avg_pool = AveragePooling2D(pool_size=3, strides=2, padding="same")
        self.max_pool = MaxPooling2D(pool_size=3, strides=2, padding="same")
 
    def call(self, inputs):
        avg = self.avg_pool(inputs)
        mx  = self.max_pool(inputs)
        return avg - 2 * mx
 
class PCB1(layers.Layer):
 
    def __init__(self):
        super().__init__()
        self.avg2max   = Avg2MaxPooling()
        self.concat = Concatenate(axis=-1)
 
    def call(self, skip_input, main_input):
        skip = self.avg2max(skip_input)
        return self.concat([main_input, skip])

class PCB2(layers.Layer):
 
    def __init__(self, branch_filters=24):
        super().__init__()
 
        self.conv    = Conv2D(branch_filters, kernel_size=3, padding='same',
                              kernel_initializer='he_normal')
        self.bn      = BatchNormalization()
        self.relu    = ReLU()
        self.avg2max = Avg2MaxPooling()
        self.concat  = Concatenate(axis=-1)
 
    def call(self, skip_input, main_input, training=False):

        x = self.conv(skip_input)
        x = self.relu(x)
        x = self.bn(x, training=training)
        x = self.avg2max(x)
        return self.concat([main_input, x])
 
class DWSCBlock(layers.Layer):
 
    def __init__(self, filters, depth_multiplier):
        super().__init__()

        self.depthwise = layers.DepthwiseConv2D(
            depth_multiplier = depth_multiplier,
            kernel_size=(3, 3),
            padding='same'
        )

        self.pointwise = Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            padding='same'
        )

        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs, training=False):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x

class FibonacciNet(Model):
 
    def __init__(self, num_classes=1, activation = "sigmoid"):
        super().__init__()
 
        fib = fibonacci_filters(7)  # [21, 34, 55, 89, 144, 233, 377]
 
        self.block1 = FibonacciConvBlock(fib[0])
        self.block2 = FibonacciConvBlock(fib[1])
        self.pcb1   = PCB1()
        self.block3 = FibonacciConvBlock(fib[2])
        self.pcb2   = PCB2(branch_filters=24)
        self.block4 = FibonacciConvBlock(fib[3])
        self.block5 = FibonacciConvBlock(fib[4])
 
        self.dwsc1  = DWSCBlock(fib[5], 1)
        self.dwsc2  = DWSCBlock(fib[6], 2)          
 
        self.gap        = GlobalAveragePooling2D()
        self.dropout    = Dropout(0.5)
        self.classifier = Dense(num_classes, activation)
 
    def call(self, inputs, training=False):
 
        x1 = self.block1(inputs, training=training)
 
        x2 = self.block2(x1, training=training)

        x3 = self.block3(x2, training=training)
 
        x = self.pcb1(x2, x3)
 
        x = self.block4(x, training=training)
 
        x = self.pcb2(x3, x, training=training)

        x = self.block5(x, training= training)
 
        x = self.dwsc1(x, training=training)
        x = self.dwsc2(x, training=training)
 
        x = self.gap(x)
        x = self.dropout(x, training=training)
        x = self.classifier(x)
 
        return x


model = FibonacciNet(num_classes=1)

model.compile(
    optimizer= tf.keras.optimizers.Adam(
        learning_rate = 0.001
    ),
    loss= "binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC()
    ]
)

early_stop = EarlyStopping(monitor= "val_loss", patience= 3, restore_best_weights= True)
reduce_lr = ReduceLROnPlateau(monitor= "val_loss", factor= 0.5, patience= 2, verbose = 1)

checkpoint = ModelCheckpoint("models/thyroid_fibonaccinet_best.keras", monitor="val_loss", save_best_only=True, verbose=1)

history1 = model.fit(
    train_gen_new, 
    validation_data = valid_gen_new,
    epochs = 30,
    callbacks = [early_stop, reduce_lr, checkpoint]
)

model.save("models/fibonacci_net_final.keras")

with open("models/history.pkl", "wb") as f:
    pickle.dump(history1.history, f)