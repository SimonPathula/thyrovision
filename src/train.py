import os
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


class Avg2MaxPooling(layers.Layer):
    def __init__(self, pool_size=3, strides=2, padding="same"):
        super().__init__()
        self.avg_pool = layers.AveragePooling2D(pool_size, strides, padding)
        self.max_pool = layers.MaxPooling2D(pool_size, strides, padding)

    def call(self, inputs):
        # ── FIX 4: Compute max_pool once, reuse — avoids duplicate graph nodes ─
        avg = self.avg_pool(inputs)
        mx = self.max_pool(inputs)
        return avg - (mx + mx)


class DepthwiseSeparableConv(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1):
        super().__init__()
        self.dw = layers.DepthwiseConv2D(kernel_size, strides, padding="same")
        self.pw = layers.Conv2D(filters, 1, strides=1)
        self.bn = layers.BatchNormalization()

    def call(self, inputs):
        x = self.dw(inputs)
        x = self.pw(x)
        return tf.nn.relu(self.bn(x))


def create_fibonacci_net(input_shape=(224, 224, 3), num_classes=1):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(21, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 2
    x = layers.Conv2D(34, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x2 = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x2)

    # Block 3
    x = layers.Conv2D(55, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x3 = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x3)

    # PCB 1: from Block 2 features → merges at Block 4
    pcb1 = layers.Conv2D(24, 3, padding="same")(x2)
    pcb1 = Avg2MaxPooling()(pcb1)
    pcb1 = layers.Conv2D(24, 3, padding="same")(pcb1)
    pcb1 = Avg2MaxPooling()(pcb1)

    # Block 4
    x = layers.Conv2D(89, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    pcb1 = layers.Resizing(14, 14)(pcb1)
    x = layers.concatenate([x, pcb1])

    # PCB 2: from Block 3 features → merges at Block 5
    pcb2 = layers.Conv2D(24, 3, padding="same")(x3)
    pcb2 = Avg2MaxPooling()(pcb2)
    pcb2 = layers.Conv2D(24, 3, padding="same")(pcb2)
    pcb2 = Avg2MaxPooling()(pcb2)

    # Block 5
    x = layers.Conv2D(144, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    pcb2 = layers.Resizing(7, 7)(pcb2)
    x = layers.concatenate([x, pcb2])

    x = DepthwiseSeparableConv(233)(x)
    x = DepthwiseSeparableConv(377)(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    return Model(inputs, outputs)


model = create_fibonacci_net(num_classes=1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC()]
)

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
checkpoint = ModelCheckpoint("thyroid_fibonaccinet_best.keras", monitor="val_loss", save_best_only=True, verbose=1)

history1 = model.fit(
    train_gen_new,
    validation_data=valid_gen_new,
    epochs=30,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

model.save("thyroid_fibonaccinet.keras")
