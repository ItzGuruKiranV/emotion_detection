import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# ===============================
# 1️⃣ Paths
# ===============================
train_dir = "dataset/image_dataset/train"
val_dir = "dataset/image_dataset/test"  # or "dataset/validation" if you have one
model_path = "models/image_emotion_model_refined.h5"

# ===============================
# 2️⃣ Load the model
# ===============================
print("🔄 Loading model...")
model = load_model(model_path)
print("✅ Model loaded successfully!")

# ===============================
# 3️⃣ Unfreeze deeper layers for fine-tuning
# ===============================
set_trainable = False
for i, layer in enumerate(model.layers):
    if "expanded_conv_depthwise" in layer.name or "block_1" in layer.name:
        set_trainable = True
    layer.trainable = set_trainable
    print(f"{i} {layer.name} {'Trainable' if layer.trainable else 'Frozen'}")

# ===============================
# 4️⃣ Recompile model
# ===============================
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# 5️⃣ Data Generators (with augmentation)
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.15
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# ===============================
# 6️⃣ Train the model
# ===============================
print("🚀 Starting fine-tuning...")
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    verbose=1
)

# ===============================
# 7️⃣ Evaluate model accuracy
# ===============================
loss, acc = model.evaluate(val_gen)
print(f"✅ Validation Accuracy: {acc*100:.2f}%")
print(f"✅ Validation Loss: {loss:.4f}")

# ===============================
# 8️⃣ Save fine-tuned model
# ===============================
model.save("emotion_model_finetuned.h5")
print("💾 Fine-tuned model saved successfully as emotion_model_finetuned.h5 ✅")
