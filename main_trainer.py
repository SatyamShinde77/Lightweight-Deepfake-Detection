
import yaml
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from lightning_modules.detector import DeepfakeDetector
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
torch.backends.cudnn.benchmark = True

# ===============================
# Load Config
# ===============================
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# ===============================
# Transforms
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# Datasets (IMAGE-BASED)
# ===============================
train_dataset = ImageFolder(
    root="data/train",
    transform=transform
)

val_dataset = ImageFolder(
    root="data/validation",
    transform=transform
)

# Safety check (prevents silent failures)
assert len(train_dataset) > 0, "❌ Training dataset is empty"
assert len(val_dataset) > 0, "❌ Validation dataset is empty"

print(f"✅ Train samples: {len(train_dataset)}")
print(f"✅ Val samples: {len(val_dataset)}")
print(f"✅ Classes: {train_dataset.classes}")

# ===============================
# DataLoaders
# ===============================
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg["batch_size"],
    shuffle=True,
    num_workers=0,      # 🔥 IMPORTANT FOR COLAB
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg["batch_size"],
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ===============================
# Model (EfficientNet-B0)
# ===============================
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
backbone = efficientnet_b0(weights=weights)

in_features = backbone.classifier[1].in_features
backbone.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(in_features, 2)
)

model = DeepfakeDetector(
    model=backbone,
    lr=cfg["lr"]
)

# ===============================
# Callbacks
# ===============================
checkpoint_cb = ModelCheckpoint(
    monitor=cfg["monitor_metric"],
    dirpath="models",
    filename="best_model",
    save_top_k=cfg["save_top_k"],
    mode="min"
)

early_stop_cb = EarlyStopping(
    monitor=cfg["monitor_metric"],
    patience=cfg["early_stopping_patience"],
    mode="min"
)

# ===============================
# Trainer
# ===============================
trainer = pl.Trainer(
    max_epochs=cfg["num_epochs"],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    callbacks=[checkpoint_cb, early_stop_cb],
    log_every_n_steps=1,        # 🔥 force frequent logs
    enable_progress_bar=True,  # 🔥 show epoch
    num_sanity_val_steps=0
)

# ===============================
# Train
# ===============================
trainer.fit(model, train_loader, val_loader)
