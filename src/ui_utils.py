import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    elif torch.backends.mps.is_available():
        return torch.device("mps")

    else:
        return torch.device("cpu")


DEVICE = get_device()
MODEL_PATH = "post_train_nb/train_logs/siamese_signature/version_10/checkpoints/epoch=22-step=3588.ckpt"
IMAGE_SIZE = 224
THRESHOLD = 0.5655121


# ---------------------------------------------------
# TRANSFORM
# ---------------------------------------------------

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

def load_model(model_class):
    model = model_class.load_from_checkpoint(checkpoint_path=MODEL_PATH,embedding_dim=256)
    model = model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------
# PREPROCESS IMAGE
# ---------------------------------------------------

def preprocess_image(image):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    img = transform(image).unsqueeze(0)
    return img.to(DEVICE)


# ---------------------------------------------------
# GET EMBEDDING
# ---------------------------------------------------

def get_embedding(model, img_tensor):
    with torch.no_grad():
        emb = model.forward_once(img_tensor)
    emb = F.normalize(emb, p=2, dim=1)

    return emb


# ---------------------------------------------------
# DISTANCE
# ---------------------------------------------------

def compute_distance(emb1, emb2):
    distance = torch.norm(emb1 - emb2, dim=1)
    return distance.item()


# ---------------------------------------------------
# VERIFY SIGNATURES
# ---------------------------------------------------

def verify_signatures(model, img1, img2):

    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)

    emb1 = get_embedding(model, img1)
    emb2 = get_embedding(model, img2)

    distance = compute_distance(emb1, emb2)
    same = distance < THRESHOLD

    return {
        "distance": float(distance),
        "threshold": THRESHOLD,
        "same_signature": bool(same)
    }