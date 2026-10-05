import torch, os, numpy as np
from PIL import Image
from backend.app.ml.uav.tiling import segment_large_image
from backend.app.services.model_loader import load_checkpoint
from backend.app.ml.uav.regions import find_regions
from backend.app.services.uav_inference import load_classes

model, ckpt = load_checkpoint('models/uav/best_model.pth', 'uav')
classes, suspicious = load_classes()

def predict(batch):
    x = torch.from_numpy(batch).permute(0,3,1,2).float()/255.
    return torch.softmax(model(x), 1).detach().numpy()

# Test one of the user uploads
rgb = np.asarray(Image.open('uploads/uav/7ab2ce9903414bafb829dc6ef6cd9b8e.jpg').convert('RGB'))
labels, _, _ = segment_large_image(rgb, predict, 5, 224)
regions = find_regions(labels, suspicious, classes, 50)
print('Unique labels:', np.unique(labels))
print('Number of regions:', len(regions))
for r in regions:
    print(f"Region id: {r['id']}, Dominant: {r['dominant_class']}")
