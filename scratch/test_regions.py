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

rgb = np.asarray(Image.open('datasets/uav/images/mix_p0000007__water__water_p0000007_y0_x5152_t224.jpg').convert('RGB'))
labels, _, _ = segment_large_image(rgb, predict, 5, 224)
regions = find_regions(labels, suspicious, classes, 50)
print('suspicious ids:', suspicious)
for r in regions:
    print(f"Region id: {r['id']}, Area: {r['area_px']}, Dominant: {r['dominant_class']}")
    for c in r['composition']:
        print(f"  - {c['class_name']}: {c['fraction']:.2f} ({c['pixel_count']} px)")
