import torch, os, numpy as np
from PIL import Image
from backend.app.ml.uav.tiling import segment_large_image
from backend.app.services.model_loader import load_checkpoint

model, ckpt = load_checkpoint('models/uav/best_model.pth', 'uav')
imgs = os.listdir('datasets/uav/images')[:20]

def predict(batch):
    x = torch.from_numpy(batch).permute(0,3,1,2).float()/255.
    return torch.softmax(model(x), 1).detach().numpy()

for i in imgs:
    if not i.endswith('.jpg'): continue
    rgb = np.asarray(Image.open('datasets/uav/images/' + i).convert('RGB'))
    labels, _, _ = segment_large_image(rgb, predict, 5, 224)
    print(i, np.unique(labels))
