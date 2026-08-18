"""
Standalone localization inference script.

Usage:
    python inference.py --reference path/to/reference.png --search path/to/search.png [--weights model_weights.pt]

Prints and returns the predicted center (x, y) in pixel coordinates of the
Search Image (in the ORIGINAL image resolution, e.g. 1000x1000).
Runs with no manual edits, given a weights file trained by this notebook.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_weights.pt')


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, stride=1, pool=False):
        super().__init__()
        layers = [nn.Conv2d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False),
                  nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SharedBackbone(nn.Module):
    def __init__(self, c_out=64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1, 16, k=5, stride=2),
            ConvBlock(16, 32, k=3, stride=1, pool=True),
            ConvBlock(32, 64, k=3, stride=1, pool=True),
            ConvBlock(64, c_out, k=3, stride=1),
        )

    def forward(self, x):
        return self.net(x)


class SiameseLocalizer(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()
        self.backbone = SharedBackbone(feat_dim)

    def forward(self, template, search):
        f_t = self.backbone(template)
        f_s = self.backbone(search)
        B, C, t, t2 = f_t.shape
        _, _, H, W = f_s.shape
        f_s_grouped = f_s.reshape(1, B * C, H, W)
        kernel = f_t.reshape(B * C, 1, t, t2)
        response = F.conv2d(f_s_grouped, kernel, groups=B * C)
        response = response.reshape(B, C, response.shape[-2], response.shape[-1]).mean(dim=1, keepdim=True)
        heatmap = F.interpolate(response, size=(search.shape[-2], search.shape[-1]),
                                 mode='bilinear', align_corners=False)
        return heatmap


def soft_argmax_2d(heatmap):
    B, _, H, W = heatmap.shape
    flat = heatmap.view(B, -1)
    probs = F.softmax(flat, dim=1).view(B, 1, H, W)
    xs = torch.linspace(0, 1, W, device=heatmap.device).view(1, 1, 1, W)
    ys = torch.linspace(0, 1, H, device=heatmap.device).view(1, 1, H, 1)
    x = (probs * xs).sum(dim=[1, 2, 3])
    y = (probs * ys).sum(dim=[1, 2, 3])
    return torch.stack([x, y], dim=1), probs


def localize(reference_path, search_path, weights_path=DEFAULT_WEIGHTS, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(weights_path, map_location=device)
    cfg = ckpt.get('config', {'SEARCH_SIZE': 512, 'TEMPLATE_SIZE': 51, 'ORIG_SIZE': 1000.0})

    model = SiameseLocalizer().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    ref = Image.open(reference_path).convert('L')
    search = Image.open(search_path).convert('L')
    orig_w, orig_h = search.size  # (width, height) of the ORIGINAL search image

    S, T = cfg['SEARCH_SIZE'], cfg['TEMPLATE_SIZE']
    template_np = np.asarray(ref.resize((T, T), Image.BILINEAR), dtype=np.float32) / 255.0
    search_np = np.asarray(search.resize((S, S), Image.BILINEAR), dtype=np.float32) / 255.0

    template_t = torch.from_numpy(template_np).unsqueeze(0).unsqueeze(0).float().to(device)
    search_t = torch.from_numpy(search_np).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        heatmap = model(template_t, search_t)
        pred_xy_norm, _ = soft_argmax_2d(heatmap)

    pred_x = pred_xy_norm[0, 0].item() * orig_w
    pred_y = pred_xy_norm[0, 1].item() * orig_h
    return pred_x, pred_y


def main():
    parser = argparse.ArgumentParser(description="Locate a reference SEM pattern inside a search image.")
    parser.add_argument('--reference', required=True, help='Path to reference image')
    parser.add_argument('--search', required=True, help='Path to search image')
    parser.add_argument('--weights', default=DEFAULT_WEIGHTS, help='Path to model_weights.pt')
    args = parser.parse_args()

    x, y = localize(args.reference, args.search, args.weights)
    print(f"{x:.2f},{y:.2f}")
    return x, y

if __name__ == '__main__':
    main()
