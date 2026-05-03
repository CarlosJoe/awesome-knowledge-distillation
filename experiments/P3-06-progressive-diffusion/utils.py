import torch
import numpy as np


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_fid_placeholder(real_features, fake_features):
    mu_real = real_features.mean(dim=0)
    mu_fake = fake_features.mean(dim=0)
    sigma_real = np.cov(real_features.cpu().numpy(), rowvar=False)
    sigma_fake = np.cov(fake_features.cpu().numpy(), rowvar=False)

    diff = mu_real - mu_fake
    covmean, _ = _matrix_sqrt(sigma_real @ sigma_fake)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(fid)


def _matrix_sqrt(matrix):
    from scipy.linalg import sqrtm
    return sqrtm(matrix), None


def extract_features(model, dataloader, device, num_samples=1000):
    features = []
    count = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            feat = model(images)
            features.append(feat.flatten(1))
            count += images.shape[0]
            if count >= num_samples:
                break
    features = torch.cat(features, dim=0)[:num_samples]
    return features
