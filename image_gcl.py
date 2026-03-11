import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from tqdm import tqdm

from GNN_models import ImageEncoder, ProjectionHead


class TwoViewDataset(Dataset):
    def __init__(self, base_dataset, view_transform):
        self.base_dataset = base_dataset
        self.view_transform = view_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        view1 = self.view_transform(image)
        view2 = self.view_transform(image)
        return view1, view2, label


def build_contrastive_view_transform(image_size=28):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.8),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
    ])


def simclr_loss(z1, z2, temperature=0.2):
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)
    logits = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(mask, -1e9)

    targets = torch.arange(batch_size, device=z.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)
    return F.cross_entropy(logits, targets)


def supervised_contrastive_loss(features, labels, temperature=0.2):
    features = F.normalize(features, dim=1)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)

    logits = torch.matmul(features, features.T) / temperature
    logits_mask = 1.0 - torch.eye(features.size(0), device=features.device)
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    mask = mask * logits_mask
    pos_count = mask.sum(1)
    pos_count = torch.where(pos_count > 0, pos_count, torch.ones_like(pos_count))
    mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count

    loss = -mean_log_prob_pos.mean()
    return loss


def train_image_cl(backbone, projector, dataloader, optimizer, cl_mode="both", device="cpu"):
    backbone.train()
    projector.train()
    epoch_loss = 0.0

    for x1, x2, labels in dataloader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        labels = labels.to(device)

        z1 = projector(backbone(x1))
        z2 = projector(backbone(x2))

        loss_simclr = simclr_loss(z1, z2)

        z_cat = torch.cat([z1, z2], dim=0)
        labels_cat = torch.cat([labels, labels], dim=0)
        loss_supcon = supervised_contrastive_loss(z_cat, labels_cat)

        if cl_mode == "simclr":
            loss = loss_simclr
        elif cl_mode == "supcon":
            loss = loss_supcon
        else:
            loss = 0.5 * (loss_simclr + loss_supcon)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / max(len(dataloader), 1)


@torch.no_grad()
def extract_embeddings(backbone, dataloader, device="cpu"):
    backbone.eval()
    features = []
    labels = []
    for x, y in dataloader:
        x = x.to(device)
        feat = backbone(x)
        features.append(feat.cpu())
        labels.append(y.cpu())
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels


def test_embedding_svm(backbone, dataloader, split, device="cpu"):
    x, y = extract_embeddings(backbone, dataloader, device=device)
    train_idx = split["train"].cpu().numpy()
    test_idx = split["test"].cpu().numpy()

    clf = LinearSVC(max_iter=10000)
    clf.fit(x[train_idx].numpy(), y[train_idx].numpy())
    pred = clf.predict(x[test_idx].numpy())

    y_true = y[test_idx].numpy()
    return {
        "micro_f1": f1_score(y_true, pred, average="micro"),
        "macro_f1": f1_score(y_true, pred, average="macro"),
    }


def image_GCL_training(train_dataset, file_path, hdim=32, epochs=200, cl_mode="both", batch_size=256, device="cpu"):
    view_transform = build_contrastive_view_transform(image_size=28)
    train_data = TwoViewDataset(train_dataset, view_transform)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    sample_x, _ = train_dataset[0]
    input_channels = sample_x.size(0) if torch.is_tensor(sample_x) else 1

    backbone = ImageEncoder(input_channels=input_channels, hidden_dim=hdim).to(device)
    projector = ProjectionHead(backbone.output_dim, backbone.output_dim).to(device)
    optimizer = Adam(list(backbone.parameters()) + list(projector.parameters()), lr=1e-3)

    with tqdm(total=epochs, desc="(T)") as pbar:
        for _ in range(epochs):
            loss = train_image_cl(backbone, projector, dataloader, optimizer, cl_mode=cl_mode, device=device)
            pbar.set_postfix({"loss": loss})
            pbar.update()

    torch.save(backbone.state_dict(), file_path)
    return backbone
