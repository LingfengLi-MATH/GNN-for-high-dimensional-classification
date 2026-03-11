import os
import os.path as osp
import copy
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST

from GNN_models import ImageEncoder
from image_gcl import image_GCL_training
from eval_methods import eval_GNN, eval_svm, eval_GPN, eval_MBO
from utils import DatasetSplit, get_image_vector_representations, construct_dgl_graph


parser = argparse.ArgumentParser(description="image classification on graph built from contrastive embeddings")
parser.add_argument("--dataset", type=str, default="MNIST", help="dataset to use (MNIST,FashionMNIST)")
parser.add_argument("--model", type=str, default="svm", help="choose from GNN, svm, GPN, MBO")
parser.add_argument("--gpu", type=int, default=0, help="gpu id")
parser.add_argument("--hdim", type=int, default=32, help="embedding hidden dimension")
parser.add_argument("--epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("--cl_mode", type=str, default="both", help="contrastive mode: simclr, supcon, both")
parser.add_argument("--noise_std", type=float, default=0.2, help="std for Gaussian noisy hold-out embeddings")
args = parser.parse_args()


device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
path = osp.join(osp.expanduser("~"), "datasets")

if args.dataset == "MNIST":
    DatasetClass = MNIST
elif args.dataset == "FashionMNIST":
    DatasetClass = FashionMNIST
else:
    raise ValueError("dataset must be one of: MNIST, FashionMNIST")

raw_dataset = DatasetClass(root=path, train=True, download=True, transform=None)
tensor_dataset = DatasetClass(root=path, train=True, download=True, transform=transforms.ToTensor())

indices = np.arange(len(tensor_dataset))
np.random.seed(0)
np.random.shuffle(indices)
left = indices[: int(len(indices) * 0.05)]
hold = indices[int(len(indices) * 0.05) :]

raw_subset = Subset(raw_dataset, left.tolist())
tensor_subset = Subset(tensor_dataset, left.tolist())

model_path = f"models/{args.dataset}_img_encoder_hdim{args.hdim}_{args.cl_mode}.pth"
data_path = f"data/{args.dataset}_img_{args.cl_mode}{'_gpu' if torch.cuda.is_available() else '_cpu'}.pth"

sample_x, _ = tensor_dataset[0]
input_channels = sample_x.size(0)
encoder_model = ImageEncoder(input_channels=input_channels, hidden_dim=args.hdim).to(device)

if osp.isfile(model_path):
    print("image encoder exists.")
    encoder_model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("start training image encoder by contrastive learning.")
    encoder_model = image_GCL_training(
        train_dataset=raw_subset,
        file_path=model_path,
        hdim=args.hdim,
        epochs=args.epochs,
        cl_mode=args.cl_mode,
        device=device,
    )

encoder_model.eval()

if osp.isfile(data_path):
    loaded_data = torch.load(data_path, map_location=device)
    graph_sub = loaded_data["graph_sub"]
    graph_full = loaded_data["graph_full"]
    graph_full_noisy = loaded_data["graph_full_noisy"]
else:
    full_loader = DataLoader(tensor_dataset, batch_size=512, shuffle=False)
    sub_loader = DataLoader(tensor_subset, batch_size=512, shuffle=False)

    rep = get_image_vector_representations(encoder_model, full_loader, device=device, noise_std=args.noise_std)
    rep_sub = get_image_vector_representations(encoder_model, sub_loader, device=device, noise_std=args.noise_std)

    graph_full, graph_sub = construct_dgl_graph(rep["x"].detach().cpu(), rep["y"], left)
    graph_full.hold = hold
    graph_full.left = left
    graph_full.rep = rep
    graph_sub.rep = rep_sub

    graph_full_noisy = copy.deepcopy(graph_full)
    graph_full_noisy.ndata["feat"][hold] = rep["x1"][hold]

graph_full = graph_full.to(device)
graph_sub = graph_sub.to(device)
graph_full_noisy = graph_full_noisy.to(device)

test = []
test_hold = []
test_hold_noise = []

for seed in range(1, 6):
    torch.manual_seed(seed)
    split = DatasetSplit(num_samples=len(left), train_ratio=0.25, val_ratio=0.25, k=1)[0]

    if args.model == "GNN":
        acc1, acc2, acc3 = eval_GNN(graph_sub, graph_full, graph_full_noisy, split, args.epochs, device)
    elif args.model == "svm":
        acc1, acc2, acc3 = eval_svm(graph_full.rep, left, hold, split, device)
    elif args.model == "MBO":
        acc1, acc2, acc3 = eval_MBO(graph_sub, left, hold, split, device)
    elif args.model == "GPN":
        acc1, acc2, acc3 = eval_GPN(graph_sub, graph_full, graph_full_noisy, split, args.epochs, device)
    else:
        raise ValueError("model must be one of: GNN, svm, GPN, MBO")

    test.append(acc1)
    test_hold.append(acc2)
    test_hold_noise.append(acc3)

with open("results.txt", "a") as myfile:
    myfile.write(f"dataset:{args.dataset}, model:{args.model}, cl_mode:{args.cl_mode}\n")
    myfile.write(
        "prediction accuracy on the testing set:{}({})\n".format(np.mean(test), np.std(test))
    )
    myfile.write(
        "prediction accuracy on the hold-out set:{}({})\n".format(np.mean(test_hold), np.std(test_hold))
    )
    myfile.write(
        "prediction accuracy on the noisy hold-out set:{}({})\n".format(
            np.mean(test_hold_noise), np.std(test_hold_noise)
        )
    )

print(f"dataset:{args.dataset}, model:{args.model}, cl_mode:{args.cl_mode}")
print("prediction accuracy on the testing set:{}({})".format(np.mean(test), np.std(test)))
print("prediction accuracy on the hold-out set:{}({})".format(np.mean(test_hold), np.std(test_hold)))
print(
    "prediction accuracy on the noisy hold-out set:{}({})".format(
        np.mean(test_hold_noise), np.std(test_hold_noise)
    )
)

if not osp.isfile(data_path):
    saved_data = {
        "graph_sub": graph_sub.to("cpu"),
        "graph_full": graph_full.to("cpu"),
        "graph_full_noisy": graph_full_noisy.to("cpu"),
    }
    torch.save(saved_data, data_path)
