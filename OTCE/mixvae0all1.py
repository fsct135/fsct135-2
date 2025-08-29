import os
import random
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import make_scorer
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from lifelines import KaplanMeierFitter
from lifelines.calibration import survival_probability_calibration
from sksurv.metrics import integrated_brier_score
import pandas as pd
from sksurv.metrics import brier_score
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.datasets import load_gbsg2
from sksurv.util import Surv
from sklearn.metrics import brier_score_loss

import argparse
import builtins
import csv
import math
import shutil
import warnings

from torchvision.models import resnet18, resnet50
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from myuntil import *

import torchvision
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
torch.cuda.empty_cache()



class DCNNModule(nn.Module):
    def __init__(self):
        super(DCNNModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 30 * 30, 256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TransformerModule(nn.Module):
    def __init__(self, img_size=32, patch_size=8, dim=256, num_heads=4, num_classes=2):
        super(TransformerModule, self).__init__()
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads),
            num_layers=6
        )
        self.fc = nn.Linear(dim, 256)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = F.relu(self.fc(x))
        return x

class VAEModule(nn.Module):
    def __init__(self, input_dim=3, latent_dim=64, img_size=120):
        super(VAEModule, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_mu = nn.Linear(128 * (img_size // 4) * (img_size // 4), latent_dim)
        self.fc_logvar = nn.Linear(128 * (img_size // 4) * (img_size // 4), latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * (img_size // 4) * (img_size // 4))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        encoded = self.encoder(x).view(batch_size, -1)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.fc_decode(z).view(batch_size, 128, x.size(2) // 4, x.size(3) // 4)
        decoded = self.decoder(decoded)
        return decoded, mu, logvar

class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, features):
        attention_weights = torch.softmax(self.fc(features), dim=1)
        attended_features = features * attention_weights
        return attended_features, attention_weights

class DCNNTransformerVAEAttentionParallel(nn.Module):
    def __init__(self, num_classes=1, img_size=120, latent_dim=64):
        super(DCNNTransformerVAEAttentionParallel, self).__init__()
        self.dcnn = DCNNModule()
        self.transformer = TransformerModule(img_size=img_size)
        self.vae = VAEModule(input_dim=3, latent_dim=latent_dim, img_size=img_size)
        self.attention = AttentionModule(input_dim=512 + latent_dim)
        self.fc = nn.Linear(512 + latent_dim, num_classes)

    def forward(self, x):
        vae_recon, mu, logvar = self.vae(x)
        dcnn_features = self.dcnn(x)
        transformer_features = self.transformer(x)
        combined_features = torch.cat((dcnn_features, transformer_features, mu), dim=1)
        attended_features, attention_weights = self.attention(combined_features)
        output = self.fc(attended_features)
        return output, vae_recon, mu, logvar, attention_weights

class MyDataset2(Dataset):
    def __init__(self, root_dir, labels_txt, size, type="train"):
        self.root_dir = root_dir
        labels = []
        image_paths = []
        with open(labels_txt, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip("\n")
                img, label = line.split("\t")
                image_paths.append(os.path.join(root_dir, img))
                labels.append(label)
        self.image_paths = image_paths
        self.labels = [int(char) for char in labels]

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5
                ),
                transforms.ToTensor(),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ]
        )

        self.type = type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path)

        if self.type == "train":
            img1 = self.train_transform(image)
            label = torch.tensor(label, dtype=torch.float32)
            return img1, label
        else:
            image = self.test_transform(image)
            label = torch.tensor(label, dtype=torch.float32)
            return image, label

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def generate_labels_file(image_paths, labels, file_path):
    with open(file_path, 'w') as f:
        for img_path, label in zip(image_paths, labels):
            f.write(f"{os.path.basename(img_path)}\t{label}\n")

def main(args):
    train_ci = []
    model_name = f"CnnCoxVAE"
    dataset_name = f"{args.pre_method}{args.cancer}{args.zuxue}"
    resname = f"./CiRes/{dataset_name}/{model_name}/train_ci.csv"

    if not os.path.exists(f"./CiRes/{dataset_name}/{model_name}"):
        os.makedirs(f"./CiRes/{dataset_name}/{model_name}")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.empty_cache()
    root_folder = f"data/{args.cancer}/{args.pre_method}/img{args.zuxue}/"
    txt_file = f"data/{args.cancer}/{args.pre_method}/img{args.zuxue}/all_label.txt"

    
    all_image_paths, all_labels = [], []
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip("\n")
            img, label = line.split("\t")
            all_image_paths.append(os.path.join(root_folder, img))
            all_labels.append(int(label))


    root_folder = f"data/{args.cancer}/{args.pre_method}/img{args.zuxue}/"
    txt_file = f"data/{args.cancer}/{args.pre_method}/img{args.zuxue}/all_label.txt"

    train_dataset = MyDataset2(root_folder, txt_file, size=args.imgsize, type="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = DCNNTransformerVAEAttentionParallel(num_classes=2)
    print(model)
    #model = model.to(device)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    cox_criterion = DeepCox_LossFunc().cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)


        train(trainloader, model, cox_criterion, optimizer, epoch, args)

        print(f"Model type: {type(model)}")

        train_outputs = []
        train_labels_list = []

        model.eval()
        with torch.no_grad():
            for images, labels in trainloader:
                images = images.to(args.gpu).float()
                labels = labels.to(args.gpu)
                output, *_ = model(images)
                output = output[:, 1]

                train_outputs.extend(output.cpu().numpy())
                train_labels_list.extend(labels.cpu().numpy())

        train_results = pd.DataFrame({
            'True Survival Time': [abs(label) for label in train_labels_list],
            'True Status': [1 if label > 0 else 0 for label in train_labels_list],
            'Predicted Risk': train_outputs
        })


        output_dir = f"./CiRes/{dataset_name}/{model_name}/{args.nums_layer}_{args.epochs}_{args.batch_size}_{args.lr}_{args.moco_dim}_{args.seed}_{epoch}/"
        os.makedirs(output_dir, exist_ok=True)
        train_results_file = os.path.join(output_dir, "train_results.csv")
        train_results.to_csv(train_results_file, index=False)


        train_ci_value = inference_ci(trainloader, model, args.gpu)
        print(f"Epoch {epoch} - Train C-index = {train_ci_value:.4f}")
        train_ci.append(train_ci_value)


        save_model(model, epoch, f"./saved_models/{args.cancer}/{args.pre_method}/{args.zuxue}")

    with open(resname, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            f"epochs={args.epochs}", 
            f"layer={args.nums_layer}",
            f"batch_size={args.batch_size}", 
            f"lr={args.lr}", 
            f"seed={args.seed}",
            f"moco_dim={args.moco_dim}"
        ])
        writer.writerow(train_ci)
        writer.writerow([f"Mean Train C-index: {np.mean(train_ci):.4f}"])


def save_model(model, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存至 {model_save_path}")

def train(train_loader, model, cox_criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    coxlosses = AverageMeter("coxLoss", ":.4e")
    vaelosses = AverageMeter("vaeLoss", ":.4e")

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, coxlosses, vaelosses],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)


        output, vae_recon, mu, logvar, attention_weights = model(images)
        output = output[:, 1]


        cox_loss = cox_criterion(output, labels.cuda())
        vae_loss = vae_loss_function(vae_recon, images, mu, logvar)

        total_loss = cox_loss + vae_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        coxlosses.update(cox_loss.item(), images[0].size(0))
        vaelosses.update(vae_loss.item(), images[0].size(0))

        batch_time.update(time.time() - end)
        end = time.time()

    progress.display(i + 1)



def inference_ci(data_loader, model,  flag):
    # model.eval()
    features = []
    risks = []
    #all_labels = []
    #all_outputs = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device).float()
            labels = labels.to(device)


            output, *_ = model(images)
            output = output[:, 1]
            if flag:
                ci_value = concordance_index(labels.view(-1).cpu().numpy(), -output.cpu().numpy())
                p_value = cal_pval(labels.view(-1).cpu().numpy(), output.cpu().numpy())
                print(f"Test CI value: {ci_value}, p-value: {p_value}")
            else:

                ci_value = concordance_index(labels.view(-1).cpu().numpy(), output.cpu().numpy())
                print(f"Train CI value: {ci_value}")

    return ci_value


def inference_auc(data_loader, model, flag):
    model.eval()
    auc_values = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device).float()
            labels = labels.to(device)


            output, *_ = model(images)
            output = output[:, 1]
            binary_labels = (labels > 0).float()

            if flag:

                auc_value = roc_auc_score(binary_labels.view(-1).cpu().numpy(), output.view(-1).cpu().numpy())
                print(f"Test AUC: {auc_value}")
            else:

                auc_value = roc_auc_score(binary_labels.view(-1).cpu().numpy(), output.view(-1).cpu().numpy())
                print(f"Train AUC: {auc_value}")

    return auc_value


def inference_ibs(data_loader, model, flag):
    model.eval()
    scores = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device).float()
            labels = labels.to(device)


            output, *_ = model(images)
            output = torch.sigmoid(output[:, 1]).cpu().numpy()


            event = (labels > 0).cpu().numpy().astype(bool)
            survival_time = np.abs(labels.cpu().numpy())

            y_true = np.array([(e, t) for e, t in zip(event, survival_time)],
                              dtype=[('event', bool), ('time', float)])

            times = np.linspace(0, np.max(survival_time), num=100).astype(float)


            for t in times:
                mask = y_true['time'] >= t
                if mask.sum() == 0:
                    continue


                brier_score = brier_score_loss(y_true['event'][mask], output[mask])
                scores.append(brier_score)


                clipped_output = np.clip(output[mask], 0, 1)
                brier_score = brier_score_loss(y_true['event'][mask], clipped_output)
                scores.append(brier_score)

            if flag:
                ibs_value = np.mean(scores) if scores else np.nan
                print(f"Test IBS: {ibs_value}")
            else:
                ibs_value = np.mean(scores) if scores else np.nan
                print(f"Train IBS: {ibs_value}")



    return ibs_value


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    args = parser.parse_args()
    args.seed = 7
    #args.KFlod = 5
    args.pre_method = "pca_all"
    args.cancer = "kirc"
    args.zuxue = 5
    args.imgsize = 120
    args.gpu = 0
    args.moco_dim = 64
    args.nums_layer = 2
    args.lr = 3e-4
    args.momentum = 0.9  # SGD
    args.weight_decay = 1e-4
    args.start_epoch = 0
    args.batch_size = 32
    args.schedule = [120, 160]
    args.cos = True
    args.epochs = 700
    main(args)