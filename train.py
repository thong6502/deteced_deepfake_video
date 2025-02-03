from src.model import MyConvNeXt
from src.dataset import MyDataset

import torch
import torch.nn as nn
from argparse import ArgumentParser
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap='Wistia')
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, f"{cm[i, j]:.2f}", horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if writer:
        writer.add_figure('Confusion_Matrix', figure, epoch)

    return figure

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset224", help="data directory")
    parser.add_argument("--batch_size", "-b", type=int, default=20, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--ratio", type=float, default=0.8, help="train data ratio")
    parser.add_argument("--log_dir", type=str, default="logs", help="log directory for tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_model", help="model directory")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="checkpoint path")

    args = parser.parse_args()
    return args

def train(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    data_dir = args.data_dir
    log_dir = args.log_dir
    saved_path = args.saved_path
    start_epoch = 0
    best_acc = 0    

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    dataset = MyDataset(dataset_path=data_dir)
    train_size = int(len(dataset) * args.ratio)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MyConvNeXt(len(dataset.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=log_dir)

    if args.checkpoint:
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_acc = checkpoint.get("best_acc", 0)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_acc = []
        train_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", colour="green")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predicts = torch.argmax(outputs, dim=1)
            acc = accuracy_score(labels.cpu().numpy(), predicts.cpu().numpy())
            train_acc.append(acc)
            train_losses.append(loss.item())
            progress_bar.set_postfix(acc=np.mean(train_acc), loss=np.mean(train_losses))

        writer.add_scalar("Train/Acc", np.mean(train_acc), epoch)
        writer.add_scalar("Train/Loss", np.mean(train_losses), epoch)

        # Validation
        model.eval()
        val_losses = []
        all_labels = []
        all_predicts = []
        for images, labels in val_loader:
            with torch.no_grad():
                images, labels = images.to(device), labels.to(device)
                predicts = model(images)
                loss = criterion(predicts, labels)
                val_losses.append(loss.item())

                indices = torch.argmax(predicts, dim=1)
                all_predicts.extend(indices.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = np.mean(val_losses)
        avg_acc = accuracy_score(all_labels, all_predicts)
        writer.add_scalar("Val/Acc", avg_acc, epoch)
        writer.add_scalar("Val/Loss", avg_loss, epoch)
        print(f"Val/Acc = {avg_acc:.4f},  Val/Loss = {avg_loss:.4f}")

        cm = confusion_matrix(all_labels, all_predicts)
        plot_confusion_matrix(writer, cm, dataset.classes, epoch)

        # Save checkpoints
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc
        }
        torch.save(checkpoint, os.path.join(saved_path, "last.pt"))
        if best_acc < avg_acc:
            best_acc = avg_acc
            torch.save(checkpoint, os.path.join(saved_path, "best.pt"))

    writer.close()


if __name__ == "__main__":
    torch.manual_seed(42)
    args = get_args()
    train(args)
