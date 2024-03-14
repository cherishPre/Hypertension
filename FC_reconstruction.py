import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from torch import nn, optim
from pathlib import Path
from tqdm import tqdm
import os
import time
from dataloader import TCMDataloader
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger = logging.getLogger()

class FeatureReconstructionLoss(nn.Module):
    def __init__(self,hidden_dims):
        super(FeatureReconstructionLoss, self).__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=[hidden_dims], elementwise_affine=False)

    def forward(self, input_features, reconstructed_features):
        loss = torch.mean(torch.absolute(self.layernorm(input_features) - self.layernorm(reconstructed_features)))
        return loss

class RUnit(nn.Module):
    def __init__(self, in_dim=3, mid_dim=3, out_dim=1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, mid_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return out

class FC(nn.Module):
    def __init__(self, modality=[], ask_dim = 24,
        face_dim=4500, face_h=64, face_out=3, face_dropout=0.1,
        top_dim=1500, top_h=32, top_out=3, top_dropout=0.1,
        bottom_dim=3, 
        pulse_dim=48, pulse_h=12, pulse_out=2, pulse_dropout=0.1,
        final_h=32, final_dropout=0.1, num_classes=4
    ):
        super().__init__()
        self.modality = modality
        if 'face' in modality:
            self.face = nn.Sequential(
                nn.Conv2d(3, 3, (50, 1), stride=(50, 1)),
                nn.ReLU(),
            )
            self.face_fc = nn.Sequential(
                nn.Linear(90, face_h),
                nn.Dropout(face_dropout),
                nn.ReLU(),
                nn.BatchNorm1d(face_h),
                nn.Linear(face_h, face_out)
            )
        if 'top' in modality:
            self.top = nn.Sequential(
                nn.Conv2d(3, 3, (50, 1), stride=(50, 1)),
                nn.ReLU(),
            )
            self.top_fc = nn.Sequential(
                nn.Linear(30, top_h),
                nn.Dropout(top_dropout),
                nn.ReLU(),
                nn.BatchNorm1d(top_h),
                nn.Linear(top_h, top_out)
            )
        if 'pulse' in modality:
            self.pulse_fc = nn.Sequential(
                nn.Linear(pulse_dim, pulse_h),
                nn.Dropout(pulse_dropout),
                nn.ReLU(),
                nn.BatchNorm1d(pulse_h),
                nn.Linear(pulse_h, pulse_out)
            )
        
        concat_dim = ask_dim
        if 'face' in modality:
            concat_dim += face_out
        if 'top' in modality:
            concat_dim += top_out
        if 'bottom' in modality:
            concat_dim += bottom_dim
        if 'pulse' in modality:
            concat_dim += pulse_out

        self.runit = RUnit(in_dim=concat_dim, mid_dim=concat_dim*2, out_dim=concat_dim)
        self.recon = FeatureReconstructionLoss(concat_dim)

        self.fc = nn.Sequential(
            nn.Linear(concat_dim, final_h),
            nn.Dropout(final_dropout),
            nn.ReLU(),
            nn.BatchNorm1d(final_h),
            nn.Linear(final_h, num_classes)
        )

    def forward(self, x1, x2=None, x3=None, x4=None, x5=None):
        x = x1
        b_size = x1.shape[0]
        if 'face' in self.modality:
            x2 = x2.reshape(b_size, 1500, 3, 1).permute(0, 2, 1, 3)
            x2 = self.face(x2).view(b_size, -1)
            x2 = self.face_fc(x2)
            x = torch.cat([x, x2], dim=1)
        if 'top' in self.modality:
            x3 = x3.reshape(b_size, 500, 3, 1).permute(0, 2, 1, 3)
            x3 = self.top(x3).view(b_size, -1)
            x3 = self.top_fc(x3)
            x = torch.cat([x, x3], dim=1)
        if 'bottom' in self.modality:
            x4 = x4.view(b_size, -1)
            x = torch.cat([x, x4], dim=1)
        if 'pulse' in self.modality:
            x5 = x5.view(b_size, -1)
            x5 = self.pulse_fc(x5)
            x = torch.cat([x, x5], dim=1)

        temp = x.clone()
        mask = torch.rand(x.shape) < 0.1
        x[mask] = 0
        x = self.runit(x)
        rloss = self.recon(x,temp)

        x = self.fc(x)
        return x,rloss

def evaluate_dev(modality, net, dev_loader, loss, device, f1=False, details=False):
    dev_loss_sum, dev_acc_sum, n_samples = 0.0, 0.0, 0
    y_pred, y_true = [], []
    with torch.no_grad():
        net.eval()
        for batch_data in dev_loader:
            b_size = len(batch_data['id'])
            X1 = batch_data['ask'].view(b_size, -1).to(device)
            if 'face' in modality:
                X2 = batch_data['face'].view(b_size, -1).to(device)
            else:
                X2 = None
            if 'top' in modality:
                X3 = batch_data['top'].view(b_size, -1).to(device)
            else:
                X3 = None
            if 'bottom' in modality:
                X4 = batch_data['bottom'].view(b_size, -1).to(device)
            else:
                X4 = None
            if 'pulse' in modality:
                X5 = batch_data['pulse'].view(b_size, -1).to(device)
            else:
                X5 = None
            y = batch_data['label']
            y = y.to(device)
            y_hat,rloss = net(X1, X2, X3, X4, X5)
            l = loss(y_hat, y)
            y_pred.append(y_hat.cpu())
            y_true.append(y.cpu())
            dev_loss_sum += l
            dev_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n_samples += y.shape[0]
    if f1:
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        y_pred_4 = np.argmax(pred, axis=1)
        f1_weighted = f1_score(true, y_pred_4, average='weighted')
    else:
        f1_weighted = None
    if details:
        c_matrix = confusion_matrix(true, y_pred_4)
        c_report = classification_report(true, y_pred_4)
        logger.info(c_matrix)
        logger.info(c_report)
        per_class_accuracies = []
        for idx in range(4):
            true_negatives = np.sum(np.delete(np.delete(c_matrix, idx, axis=0), idx, axis=1))
            true_positives = c_matrix[idx, idx]
            per_class_accuracies.append((true_positives + true_negatives) / np.sum(c_matrix))
        logger.info(per_class_accuracies)
    return dev_loss_sum / len(dev_loader), dev_acc_sum / n_samples, f1_weighted

def train_model(modality, net, train_loader, dev_loader, test_loader, loss, optimizer, n_epochs, device, early_stop=-1, scheduler=None, model_save_path=None):
    net = net.to(device)
    logger.info(f"Training on {device}")
    best_dev_acc, best_epoch, best_test_acc = 0.0, 0, 0.0
    for epoch in range(1, n_epochs+1):
        train_loss_sum, train_acc_sum, n_samples = 0.0, 0.0, 0
        net.train()
        for batch_data in tqdm(train_loader):
            b_size = len(batch_data['id'])
            X1 = batch_data['ask'].view(b_size, -1).to(device)
            if 'face' in modality:
                X2 = batch_data['face'].view(b_size, -1).to(device)
            else:
                X2 = None
            if 'top' in modality:
                X3 = batch_data['top'].view(b_size, -1).to(device)
            else:
                X3 = None
            if 'bottom' in modality:
                X4 = batch_data['bottom'].view(b_size, -1).to(device)
            else:
                X4 = None
            if 'pulse' in modality:
                X5 = batch_data['pulse'].view(b_size, -1).to(device)
            else:
                X5 = None
            y = batch_data['label']
            y = y.to(device)
            y_hat, rloss= net(X1, X2, X3, X4, X5)
            l = loss(y_hat, y)
            l += rloss * 0.7
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_loss_sum += l
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n_samples += y.shape[0]
        train_loss = train_loss_sum / len(train_loader)
        train_acc = train_acc_sum/n_samples

        dev_loss, dev_acc, _ = evaluate_dev(modality, net, dev_loader, loss, device)
        test_loss, test_acc, _ = evaluate_dev(modality, net, test_loader, loss, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if scheduler is not None:
            scheduler.step(train_loss)

        logger.info("Epoch %d: train_loss: %.4f, train_acc: %.4f, dev_loss: %.4f, dev_acc: %.4f, test_acc: %.4f" % (
            epoch, train_loss, train_acc, dev_loss, dev_acc, test_acc))
        
        if early_stop > 0:
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_epoch = epoch
                if model_save_path is not None:
                    torch.save(net.state_dict(), model_save_path)
            elif epoch - best_epoch >= early_stop:
                logger.info(f"Early Stopped. best_epoch: {best_epoch}. best_dev_acc: {best_dev_acc}. best_test_acc: {best_test_acc}\n")
                return

    logger.info(f"Finished. best_epoch: {best_epoch}. best_dev_acc: {best_dev_acc}. best_test_acc: {best_test_acc}\n")


def main(modality=[], seed=2022, batch_size=32, n_epochs=300, lr=0.0005, early_stop=30):
    torch.manual_seed(seed)
    net = FC(modality=modality)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    # while logger.hasHandlers():
    #     logger.removeHandler(logger.handlers[0])
    fh = logging.FileHandler(Path('./train_log', f"ask-{'-'.join(modality)}.log"))
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    # Hyper Params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    model_save_path = f"./model_save/ask-{'-'.join(modality)}.pt"

    logger.info("\n")
    logger.info("==================== Start Training ====================")
    logger.info(f"seed: {seed}")
    logger.info(f"model: {net}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"n_epochs: {n_epochs}")
    logger.info(f"lr: {lr}")
    logger.info(f"early_stop: {early_stop}")
    logger.info(f"loss_fn: {loss_fn}")
    logger.info(f"optimizer: {optimizer}")
    logger.info(f"scheduler: {scheduler}")

    train_loader, val_loader, test_loader = TCMDataloader(
        root_path = "./data",
        modality = modality,
        batch_size = batch_size,
        seed = seed
    )

    train_model(modality, net, train_loader, val_loader, test_loader, loss_fn, optimizer, n_epochs, device, early_stop, scheduler, model_save_path)

    net.load_state_dict(torch.load(model_save_path))
    test_loss, test_acc, f1_weighted = evaluate_dev(modality, net, test_loader, loss_fn, device, f1=True, details=True)
    logger.info(f"test_loss: {test_loss}, test_acc: {test_acc}, f1_weighted: {f1_weighted}")


if __name__ == "__main__":
    modality_list = [
        ['face', 'top', 'bottom', 'pulse']
    ]
    for modality in modality_list:
        main(modality=modality, seed=2022)