import sys
from git import Repo

Repo.clone_from(
    "https://github.com/Jonathan-Pearce/calibration_library.git",
    "/mnt/d/Calibration/calibration_library",
)

Repo.clone_from(
    "https://github.com/ptnv-s/metrics-saliency-maps.git",
    "/mnt/d/Calibration/metrics-saliency-maps",
)

sys.path.insert(1, "/mnt/d/Calibration/calibration_library")
sys.path.insert(2, "/mnt/d/Calibration/metrics-saliency-maps")
import torch
import warnings
import metrics
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from sklearn.metrics import classification_report
from captum.attr import LayerGradCam, LayerAttribution
from saliency_maps_metrics.single_step_metrics import IIC_AD, ADD
from saliency_maps_metrics.multi_step_metrics import Deletion, Insertion


torch.manual_seed(42)
warnings.simplefilter("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4
WD = 1e-6
BATCH_SIZE = 512
EPOCHS = 10
print(DEVICE)


def accuracy(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    acc = np.sum((true == pred).astype(np.float32)) / len(true)
    return acc * 100


def train(model, dataloader):
    model.to(DEVICE)
    optim = torch.optim.Adam(params=model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss()
    ece_criterion = metrics.ECELoss()
    model.train()
    for epoch in range(EPOCHS):
        print(f"{epoch+1}/{EPOCHS}")
        train_logits = []
        train_loss = []
        train_preds = []
        train_labels = []
        for batch in tqdm(dataloader):
            imgs = torch.Tensor(batch[0]).to(DEVICE)
            labels = torch.Tensor(batch[1]).to(DEVICE)
            scores = model(imgs)
            train_logits.append(scores.detach())
            loss = criterion(scores, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss.append(loss.detach().cpu().numpy())
            train_labels.append(batch[1])
            train_preds.append(scores.argmax(dim=-1))
        logits = torch.cat(train_logits)
        labels = torch.cat(train_labels)
        logits_np = logits.detach().cpu().numpy()
        labels_np = labels.cpu().numpy()
        loss = sum(train_loss) / len(train_loss)
        acc = accuracy(
            torch.concat(train_labels, dim=0).cpu(),
            torch.concat(train_preds, dim=0).cpu(),
        )
        print(
            f"\tTrain\tLoss - {round(loss, 3)}",
            "\t",
            f"Accuracy - {round(acc, 3)}",
        )
        print("ECE: %f" % (ece_criterion.loss(logits_np, labels_np, 15)))
        if epoch + 1 == EPOCHS:
            print(
                classification_report(
                    torch.concat(train_labels, dim=0).cpu(),
                    torch.concat(train_preds, dim=0).cpu(),
                    digits=3,
                )
            )
    torch.save(
        model.state_dict(),
        f"resnet18_cifar.pt",
    )
    torch.cuda.empty_cache()


def test(model, test_loader):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    ece_criterion = metrics.ECELoss()
    model.eval()
    with torch.no_grad():
        test_logits = []
        test_loss = []
        test_preds = []
        test_labels = []
        for batch in tqdm(test_loader):
            imgs = torch.Tensor(batch[0]).to(DEVICE)
            labels = torch.Tensor(batch[1]).to(DEVICE)
            scores = model(imgs)
            test_logits.append(scores.detach())
            loss = criterion(scores, labels)
            test_loss.append(loss.detach().cpu().numpy())
            test_labels.append(batch[1])
            test_preds.append(scores.argmax(dim=-1))
        logits = torch.cat(test_logits)
        labels = torch.cat(test_labels)
        logits_np = logits.detach().cpu().numpy()
        labels_np = labels.cpu().numpy()
        loss = sum(test_loss) / len(test_loss)
        acc = accuracy(
            torch.concat(test_labels, dim=0).cpu(),
            torch.concat(test_preds, dim=0).cpu(),
        )
        print(f"\tTest:\tLoss - {round(loss, 3)}", "\t", f"Accuracy - {round(acc,3)}")
        print("ECE: %f" % (ece_criterion.loss(logits_np, labels_np, 15)))
        print(
            classification_report(
                torch.concat(test_labels, dim=0).cpu(),
                torch.concat(test_preds, dim=0).cpu(),
                digits=3,
            )
        )
    torch.cuda.empty_cache()


def train_adversarial(
    model, train_loader, attack="PGD", save=False, save_name="resnet_cifar"
):
    print(attack)
    model.cuda()
    model.train()
    optim = torch.optim.Adam(params=model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss()
    ece_criterion = metrics.ECELoss()
    for epoch in range(EPOCHS):
        print(f"{epoch+1}/{EPOCHS}")
        train_logits = []
        train_loss = []
        train_preds = []
        train_labels = []
        for x, y in tqdm(train_loader):
            x = torch.Tensor(x).to(DEVICE)
            y = torch.Tensor(y).to(DEVICE)
            if attack == "PGD":
                x = projected_gradient_descent(model, x, 8 / 255, 0.01, 40, np.inf)
            elif attack == "FGSM":
                x = fast_gradient_method(model, x, 8 / 255, np.inf)
            scores = model(x)
            train_logits.append(scores.detach())
            loss = criterion(scores, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss.append(loss.detach().cpu().numpy())
            train_labels.append(y)
            train_preds.append(scores.argmax(dim=-1))
        logits = torch.cat(train_logits)
        labels = torch.cat(train_labels)
        logits_np = logits.detach().cpu().numpy()
        labels_np = labels.cpu().numpy()
        loss = sum(train_loss) / len(train_loss)
        acc = accuracy(
            torch.concat(train_labels, dim=0).cpu(),
            torch.concat(train_preds, dim=0).cpu(),
        )
        print(
            f"\tTrain\tLoss - {round(loss, 3)}",
            "\t",
            f"Accuracy - {round(acc, 3)}",
        )
        print("ECE: %f" % (ece_criterion.loss(logits_np, labels_np, 15)))
        if epoch + 1 == EPOCHS:
            print(
                classification_report(
                    torch.concat(train_labels, dim=0).cpu(),
                    torch.concat(train_preds, dim=0).cpu(),
                    digits=3,
                )
            )
        if save == True:
            torch.save(
                model.state_dict(),
                f"{save_name}-{attack}.pt",
            )
    torch.cuda.empty_cache()


def test_adversarial(
    model,
    test_loader,
    attack="PGD",
):
    print(attack)
    model.cuda()
    model.eval()
    criterion = nn.CrossEntropyLoss()
    ece_criterion = metrics.ECELoss()
    test_logits = []
    test_loss = []
    test_preds = []
    test_labels = []
    for x, y in tqdm(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        if attack == "FGSM":
            x = fast_gradient_method(model, x, 8 / 255, np.inf)
        elif attack == "PGD":
            x = projected_gradient_descent(model, x, 8 / 255, 0.01, 40, np.inf)
        scores = model(x)
        test_logits.append(scores.detach())
        loss = criterion(scores, y)
        test_loss.append(loss.detach().cpu().numpy())
        test_labels.append(y)
        test_preds.append(scores.argmax(dim=-1))
    logits = torch.cat(test_logits)
    labels = torch.cat(test_labels)
    logits_np = logits.detach().cpu().numpy()
    labels_np = labels.cpu().numpy()
    loss = sum(test_loss) / len(test_loss)
    acc = accuracy(
        torch.concat(test_labels, dim=0).cpu(),
        torch.concat(test_preds, dim=0).cpu(),
    )
    print(f"\tTest:\tLoss - {round(loss, 3)}", "\t", f"Accuracy - {round(acc,3)}")
    print("ECE: %f" % (ece_criterion.loss(logits_np, labels_np, 15)))
    print(
        classification_report(
            torch.concat(test_labels, dim=0).cpu(),
            torch.concat(test_preds, dim=0).cpu(),
            digits=3,
        )
    )
    torch.cuda.empty_cache()


def get_saliency_metrics(model, dataset):
    gradcam = LayerGradCam(model, model.conv4[0])
    inds = torch.randint(size=(1,), high=len(dataset))

    allImg = []
    allExpl = []
    allInds = []

    for ind in inds:
        batch = dataset.__getitem__(ind)
        img = batch[0].unsqueeze(0)
        class_ind = model(img).argmax(dim=-1)
        attr = gradcam.attribute(img, class_ind)

        allImg.append(img)
        allExpl.append(attr)
        allInds.append(class_ind)

    allImg = torch.cat(allImg, dim=0)
    allExpl = torch.cat(allExpl, dim=0)
    allInds = torch.cat(allInds, dim=0)

    iic_ad = IIC_AD()
    result_dic = iic_ad(model, allImg, allExpl, allInds)
    iic_mean, ad_mean = result_dic["iic"], result_dic["ad"]
    print("IIC", iic_mean)
    print("AD", ad_mean)

    add = ADD()
    result_dic = add(model, allImg, allExpl, allInds)
    add_mean = result_dic["add"]
    print("ADD", add_mean)

    allExpl = torch.nn.functional.interpolate(allExpl, (3, 3))

    deletion = Deletion()
    result_dic = deletion(model, allImg.clone(), allExpl.clone(), allInds)
    dauc_mean = result_dic["dauc"]
    dc_mean = result_dic["dc"]
    print("DAUC", dauc_mean)
    print("DC", dc_mean)

    insertion = Insertion()
    result_dic = insertion(model, allImg.clone(), allExpl.clone(), allInds)
    iauc_mean = result_dic["iauc"]
    ic_mean = result_dic["ic"]
    print("IAUC", iauc_mean)
    print("IC", ic_mean)
