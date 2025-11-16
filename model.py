from sympy import intersection
from Dataset.dataLoader import *
from Dataset.makeGraph import *
from Networks.Architectures.basicNetwork import *
from Networks.Architectures.U_Net import *
from Networks.Architectures.Seg_Net import *
from Networks.Architectures.Attention_U_Net import *
from Networks.Architectures.Deep_U_Net import *

import numpy as np
np.random.seed(2885)
import os
import copy

import torch
torch.manual_seed(2885)
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim


# --------------------------------------------------------------------------------
# CREATE A FOLDER IF IT DOES NOT EXIST
# INPUT: 
#     - desiredPath (str): path to the folder to create
# --------------------------------------------------------------------------------
def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


######################################################################################
#
# CLASS DESCRIBING THE INSTANTIATION, TRAINING AND EVALUATION OF THE MODEL 
# An instance of Network_Class has been created in the main.py file
# 
######################################################################################

class Network_Class: 
    # --------------------------------------------------------------------------------
    # INITIALISATION OF THE MODEL 
    # INPUTS: 
    #     - param (dic): dictionnary containing the parameters defined in the 
    #                    configuration (yaml) file
    #     - imgDirectory (str): path to the folder containing the images 
    #     - maskDirectory (str): path to the folder containing the masks
    #     - resultsPath (str): path to the folder containing the results of the 
    #                          experiement
    # --------------------------------------------------------------------------------
    def __init__(self, param, imgDirectory, maskDirectory, resultsPath):
        # ----------------
        # USEFUL VARIABLES 
        # ----------------
        self.imgDirectory  = imgDirectory
        self.maskDirectory = maskDirectory
        self.resultsPath   = resultsPath
        self.epoch         = param["TRAINING"]["EPOCH"]
        self.device        = param["TRAINING"]["DEVICE"]
        self.lr            = param["TRAINING"]["LEARNING_RATE"]
        self.batchSize     = param["TRAINING"]["BATCH_SIZE"]

        # -----------------------------------
        # NETWORK ARCHITECTURE INITIALISATION
        # -----------------------------------
        self.model = Deep_U_Net(param).to(self.device)

        # -------------------
        # TRAINING PARAMETERS
        # -------------------
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # ----------------------------------------------------
        # DATASET INITIALISATION (from the dataLoader.py file)
        # ----------------------------------------------------
        self.dataSetTrain    = LLNDataset(imgDirectory, maskDirectory, "train", param) 
        self.dataSetVal      = LLNDataset(imgDirectory, maskDirectory, "val",   param)
        self.dataSetTest     = LLNDataset(imgDirectory, maskDirectory, "test",  param)
        self.trainDataLoader = DataLoader(self.dataSetTrain, batch_size=self.batchSize, shuffle=True,  num_workers=4)
        self.valDataLoader   = DataLoader(self.dataSetVal,   batch_size=self.batchSize, shuffle=False, num_workers=4)
        self.testDataLoader  = DataLoader(self.dataSetTest,  batch_size=self.batchSize, shuffle=False, num_workers=4)


    # ---------------------------------------------------------------------------
    # LOAD PRETRAINED WEIGHTS (to run evaluation without retraining the model...)
    # ---------------------------------------------------------------------------
    def loadWeights(self): 
        self.model.load_state_dict(torch.load(self.resultsPath + '/_Weights/wghts.pkl', weights_only=True))

    # -----------------------------------
    # TRAINING LOOP (dummy implementation)
    # -----------------------------------
    def train(self): 
        print("Starting training on", self.device)
        train_losses, val_losses = [], []

        best_val_loss = float('inf')
        best_wts = copy.deepcopy(self.model.state_dict())

        for epoch in range(self.epoch):
            print(f"\nEpoch [{epoch+1}/{self.epoch}]")
            # -----------------
            # TRAINING PHASE
            # -----------------
            self.model.train()
            running_loss = 0.0

            for imgs, masks, _, _ in self.trainDataLoader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, masks.long())
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(self.trainDataLoader)
            train_losses.append(avg_train_loss)

            # -----------------
            # VALIDATION PHASE
            # -----------------
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, masks, _, _ in self.valDataLoader:
                    imgs, masks = imgs.to(self.device), masks.to(self.device)
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, masks.long())
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(self.valDataLoader)
            val_losses.append(avg_val_loss)

            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # -----------------
            # SAVE BEST WEIGHTS
            # -----------------
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_wts = copy.deepcopy(self.model.state_dict())

        # -----------------
        # SAVE LEARNING CURVES
        # -----------------
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.resultsPath, 'loss_curves.png'))
        plt.close()

        # -----------------
        # SAVE BEST MODEL
        # -----------------
        wghtsPath = os.path.join(self.resultsPath, '_Weights')
        createFolder(wghtsPath)
        torch.save(best_wts, os.path.join(wghtsPath, 'wghts.pkl'))
        print("Training complete. Best model saved.")

    # -------------------------------------------------
    # EVALUATION PROCEDURE (dummy implementation)
    # -------------------------------------------------
    def evaluate(self):
        self.model.train(False)
        self.model.eval()
         
        allMasks, allMasksPreds, allTileNames, allResizedImgs = [], [], [], []
        for (images, masks, tileNames, resizedImgs) in self.testDataLoader:
            images      = images.to(self.device)
            outputs     = self.model(images)

            images      = images.to('cpu')
            outputs     = outputs.to('cpu')

            masksPreds   = torch.argmax(outputs, dim=1)

            allMasks.extend(masks.data.numpy())
            allMasksPreds.extend(masksPreds.data.numpy())
            allResizedImgs.extend(resizedImgs.data.numpy())
            allTileNames.extend(tileNames)
        
        allMasks       = np.array(allMasks)
        allMasksPreds  = np.array(allMasksPreds)
        allResizedImgs = np.array(allResizedImgs)
            
        # -------------------------
        # QUALITATIVE EVALUATION
        # -------------------------
        print("\nQualitative Evaluation:")
        savePath = os.path.join(self.resultsPath, "Test")
        reconstruct_from_tiles(allResizedImgs, allMasksPreds, allMasks, allTileNames, savePath)

        # -------------------------
        # QUANTITATIVE EVALUATION
        # -------------------------
        print("\nQuantitative Evaluation:")
        num_classes = len(np.unique(allMasks))
        eps = 1e-6
    
        iou_scores, dice_scores, precision_scores, recall_scores = [], [], [], []
        class_pixel_counts = []
        lines = []  # to store text output
    
        for cls in range(num_classes):
            intersection = np.logical_and(allMasksPreds == cls, allMasks == cls).sum()  # TP
            union = np.logical_or(allMasksPreds == cls, allMasks == cls).sum()  # TP + FP + FN
            pred_sum = (allMasksPreds == cls).sum()  # TP + FP
            true_sum = (allMasks == cls).sum()  # TP + FN
            class_pixel_counts.append(true_sum)
    
            # IoU
            iou = intersection / (union + eps)
            iou_scores.append(iou)
    
            # Dice Coefficient
            dice = (2.0 * intersection) / (pred_sum + true_sum + eps)
            dice_scores.append(dice)
    
            # Precision & Recall
            precision = intersection / (pred_sum + eps)
            recall = intersection / (true_sum + eps)
            precision_scores.append(precision)
            recall_scores.append(recall)
    
            text = (f"Class {cls}: IoU = {iou:.4f}, Dice = {dice:.4f}, "
                    f"Precision = {precision:.4f}, Recall = {recall:.4f}")
            lines.append(text)
    
        # Aggregate metrics (simple mean)
        mean_iou = np.mean(iou_scores)
        mean_dice = np.mean(dice_scores)
        mean_precision = np.mean(precision_scores)
        mean_recall = np.mean(recall_scores)
        pixel_acc = np.mean(allMasksPreds == allMasks)
    
        # Frequency-weighted averages
        class_pixel_counts = np.array(class_pixel_counts)
        weights = class_pixel_counts / (np.sum(class_pixel_counts) + eps)
    
        fw_iou = np.sum(weights * np.array(iou_scores))
        fw_dice = np.sum(weights * np.array(dice_scores))
        fw_precision = np.sum(weights * np.array(precision_scores))
        fw_recall = np.sum(weights * np.array(recall_scores))
    
        # Summary text
        lines.append("\nOverall Performance:")
        lines.append(f"Pixel Accuracy        : {pixel_acc:.4f}")
        lines.append(f"Mean IoU              : {mean_iou:.4f}")
        lines.append(f"Mean Dice             : {mean_dice:.4f}")
        lines.append(f"Mean Precision        : {mean_precision:.4f}")
        lines.append(f"Mean Recall           : {mean_recall:.4f}")
        lines.append("")
        lines.append("Frequency-Weighted Performance:")
        lines.append(f"FW IoU                : {fw_iou:.4f}")
        lines.append(f"FW Dice               : {fw_dice:.4f}")
        lines.append(f"FW Precision          : {fw_precision:.4f}")
        lines.append(f"FW Recall             : {fw_recall:.4f}")
    
        # Print to console
        print("\n".join(lines))
    
        # -------------------------
        # SAVE RESULTS TO FILE
        # -------------------------
        metricsPath = os.path.join(self.resultsPath, "_Metrics")
        os.makedirs(metricsPath, exist_ok=True)
        metrics_file = os.path.join(metricsPath, "evaluation_results.txt")
    
        with open(metrics_file, "w") as f:
            f.write("\n".join(lines))