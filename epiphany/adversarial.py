#!/usr/bin/env python
# -*- coding: utf-8 -*-
print("Starting adversarial.py")
import os

# Force CPU-only mode to test
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
import torch
# print("Torch version: ", torch.__version__)
import torch.nn as nn
# print("Torch version: ", torch.__version__)
from torchvision import datasets
# print("Torchvision version: ", datasets.__version__)
from torchvision import transforms
# print("Torchvision version: ", transforms.__version__)
import numpy as np
# print("Numpy version: ", np.__version__)
import torch.nn.functional as F
# print
import torch.optim as optim
# print("Torch version: ", optim.__version__)
import argparse
# print("Argparse version: ", argparse.__version__)
import epiphany_utils as eutils
# print("Utils version: ", utils.__version__)
import time
# print("Time module imported")

print(torch.__version__)

def safe_mean(data):
    """Safely compute mean of tensor or array"""
    if isinstance(data, torch.Tensor):
        return torch.mean(data).detach().cpu().item()
    elif isinstance(data, (list, tuple)) and len(data) > 0:
        if isinstance(data[0], torch.Tensor):
            # List of tensors
            stacked = torch.stack(data)
            return torch.mean(stacked).detach().cpu().item()
        else:
            # List of numbers
            return np.mean(data)
    else:
        return float(np.mean(data)) if hasattr(data, '__len__') else float(data)

def safe_min(data):
    """Safely compute min of tensor or array"""
    if isinstance(data, torch.Tensor):
        return torch.min(data).detach().cpu().item()
    elif isinstance(data, (list, tuple)) and len(data) > 0:
        if isinstance(data[0], torch.Tensor):
            stacked = torch.stack(data)
            return torch.min(stacked).detach().cpu().item()
        else:
            return min(data)
    else:
        return float(min(data)) if hasattr(data, '__len__') else float(data)
    # ... (rest of your code unchanged)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", help="CUDA ID", default="0")
    parser.add_argument("-b", "--batchSize", help="Batch size", default="1")
    parser.add_argument("-e", "--epochs", help="Number of epochs", default="55")
    parser.add_argument("-l", "--lr", help="Initial learning rate", default="1e-4")
    parser.add_argument("-v", "--version", help="Experiment version", default="0.1")
    parser.add_argument("--lam", help="Tradeoff between l2 and adversarial loss", default="0.95")
    parser.add_argument("-w", "--windowSize", help="Context (in terms of 100kb) for each orthogonal vector", default="14000")
    parser.add_argument("-m", "--message", help="Additional comments", default="")
    parser.add_argument("--highRes", action='store_true', help="Use if predicting 5kb resolution Hi-C (10kb is used by default)")
    parser.add_argument("--wandb", action='store_true', help="Toggle wandb")
    parser.add_argument("-x", "--xFile", help="X input file name", default="GM12878_X.h5")
    parser.add_argument("-y", "--yFile", help="y input file name", default="GM12878_y.pickle")
    parser.add_argument("-p", "--pretrainedModel", help="Path to pretrained model (e.g., 1_5kb_Akita_ICE_microC_mean.pt)", default=None)
    parser.add_argument("-o", "--outputFolder", help="Output folder for logs and models", default="./logs")
    parser.add_argument("-tr", "--trainingChromosomes", nargs='+', default=['chr1', 'chr2', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22'], help="List of chromosomes for training")
    parser.add_argument("-te", "--validationChromosomes", nargs='+', default=['chr3', 'chr11', 'chr17'], help="List of chromosomes for testing")
    args = parser.parse_args()

    print("Arguments loaded")
    # Import based on resolution choice
    if args.highRes:
        print("Using 5kb resolution Hi-C")
        try:
            import data_loader_5kb as data_loader
            import model_5kb as model_module
            from model_5kb import Net, Disc  # Import specific classes
            from data_loader_5kb import Chip2HiCDataset
        except ImportError as e:
            print(f"Error importing 5kb modules: {e}")
            return
    else:
        print("Using 10kb resolution Hi-C")
        try:
            import data_loader_10kb as data_loader
            import model_10kb as model_module
            from model_10kb import Net, Disc
            from data_loader_10kb import Chip2HiCDataset
        except ImportError as e:
            print(f"Error importing 10kb modules: {e}")
            return
    print("Modules imported")
    
    if args.wandb:
        print("Using Weights and Biases for logging")
        import wandb
        wandb.init()

    print("Run: " + args.message)

    LEARNING_RATE = float(args.lr)
    EXPERIMENT_VERSION = args.version
    LOG_PATH = os.path.join(args.outputFolder, EXPERIMENT_VERSION)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    LAMBDA = float(args.lam)
    TRAIN_SEQ_LENGTH = 200 
    TEST_SEQ_LENGTH = 200 

    torch.cuda.set_device(int(args.gpu))

    torch.manual_seed(0)    
    model = Net(1, 5, int(args.windowSize)).cuda()
    disc = Disc().cuda()
    if args.wandb:
        wandb.watch(model, log='all')

    if os.path.exists(LOG_PATH):
        eutils.restore_latest(model, LOG_PATH, ext='.pt_model')

    with open(os.path.join(LOG_PATH, 'setup.txt'), 'a+') as f:
        f.write("\nVersion: " + args.version)
        f.write("\nBatch Size: " + args.batchSize)
        f.write("\nInitial Learning Rate: " + args.lr)
        f.write("\nComments: " + args.message)

    # GM12878 Standard
    test_chroms = args.validationChromosomes
    train_chroms = args.trainingChromosomes

    train_set = Chip2HiCDataset(seq_length=TRAIN_SEQ_LENGTH, window_size=int(args.windowSize), X_data=args.xFile, y_data=args.yFile, chroms=train_chroms, mode='train') 
    test_set = Chip2HiCDataset(seq_length=TEST_SEQ_LENGTH, window_size=int(args.windowSize), X_data=args.xFile, y_data=args.yFile, chroms=test_chroms, mode='test') 

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    train_log = os.path.join(LOG_PATH, 'train_log.txt')
    test_log = os.path.join(LOG_PATH, 'test_log.txt')

    hidden = None
    log_interval = 20
    parameters = list(model.parameters()) 
    optimizer = optim.Adam(parameters, lr=LEARNING_RATE, weight_decay=0.0005)
    disc_optimizer = optim.Adam(disc.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
    min_loss = -10

    t0 = time.time()
    #scaler = torch.cuda.amp.GradScaler()
    for epoch in range(int(args.epochs)):

        disc_preds_train = []

        lr = np.maximum(LEARNING_RATE * np.power(0.5, (int(epoch / 16))), 1e-6) # learning rate decay
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=0.0005)
        disc_optimizer = optim.Adam(disc.parameters(), lr=lr, weight_decay=0.0005)

        print("="*10 + "Epoch " + str(epoch) + "="*10)

        im = []
        test_loss = []
        preds = []
        labs = []
        model.eval()
        for test_i, (test_data, test_label) in enumerate(test_loader):

            # Don't plot empty images
            if np.linalg.norm(test_label) < 1e-8:
                continue
            
            test_data, test_label = torch.Tensor(test_data[0]).cuda(), torch.Tensor(test_label).cuda()

            with torch.no_grad():
                pred, hidden = model(test_data, hidden_state=None,seq_length=TEST_SEQ_LENGTH)
                loss = model.loss(pred, test_label, seq_length=TEST_SEQ_LENGTH)
                test_loss.append(loss)        
                
                # Plot 5 images on wandb
                if args.wandb:
                    if test_i < 5:
                        im.append(wandb.Image(eutils.generate_image(test_label.cpu(), pred.detach().cpu(), LOG_PATH, TEST_SEQ_LENGTH, bands=100)))
                    else:
                        break

        if args.wandb:
            wandb.log({"Validation Examples": im})
            wandb.log({'val_correlation': safe_mean(test_loss)})
        
        print('Test Loss: ', safe_mean(test_loss), ' Best: ', str(min_loss))

        if safe_mean(test_loss) > min_loss:
            min_loss = safe_mean(test_loss)

        eutils.save(model, os.path.join(LOG_PATH, '%03d.pt_model' % epoch), num_to_keep=1)
        with open(test_log, 'a+') as f:
            f.write(str(safe_mean(test_loss)) + "\n")
       
        losses = []
        model.train()
        disc.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            if (np.linalg.norm(data)) < 1e-8:
                continue

            hidden = None
            label = torch.Tensor(np.squeeze(label)).cuda()
            data = data[0].cuda()
            optimizer.zero_grad()
                  
            output, hidden = model(data,seq_length=TRAIN_SEQ_LENGTH)
            output = torch.squeeze(output)

            # 1 -> real, 0 -> fake

            # Train generator
            mse_loss = model.loss(output, label, seq_length=TRAIN_SEQ_LENGTH)
            disc_out = disc(output.view(1,1,output.shape[0], output.shape[1]))
            adv_loss = F.binary_cross_entropy_with_logits(disc_out.view(1), torch.Tensor([1]).cuda()) # how close is disc pred to 1      
            loss = (LAMBDA)*mse_loss + (1 - LAMBDA)*adv_loss

            loss.backward()
            optimizer.step()

            # Train discriminator
            disc_optimizer.zero_grad()

            true_pred = disc(label.view(1,1,label.shape[0], label.shape[1]))
            fake_pred = disc(output.detach().view(1,1,output.shape[0], output.shape[1]))   
            disc_preds = torch.cat((true_pred, fake_pred), dim=0)  
            disc_loss = disc.loss(disc_preds, torch.Tensor([1, 0]).view(2,1).cuda())
            disc_loss.backward()

            disc_preds_train.append(torch.sigmoid(true_pred).item())
            disc_preds_train.append(torch.sigmoid(fake_pred).item())
            disc_optimizer.step()

            if args.wandb:
                wandb.log({'mse_loss': mse_loss.item()}) 
                wandb.log({'adv_loss': adv_loss.item()})
                wandb.log({'L_G': loss.item()})
                wandb.log({'L_D': disc_loss.item()})

            if batch_idx % log_interval == 0:

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item()))
 
    t1 = time.time()
    print(t1 - t0)       
 
if __name__ == '__main__':
    print("Starting adversarial training")
    main()
