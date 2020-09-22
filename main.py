import os
import argparse
import sys
import time
import arch
import cv2 
import numpy as np
import nsml
from nsml.constants import DATASET_PATH, GPU_NUM
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


######################## DONOTCHANGE ###########################
def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(),os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_state_dict(torch.load(os.path.join(dir_name, 'model')))
        model.eval()
        print('model loaded!')

    def infer(image_path):
        result = []
        with torch.no_grad():             
            batch_loader = DataLoader(dataset=PathDataset(image_path, labels=None),
                                        batch_size=batch_size,shuffle=False)
            # Train the model 
            for i, images in enumerate(batch_loader):
                y_hat = model(images.to(device)).cpu().numpy()
                result.extend(np.argmax(y_hat, axis=1))

        print('predicted')
        return np.array(result)

    nsml.bind(save=save, load=load, infer=infer)


def path_loader (root_path):
    image_path = []
    image_keys = []
    for _,_,files in os.walk(os.path.join(root_path,'train_data')):
        for f in files:
            path = os.path.join(root_path,'train_data',f)
            if path.endswith('.png'):
                image_keys.append(int(f[:-4]))
                image_path.append(path)

    return np.array(image_keys), np.array(image_path)


def label_loader (root_path, keys):
    labels_dict = {}
    labels = []
    with open (os.path.join(root_path,'train_label'), 'rt') as f :
        for row in f:
            row = row.split()
            labels_dict[int(row[0])] = (int(row[1]))
    for key in keys:
        labels = [labels_dict[x] for x in keys]
    return labels
############################################################

def stochastic_aug(im):
    # im = [3, 512, 512], torch.tensor
    # return [3, 512, 512]
    
    transform_list = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    
    return transform_list(im)

class PathDataset(Dataset): 
    def __init__(self,image_path, labels=None, test_mode= True): 
        '''
        randperm = np.random.permutation(len(image_path))
        train_len, val_len = int(len(image_path)*0.95), len(image_path) - int(len(image_path)*0.95)
        self.train_idx, self.val_idx = self.randperm[:train_len], self.randperm[:train_len]
        '''

        self.len = len(image_path)
        self.image_path = image_path
        self.labels = labels 
        self.mode = test_mode


    def __getitem__(self, index): 
        im = cv2.imread(self.image_path[index])
        im = im.reshape(3,im.shape[0],im.shape[1])
        
        # if not self.mode:
            # im = aug(im)
                ### REQUIRED: PREPROCESSING ###

        if self.mode:  # Test
            im = im / im.max()
            return torch.tensor(im,dtype=torch.float32)
        else:
            im = im / im.max()  # Train
            im = torch.tensor(im,dtype=torch.float32)
            # im = stochastic_aug(im)
            return im,\
                 torch.tensor(self.labels[index] ,dtype=torch.long)

    def __len__(self): 
        return self.len

if __name__ == '__main__':

    ########## ENVIRONMENT SETUP ############
    args = argparse.ArgumentParser()

    ########### DONOTCHANGE: They are reserved for nsml ###################
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    ######################################################################

    # hyperparameters
    args.add_argument('--epoch', type=int, default=200)
    args.add_argument('--batch_size', type=int, default=16) 
    args.add_argument('--learning_rate', type=float, default=0.00015)

    config = args.parse_args()

    # training parameters
    num_epochs = config.epoch
    batch_size = config.batch_size
    num_classes = 2
    learning_rate = config.learning_rate 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # model setting ## 반드시 이 위치에서 로드해야함
    model = arch.CNN().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    ############ DONOTCHANGE ###############
    bind_model(model)
    if config.pause: ## test mode 일때는 여기만 접근
        print('Inferring Start...')
        nsml.paused(scope=locals())
    #######################################

    if config.mode == 'train': ### training mode 일때는 여기만 접근
        print('Training Start...')

        ############ DONOTCHANGE: Path loader ###############
        root_path = os.path.join(DATASET_PATH,'train')
        image_keys, image_path = path_loader(root_path)
        labels = label_loader(root_path, image_keys)
        ##############################################
 
        dataset = PathDataset(image_path, labels, test_mode=False)
        train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.95), len(dataset) - int(len(dataset)*0.95)])
        print(f"Train set length : {len(train_set)}, Valid set length : {len(val_set)}")
        train_loader = DataLoader(\
                dataset=train_set, 
                batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(\
                dataset=val_set, 
                batch_size=batch_size, shuffle=False)

        '''
        batch_loader = DataLoader(\
            dataset=PathDataset(image_path, labels, test_mode=False), 
                batch_size=batch_size, shuffle=True)
        '''
        
        # Train the model
        for epoch in range(num_epochs):
            train_loss_list, val_loss_list = [], []
            train_acc_list, val_acc_list = [], []
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                images = stochastic_aug(images)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, outputs_label = outputs.max(dim=1)
                outputs_label = outputs_label.float()
                # (outputs > 0.5).float()
                labels_label = labels.float()

                right = (outputs_label == labels_label).float()
                acc = right.sum().item() / right.numel()

                train_loss_list.append(loss.item())
                train_acc_list.append(acc)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            train_loss = sum(train_loss_list) / len(train_loss_list)
            train_acc = sum(train_acc_list) / len(train_acc_list)

            for i, (images, labels) in enumerate(val_loader):
                with torch.no_grad():
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    _, outputs_label = outputs.max(dim=1)
                    outputs_label = outputs_label.float()
                    # (outputs > 0.5).float()
                    labels_label = labels.float()

                    right = (outputs_label == labels_label).float()
                    acc = right.sum().item() / right.numel()

                    val_loss_list.append(loss.item())
                    val_acc_list.append(acc)
                
            val_loss = sum(val_loss_list) / len(val_loss_list)
            val_acc = sum(val_acc_list) / len(val_acc_list)
            
            # print(loss.item(), acc.item())
            print(epoch, train_loss, train_acc, val_loss, val_acc)
            nsml.report(summary=True, step=epoch, epoch_total=num_epochs, train_loss=train_loss, train_acc=train_acc,
                                                                          val_loss=val_loss, val_acc=val_acc)#, acc=train_acc)
            nsml.save(epoch)