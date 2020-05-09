#PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
#Common
import pandas as pd
import argparse
#Visualization
#from visdom import Visdom

#viz = Visdom()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(98, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 1)
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(300)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc3(x)
        #x = self.sigmoid(x)
        return x

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.df = pd.read_csv("train_data.csv", delimiter=",")
        #self.data_num = len(self.df)
        self.data_num = 10000

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = torch.tensor(self.df[["馬番","枠番","年齢","性別","馬体重","斤量","場所","頭数","距離","馬場状態","天候","人気","トラックコード",\
                                        "馬番1","枠番1","年齢1","性別1","馬体重1","斤量1","場所1","頭数1","距離1","馬場状態1","天候1","人気1","トラックコード1","単勝オッズ1","確定着順1","タイムS1","着差タイム1",\
                                        "馬番2","枠番2","年齢2","性別2","馬体重2","斤量2","場所2","頭数2","距離2","馬場状態2","天候2","人気2","トラックコード2","単勝オッズ2","確定着順2","タイムS2","着差タイム2",\
                                        "馬番3","枠番3","年齢3","性別3","馬体重3","斤量3","場所3","頭数3","距離3","馬場状態3","天候3","人気3","トラックコード3","単勝オッズ3","確定着順3","タイムS3","着差タイム3",\
                                        "馬番4","枠番4","年齢4","性別4","馬体重4","斤量4","場所4","頭数4","距離4","馬場状態4","天候4","人気4","トラックコード4","単勝オッズ4","確定着順4","タイムS4","着差タイム4",\
                                        "馬番5","枠番5","年齢5","性別5","馬体重5","斤量5","場所5","頭数5","距離5","馬場状態5","天候5","人気5","トラックコード5","単勝オッズ5","確定着順5","タイムS5","着差タイム5"]].iloc[idx])
        #out_label = torch.tensor([1.0 if self.df["確定着順"].iloc[idx] < 4 else 0.0])
        out_label = torch.tensor([self.df["確定着順"].iloc[idx]])
        return out_data, out_label

def train(args, model, device, train_loader, optimizer, epoch):
    criterion = nn.MSELoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        print("###target###")
        print(target)
        print("###output###")
        print(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        #print(batch_idx)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of horse racing prediction')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    data_set = MyDataset()
    train_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(torch.cuda.is_available())

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        #test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "horse_racing_prediction.pt")

if __name__ == '__main__':
    main()