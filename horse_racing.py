#PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
#Common
import pandas as pd
import argparse

#ネットワークの定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(236, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 1)
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(300)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.bn1(x)
        # x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

#データセットの読み込み
class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.df = pd.read_csv("/home/uk/all_data/tmp_2018~.csv", delimiter=",")
        self.data_num = len(self.df)
        #self.data_num = 10000

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = torch.tensor(self.df[["馬番","枠番","年齢","馬体重","斤量","頭数","距離","人気","牡","牝","セ","札幌","函館","福島","新潟","東京","中山","中京","京都","阪神","小倉","良","稍","重","不","晴","曇","雨","小雨","小雪","雪","芝","ダート","障害直線芝","障害直線ダート","芝外回り",\
                                        "馬番1","枠番1","年齢1","馬体重1","斤量1","頭数1","距離1","人気1","単勝オッズ1","確定着順1","タイムS1","着差タイム1","牡1","牝1","セ1","札幌1","函館1","福島1","新潟1","東京1","中山1","中京1","京都1","阪神1","小倉1","良1","稍1","重1","不1","晴1","曇1","雨1","小雨1","小雪1","雪1","芝1","ダート1","障害直線芝1","障害直線ダート1","芝外回り1",\
                                        "馬番2","枠番2","年齢2","馬体重2","斤量2","頭数2","距離2","人気2","単勝オッズ2","確定着順2","タイムS2","着差タイム2","牡2","牝2","セ2","札幌2","函館2","福島2","新潟2","東京2","中山2","中京2","京都2","阪神2","小倉2","良2","稍2","重2","不2","晴2","曇2","雨2","小雨2","小雪2","雪2","芝2","ダート2","障害直線芝2","障害直線ダート2","芝外回り2",\
                                        "馬番3","枠番3","年齢3","馬体重3","斤量3","頭数3","距離3","人気3","単勝オッズ3","確定着順3","タイムS3","着差タイム3","牡3","牝3","セ3","札幌3","函館3","福島3","新潟3","東京3","中山3","中京3","京都3","阪神3","小倉3","良3","稍3","重3","不3","晴3","曇3","雨3","小雨3","小雪3","雪3","芝3","ダート3","障害直線芝3","障害直線ダート3","芝外回り3",\
                                        "馬番4","枠番4","年齢4","馬体重4","斤量4","頭数4","距離4","人気4","単勝オッズ4","確定着順4","タイムS4","着差タイム4","牡4","牝4","セ4","札幌4","函館4","福島4","新潟4","東京4","中山4","中京4","京都4","阪神4","小倉4","良4","稍4","重4","不4","晴4","曇4","雨4","小雨4","小雪4","雪4","芝4","ダート4","障害直線芝4","障害直線ダート4","芝外回り4",\
                                        "馬番5","枠番5","年齢5","馬体重5","斤量5","頭数5","距離5","人気5","単勝オッズ5","確定着順5","タイムS5","着差タイム5","牡5","牝5","セ5","札幌5","函館5","福島5","新潟5","東京5","中山5","中京5","京都5","阪神5","小倉5","良5","稍5","重5","不5","晴5","曇5","雨5","小雨5","小雪5","雪5","芝5","ダート5","障害直線芝5","障害直線ダート5","芝外回り5"]].iloc[idx])
        out_label = torch.tensor([1.0 if self.df["確定着順"].iloc[idx] <= 3 else 0.0])
        #out_label = torch.tensor([self.df["確定着順"].iloc[idx]])
        return out_data, out_label

def train(args, model, device, train_loader, optimizer, epoch):
    # criterion = nn.MSELoss()
    # criterion = F.binary_cross_entropy()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print("###target###")
        # print(target)
        # print("###output###")
        # print(output)
        loss = F.binary_cross_entropy(output, target)
        # loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        #print(batch_idx)
        if batch_idx % args.log_interval == 0:
            # print("###target###")
            # print(target)
            # print("###output###")
            # print(output)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# def test(args, model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of horse racing prediction')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 14)')
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
    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        #test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "horse_racing_prediction.pt")

if __name__ == '__main__':
    main()