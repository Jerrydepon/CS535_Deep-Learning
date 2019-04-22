from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt


class Net(nn.Module):

    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    #     self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
    #     self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
    #     self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.fc1 = nn.Linear(64 * 8 * 8, 512)
    #     self.fc2 = nn.Linear(512, 10)
    #
    #     self.bNorm = nn.BatchNorm2d(512)
    #     self.fcm = nn.Linear(512, 512)
    #
    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = F.relu(self.conv2(x))
    #     x = self.pool(x)
    #     x = F.relu(self.conv3(x))
    #     x = F.relu(self.conv4(x))
    #     x = self.pool(x)
    #     x = x.view(-1, self.num_flat_features(x))
    #     x = F.relu(self.fc1(x))
    #     x = self.bNorm(x)   # for Q1
    #     x = F.relu(self.fcm(x))   # for Q2
    #     x = self.fc2(x)
    #     return x

    # for Q4-1
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bNorm = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fcm = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        x = self.bNorm(x)
        x = F.leaky_relu(self.fcm(x))
        x = self.bNorm(x)
        x = self.fc2(x)
        return x

    # # for Q4-2
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
    #     self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
    #     self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
    #     self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
    #     self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
    #     self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
    #     self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
    #     self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.bNorm = nn.BatchNorm2d(512)
    #     self.fc1 = nn.Linear(512 * 2 * 2, 512)
    #     self.fcm = nn.Linear(512, 512)
    #     self.fc2 = nn.Linear(512, 10)
    #
    # def forward(self, x):
    #     x = F.leaky_relu(self.conv1(x))
    #     x = F.leaky_relu(self.conv2(x))
    #     x = self.pool(x)
    #     x = F.leaky_relu(self.conv3(x))
    #     x = F.leaky_relu(self.conv4(x))
    #     x = self.pool(x)
    #     x = F.leaky_relu(self.conv5(x))
    #     x = F.leaky_relu(self.conv6(x))
    #     x = self.pool(x)
    #     x = F.leaky_relu(self.conv7(x))
    #     x = F.leaky_relu(self.conv8(x))
    #     x = self.pool(x)
    #     x = x.view(-1, self.num_flat_features(x))
    #     x = F.leaky_relu(self.fc1(x))
    #     x = self.bNorm(x)
    #     x = F.leaky_relu(self.fcm(x))
    #     x = self.fc2(x)
    #     return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def eval_net(dataloader):
    correct = 0
    total = 0
    total_loss = 0
    net.eval()  # sets the mode to test
    criterion = nn.CrossEntropyLoss(size_average=False)

    for data in dataloader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)   # return largest element in each row and its index
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        total_loss += loss.data[0]
    net.train()  # sets the mode back to train
    return total_loss / total, correct / total


if __name__ == "__main__":
    # ==================== Initialization ====================
    BATCH_SIZE = 32  # mini_batch size
    MAX_EPOCH = 30  # maximum epoch to train

    loss_train_list = []
    accuracy_train_list = []
    loss_val_list = []
    accuracy_val_list = []

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # ==================== Load Dataset ====================
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # torchvision.transforms.Normalize(mean, std) for (r,g,b) channels

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    # ==================== Main Process ====================
    print('Building model...')
    net = Net().cuda()
    net.train()  # sets the mode to train (dropout & batchnorm)

    # # for Q2
    # # partially restore weights
    # pretrained_dict = torch.load('mytraining_1.pth')
    # model_dict = net.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # net.load_state_dict(model_dict)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.8)    # for Q1, Q2
    # optimizer = optim.Adam(net.parameters(), lr=0.0005)    # for Q3
    optimizer = optim.Adam(net.parameters(), lr=0.0003)  # for Q4

    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 500 == 499:  # print every 500 mini-batches
                # print('    Step: %5d avg_batch_loss: %.5f' %
                #       (i + 1, running_loss / 500))
                running_loss = 0.0

        # print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(trainloader)
        test_loss, test_acc = eval_net(testloader)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch + 1, train_loss, train_acc, test_loss, test_acc))

        loss_train_list.append(train_loss)
        accuracy_train_list.append(train_acc * 100)
        loss_val_list.append(test_loss)
        accuracy_val_list.append(test_acc * 100)

    print('Finished Training')
    print('Saving model...')
    torch.save(net.state_dict(), 'mytraining_4-1.pth')

    # ==================== Plot ====================
    plt.switch_backend('agg')

    fig = plt.figure()
    plt.title("Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(list(range(1, MAX_EPOCH + 1)), loss_train_list)
    plt.plot(list(range(1, MAX_EPOCH + 1)), loss_val_list)
    plt.legend(['training loss', 'validation loss'], loc='best')
    fig.savefig('./plot/q4-1_loss.png', dpi=fig.dpi)

    fig = plt.figure()
    plt.title("Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.plot(list(range(1, MAX_EPOCH + 1)), accuracy_train_list)
    plt.plot(list(range(1, MAX_EPOCH + 1)), accuracy_val_list)
    plt.legend(['training accuracy', 'validation accuracy'], loc='best')
    fig.savefig('./plot/q4-1_acc.png', dpi=fig.dpi)
