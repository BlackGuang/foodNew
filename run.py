import torch
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import argparse

# from tensorboardX import SummaryWriter

from network import feature_net


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 参数设置
parser = argparse.ArgumentParser(description='cifar10')
parser.add_argument('--pre_epoch', default=0, help='begin epoch')
parser.add_argument('--total_epoch', default=1, help='time for epoch')
parser.add_argument('--model', default='vgg19', help='model for training')
parser.add_argument('--outf', default='./model', help='folder to output images and checkpoints')
parser.add_argument('--pre_model', default=False, help='use pre_model')
args = parser.parse_args()

# define use model
model = args.model

# if you have gpu then use gpu ,else use cpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load images
path = '/home/dl/qbg/dataset'

transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
data_image = {x: datasets.ImageFolder(root=os.path.join(path, x), transform=transform) for x in ["train", "val"]}
data_loader_image = {x: torch.utils.data.DataLoader(dataset=data_image[x],
                                                    batch_size=4,
                                                    shuffle=True) for x in ["train", "val"]}
classes = data_image["train"].classes
class_index = data_image["train"].class_to_idx
print(classes)
print(class_index)

# print train dataset and val dataset number
print("train data set:", len(data_image["train"]))
print("val data set:", len(data_image["val"]))

image_train, label_train = next(iter(data_loader_image["train"]))
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
img = torchvision.utils.make_grid(image_train)  # make batch_size img to one img
print(img.shape)
img = img.numpy().transpose((1, 2, 0))  # 本来是(0,1,2),相当于把第一维变为第三维，其他两维前移
print(img.shape)
img = img * std + mean
print([classes[i] for i in label_train])
plt.imshow(img)

# make net
use_model = feature_net(model, dim=512, n_classes=2)
for parma in use_model.feature.parameters():  # ???????
    parma.requires_grad = False

for index, parma in enumerate(use_model.classifier.parameters()):
    if index == 6:
        parma.requires_grad = True

if use_cuda:
    use_model = use_model.to(device)

# define loss
loss = torch.nn.CrossEntropyLoss()

# define you hua qi
optimizer = torch.optim.Adam(use_model.classifier.parameters())

print(use_model)

if args.pre_model:
    print("Resume from checkpoint...")
    assert os.path.isdir('checkpoint', 'Error:no checkpoint directory found')
    state = torch.load('./checkpoint/ckpt.t7')
    use_model.load_state_dict(state['state_dict'])
    best_test_acc = state['acc']
    pre_epoch = state['epoch']
else:
    # define the best test acc
    best_test_acc = 0
    pre_epoch = args.pre_epoch

if __name__ == '__main__':
    total_epoch = args.total_epoch
    # writer = SummaryWriter(log_dir='./log')
    print("Start Training,{}...".format(model))
    with open("acc.txt", "w")as acc_f:
        with open("log.txt", "w") as log_f:
            start_time = time.time()

            for epoch in range(pre_epoch, total_epoch):
                print("epoch{}/{}".format(epoch, total_epoch))

                print("_" * 10)
                # begin train
                sum_loss = 0.0
                accuracy = 0.0
                total = 0
                for i, data in enumerate(data_loader_image["train"]):
                    image, label = data
                    if use_cuda:
                        image, label = Variable(image.to(device)), Variable(label.to(device))
                    else:
                        image, label = Variable(image), Variable(label)

                    # forward way
                    optimizer.zero_grad()
                    label_prediction = use_model(image)

                    _, prediction = torch.max(label_prediction.data, 1)
                    total += label.size(0)  # label 每次读四个量
                    current_loss = loss(label_prediction, label)
                    # 后向传播
                    # optimizer.zero_grad()
                    current_loss.backward()
                    optimizer.step()

                    sum_loss += current_loss.item()
                    accuracy += torch.sum(prediction=label.data)

                    if total % 5 == 0:
                        print("total {},train loss:{:.4f},train accuracy:{:.4f}"
                              .format(total, sum_loss / total, 100 * accuracy / total))

                        # write rizhi
                        log_f.write("total{},train loss:{:.4f},train accuracy:{:.4f}"
                                    .format(total, sum_loss / total, 100 * accuracy / total))
                        log_f.write('\n')
                        log_f.flush()

                # write to tensorboard
                # writer.add_scalar('loss/train', sum_loss / (i + 1), epoch)
                # writer.add_scalar('accuracy/train', 100 * accuracy / total, epoch)
                # 每一个epoch测试的准确率
                print("waiting for testing")
                # 在上下文的环境中切断梯度计算，在此模式下，每一步的计算结果中
                # requires_gard都是false，即使input设置为requires_grad=True
                # 固定卷积层的参数，只更新全连接层的参数
                with torch.no_grad():
                    accuracy = 0
                    total = 0
                    for data in data_loader_image["val"]:
                        use_model.eval()
                        image, label = data
                        if use_cuda:
                            image, label = Variable(image.to(device)), Variable(label.to(device))
                        else:
                            image, label = Variable(image), Variable(label)

                        label_prediction = use_model(image)
                        _, prediction = torch.max(label_prediction.data, 1)
                        total += label.size(0)
                        accuracy += torch.sum(prediction=label.data)  ##这里是不是应该是label_prediction

                    # 输出测试准确率
                    print("test accuracy:{:.4f}".format(100 * accuracy / total))
                    acc = 100 * accuracy / total

                    # write to tensorboard
                    # writer.add_scalar('accuracy/test', acc, epoch)

                    # write test result to file
                    print('saving model...')
                    torch.save(use_model.state_dict(), '%s/net_%3d.pth' % (args.outf, epoch + 1))
                    acc_f.write("epoch = %03d,accuracy = %.3f%%" % (epoch + 1, acc))
                    acc_f.write('\n')
                    acc_f.flush()

                    # 记录最佳的测试准确率
                    if acc > best_test_acc:
                        print('saving best model...')
                        # 存储状态
                        state = {'state_dict': use_model.state_dict(),
                                 'acc': acc,
                                 'epoch': epoch + 1, }

                        # if don't have checkpoint file ,then creat it
                        if not os.path.isdir('checkpoint'):
                            os.mkdir('checkpoint')

                            torch.save(state, './checkpoint/ckpt.t7')
                            best_test_acc = acc
                            # write to tensorboard
                            # writer.add_scalar('best_accuracy/test', best_test_acc, epoch)

    end_time = time.time() - start_time
    print("training time is:{:.0f}m {:.0f}s".format(end_time // 60, end_time % 60))
    # writer.close()