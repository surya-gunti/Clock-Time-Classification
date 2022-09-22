import torch
import torchvision
from torchvision import transforms as T
from scripts import config
import numpy as np
from scripts.util import plot_accuracy



transforms = T.Compose([ T.ToTensor()])

train_ds = torchvision.datasets.ImageFolder(config.train_path, transform=transforms)
val_ds = torchvision.datasets.ImageFolder(config.val_path, transform=transforms)
test_ds = torchvision.datasets.ImageFolder(config.test_path, transform=transforms)




train_dl = torch.utils.data.DataLoader(train_ds, batch_size = config.batch_size, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size = config.batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size = config.batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('pytorch/vision:v0.10.0', config.model, weights="ResNet18_Weights.IMAGENET1K_V1")
model.fc = torch.nn.Linear(in_features=512, out_features=config.num_classes, bias=True)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

loss_fn = torch.nn.CrossEntropyLoss()


train_correct, train_total = 0, 0
val_correct, val_total = 0, 0

train_acc_li, val_acc_li = [], []

best_score = 0

for e in range(config.epochs):
    train_losses, val_losses = [], []
    for img, lab in train_dl:
        img = img.to(device)
        lab = lab.to(device)

        optimizer.zero_grad()
        out = model(img)

        pred = torch.argmax(out, dim=1)

        train_correct += (pred == lab).sum()
        train_total += lab.shape[0]
        
        train_loss = loss_fn(out, lab)
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
    train_acc = (train_correct / train_total).detach().cpu()
    train_acc_li.append(train_acc)

    with torch.no_grad():
        for img, lab in val_dl:
            img = img.to(device)
            lab = lab.to(device)

            optimizer.zero_grad()
            out = model(img)
            pred = torch.argmax(out, dim=1)

            val_correct += (pred == lab).sum()
            val_total += lab.shape[0]
            val_loss = loss_fn(out, lab)
            val_losses.append(val_loss.item())
        val_acc = (val_correct/val_total).detach().cpu()
        val_acc_li.append(val_acc)
        if(val_acc>best_score):
            best_score = val_acc
            torch.save(model, config.MODEL_PATH)

    print('Epoch: {}, Train Loss: {}, Train Acc: {}, Val Loss:{}, Val Acc:{}'.format(e, np.mean(train_losses), train_correct/train_total, np.mean(val_losses), val_correct/val_total))

plot_accuracy(train_acc_li, val_acc_li)        
    