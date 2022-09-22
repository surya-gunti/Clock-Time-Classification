import torch
import torchvision
from torchvision import transforms as T
from scripts import config


transforms = T.Compose([ T.ToTensor()])
test_ds = torchvision.datasets.ImageFolder(config.test_path, transform=transforms)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size = config.batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(config.MODEL_PATH)
model.to(device)

test_correct, test_total = 0, 0
with torch.no_grad():
    for img, lab in test_dl:
        img = img.to(device)
        lab = lab.to(device)

        out = model(img)
        pred = torch.argmax(out, dim=1)

        test_correct += (pred == lab).sum()
        test_total += lab.shape[0]
    test_acc = (test_correct/test_total).detach().cpu()
    print("Test Accuracy: {}".format(test_acc))
