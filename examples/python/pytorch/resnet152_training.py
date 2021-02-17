from resnet_torch import resnet152
import torch,torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import time

device = "cuda:0"
batch_size = 4
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
            torchvision.transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

train_dataset = torchvision.datasets.CIFAR10('./datasets/',transform=transform,download=True)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=2)


model = models.resnet152()
torch.cuda.set_device(device)
model.cuda()
print("Setting optimizer")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print("Starting training")
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    print("Epoch %d"%epoch)
    for i, data in enumerate(trainloader, 0):
        start = time.time()
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        end = time.time()

        # print statistics
        print("[Epoch %d] Batch: %d Loss: %.3f Time per Image: %.5f"%(epoch,i,loss.item(),(end - start)/batch_size))

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')