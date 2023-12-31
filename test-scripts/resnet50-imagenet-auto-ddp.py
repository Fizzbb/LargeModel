
import torch
import torchvision
import torchvision.transforms as transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("cuda device count: {}".format(torch.cuda.device_count()))
# Set hyperparameters
num_epochs = 2
batch_size = 64
learning_rate = 0.001

# Initialize transformations for data augmentation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(
    root='/home/aac/data/imagenet-kaggle/ILSVRC/Data/CLS-LOC/train', 
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4)

# Load the ResNet50 model
model = torchvision.models.resnet50(pretrained=True)

# Parallelize training across multiple GPUs
model = torch.nn.DataParallel(model)

# Set the model to run on the device
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
import time
# Train the model...
for epoch in range(num_epochs):
    step = 0
    print("start epoch {}".format(epoch))
    t0 = time.time()
    tt0 = t0
    for inputs, labels in train_loader:
        step = step + 1
        # Move input and label tensors to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero out the optimizer
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        if step%1000 == 0:
            print("step : {}, 1000 step time: {}".format(step, time.time()-tt0 ))
            tt0 = time.time()
    # Print the loss for every epoch
    t1 = (time.time() - t0)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, time: {t1:.2f} seconds')

print(f'Finished Training, Loss: {loss.item():.4f}')

