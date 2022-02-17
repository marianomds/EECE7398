import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable
import sys
from os import path
from PIL import Image
import pickle

# Define batch size and number of epochs
BATCH_SIZE = 128
EPOCHS = 40

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Image size: 32x32, 3 channels
        self.conv1 = nn.Sequential(         # input shape (3, 32, 32)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=32,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   
                padding=2,                   
            ),                              # output shape (32, 32, 32)
            nn.ReLU(),                      # activation
            nn.Dropout(0.2),                # 20 % Dropout probability
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (32, 16, 16)
        )
        self.conv2 = nn.Sequential(         # input shape (32, 16, 16)
            nn.Conv2d(32, 64, 3, 1, 1),     # output shape (64, 16, 16)
            nn.ReLU(),                      # activation
            nn.Dropout(0.2),                # 20 % Dropout probability
            nn.MaxPool2d(2),                # output shape (64, 8, 8)
        )
        self.conv3 = nn.Sequential(         # input shape (64, 8, 8)
            nn.Conv2d(64, 128, 3, 1, 1),    # output shape (128, 8, 8)
            nn.ReLU(),                      # activation
            nn.Dropout(0.2),                # 20 % Dropout probability
            nn.MaxPool2d(2),                # output shape (128, 4, 4)
        )
        self.fc1 = nn.Linear(128 * 4 * 4, 1000) # fully connected layer 1
        self.fc2 = nn.Linear(1000, 100)         # fully connected layer 2
        self.fc3 = nn.Linear(100, 50)           # fully connected layer 2
        self.out = nn.Linear(50, 10)            # fully connected layer 3, output 10 classes
        self.dropout = nn.Dropout(p=0.5)        # 50 % Dropout probability (for the FC layers)

    def forward(self, x):
        # Image size: 32x32, 3 channels
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)           
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        outp = self.out(x)
        return outp, x1    # return x1 for visualization of first CONV layer

# Define the training function
def train():

    # Load the training data set
    train_data = torchvision.datasets.CIFAR10(
        root='./data.cifar10',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False)

    # Mini-batch configuration
    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True)

    # Load the test data set
    test_data = torchvision.datasets.CIFAR10(
        root='./data.cifar10/',
        train=False,
        transform=torchvision.transforms.ToTensor())

    # Mini-batch configuration
    test_loader = Data.DataLoader(dataset=test_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

    # Instantiate the Neural Network
    model = CNN()
    # Use Adam optimizer with L2 Regularization
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0001)
    # Use the Cross Entropy Loss criterion for training: negative log-likelihood with softmax classifier
    loss_func = torch.nn.CrossEntropyLoss()

    print('\nEpoch | Batch | Train loss | Train accuracy |  Test loss  | Test accuracy')

    for epoch in range(EPOCHS):
        # Use mini-batch learning
        for batch, (input, target) in enumerate(train_loader):
            # Set CNN in training mode
            model.train()
            # Forward pass for this batch's inputs
            output, x1 = model(input)
            loss = loss_func(output, target)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Weights update
            optimizer.step()

            # Evaluate training results every 100 batches and also after the last batch
            if batch % 100 == 0 or batch == len(train_loader)-1:
                # Set CNN in evaluation/inference mode
                model.eval()
                # Test set evaluation
                test_loss = 0
                test_correct = 0
                for data, target in test_loader:
                    # Forward pass
                    output, x1 = model(data)
                    # Also use the Cross Entropy Loss criterion for the test loss
                    criterion = nn.CrossEntropyLoss()
                    test_loss = criterion(output, target)
                    # Get the index of the max log-probability
                    pred = output.data.max(1, keepdim=True)[1] 
                    test_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                # Compute the total test loss and accuracy
                test_loss /= len(test_loader.dataset)
                test_acc = float(100. * test_correct)  / float(len(test_loader.dataset ))

                # Train set evaluation
                train_loss = 0
                train_correct = 0
                for data, target in train_loader:
                    # Forward pass
                    output, x1 = model(data)
                    criterion = nn.CrossEntropyLoss()
                    train_loss = criterion(output, target)
                    pred = output.data.max(1, keepdim=True)[1]
                    train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                # Compute the total train loss and accuracy
                train_loss /= len(train_loader.dataset)
                train_acc = float(100. * train_correct) / float(len(train_loader.dataset))

                # Print this batch's results
                print(
                    '  {}      {}\t{:.8f}      ({:.3f}%)\t{:.8f}     ({:.3f}%)'
                    .format(epoch, batch, train_loss, train_acc,
                                          test_loss , test_acc))

        # At the end of each epoch, if the results are good enough, save the model
        if (test_acc > 74):
            # Save the model, with name: model_test_accuracy
            # Folder 'model' must exist already
            model_file = ('model_' + str(round(test_acc)))
            print('\nSaving {:.3f}% (Test accuracy) model in file {} ...'.format(test_acc, model_file))
            torch.save(model, './model/' + model_file)


# Define the testing function
def test(img):

    # Find and load the best model saved
    model_n = 99
    while path.exists('./model/model_' + str(model_n)) == False:
        model_n = model_n - 1
    print('\nLoading best model: {}% accuracy'.format(model_n))
    model = torch.load('./model/model_' + str(model_n))

    # Load image
    img = Image.open(img)
    img = img.convert('RGB')
    # Resize it to the CIFAR size
    img_r = img.resize((32, 32))
    # Convert it to a tensor
    trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])
    img_r_t = trans(img_r)

    # Load the labels from the CIFAR database
    f = open('./data.cifar10/cifar-10-batches-py/batches.meta', 'rb')
    label_names = pickle.load(f)

    # Perform inference on the input image
    model.eval()
    result, x1 = model(img_r_t)
    result = result.data[0].tolist()
    # Select the best class from the results, and print its corresponding label
    result_ind = result.index(max(result))
    result_lab = label_names['label_names'][result_ind]
    print('Predicted result: {}'.format(result_lab))

    # Generate the CONV layer visualization
    plt.figure(figsize = (6, 6))
    for i in range(32):
        ax = plt.subplot(6, 6, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        img_conv = torchvision.transforms.ToPILImage()(x1[0][i])
        plt.imshow(img_conv, cmap='gray')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('CONV_rslt.png')
    plt.show()


if __name__ == "__main__":
    task = str(sys.argv[1])
    if task == 'train':
        train()
    elif task == 'test':
        test(str(sys.argv[2]))