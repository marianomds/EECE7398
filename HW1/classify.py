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
EPOCHS = 10

# Define the NN architecture
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        # 3 Hidden layers
        self.hidden1 = nn.Linear(32 * 32 * 3, 500)
        self.hidden2 = nn.Linear(500, 100)
        self.hidden3 = nn.Linear(100, 20)
        self.out = nn.Linear(20, 10)

    def forward(self, x):
        # Image size: 32x32, 3 channels
        x = x.view(-1, 32 * 32 * 3)
        # Use ReLU activation functions
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        outp = self.out(x)
        return outp

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
    model = NN()
    # Use SGD optimizer with momentum and L2 Regularization
    # I tried other optimizers, like Adam, but they were slower in my cpu and didn't provide better results
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.8, lr=0.01, weight_decay=0.0001)
    # Use the Cross Entropy Loss criterion for training: negative log-likelihood with softmax classifier
    loss_func = torch.nn.CrossEntropyLoss()

    print('\nEpoch | Batch | Train loss | Train accuracy |  Test loss  | Test accuracy')

    for epoch in range(EPOCHS):
        # Use mini-batch learning
        for batch, (input, target) in enumerate(train_loader):
            # Set NN in training mode
            model.train()
            # Forward pass for this batch's inputs
            output = model(input)
            loss = loss_func(output, target)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Weights update
            optimizer.step()

            # Evaluate training results every 100 batches and also after the last batch
            if batch % 100 == 0 or batch == len(train_loader)-1:
                # Set NN in evaluation/inference mode
                model.eval()
                # Test set evaluation
                test_loss = 0
                test_correct = 0
                for data, target in test_loader:
                    # Forward pass
                    output = model(data)
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
                    output = model(data)
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
    result = model(img_r_t)
    result = result.data[0].tolist()
    # Select the best class from the results, and print its corresponding label
    result_ind = result.index(max(result))
    result_lab = label_names['label_names'][result_ind]
    print('Predicted result: {}'.format(result_lab))


if __name__ == "__main__":
    task = str(sys.argv[1])
    if task == 'train':
        train()
    elif task == 'test':
        test(str(sys.argv[2]))