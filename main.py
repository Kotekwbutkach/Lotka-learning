from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import CustomDataset

N = 500
k = 100


class LVPredictorNetwork(nn.Module):
    def __init__(self):
        super(LVPredictorNetwork, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2 * N, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2 * k),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x.to('cuda'))
        return logits


training_data = CustomDataset("dane_treningowe.csv", N, k)
test_data = CustomDataset("dane_treningowe_100.csv", N, k)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

model = LVPredictorNetwork()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train_one_epoch(epoch_index, tb_writer, batch_num=1000):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting

    for i, data in enumerate(train_dataloader):
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % batch_num == batch_num - 1:
            last_loss = running_loss / batch_num  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_data) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_validation_loss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_validation_loss = 0.0
    for i, validation_data in enumerate(test_dataloader):
        validation_inputs, validation_labels = validation_data
        validation_outputs = model(validation_inputs)
        validation_loss = loss_fn(validation_outputs, validation_labels)
        running_validation_loss += validation_loss

    avg_validation_loss = running_validation_loss / len(test_dataloader)

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training': avg_loss, 'Validation': avg_validation_loss},
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_validation_loss < best_validation_loss:
        best_validation_loss = avg_validation_loss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
