from Xception_net import *

xModel = XceptionNetModule()
xModel.cuda()
print(summary(xModel, (1, 128, 128)))

"""# Define Optimizer and loss"""

objectiveFun = nn.MSELoss()
optimiser = optim.Adam(xModel.parameters(), lr=0.008)

"""# Define Validation"""

@torch.no_grad()
def validate_train(saveFlag=None):
  cumulative_loss = 0.0
  xModel.eval()

  for f, labels in tqdm(val_data, desc='Validating', ncols=600):
    f = f.cuda()
    labels = labels.cuda()
    out = xModel(f)
    loss = objectiveFun(out, labels)
    cumulative_loss += loss.item()
    break
  visualize_batch(f[:16].cpu(), out[:16].cpu(), shape = (4, 4), size = 16, title = 'Validation sample predictions', save = saveFlag)
  return cumulative_loss/len(val_data)


# Train
epochs = 30
batches = len(train_data)
op_loss = np.inf
optimiser.zero_grad()

loss_vals = []

for epoch in range(epochs):
  cumulative_loss = 0.0
  epoch_loss= []

  xModel.train()
  for batch_ind, (f, labels) in enumerate(tqdm(train_data, desc=f'Epoch({epoch+1}/{epochs})', ncols=800)):
    f = f.cuda()
    labels = labels.cuda()

    out = xModel(f)
    loss = objectiveFun(out, labels)
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()
    cumulative_loss += loss.item()
    epoch_loss.append(loss.item())

  loss_vals.append(sum(epoch_loss)/len(epoch_loss))

  validation_loss = validate(os.path.join('progress', f'epoch({str(epoch+1).zfill(len(str(epochs)))})'.jpg))
  if validation_loss < op_loss:
    op_loss = validation_loss
    print('Saving the very nice model')
    torch.save(model.state_dict(), 'face_landmark_model.pt')
  print(f'Epoch({epoch + 1}/{epochs}) -> Training Loss: {cumulative_loss/batches:.8f} | Validation Loss: {validation_loss:.8f}')

epochs = 30
batches = len(train_data)
best_loss = np.inf
optimiser.zero_grad()

for epoch in range(epochs):
    cum_loss = 0.0

    xModel.train()
    for batch_index, (features, labels) in enumerate(tqdm(train_data, desc = f'Epoch({epoch + 1}/{epochs})', ncols = 800)):
        features = features.cuda()
        labels = labels.cuda()

        outputs = xModel(features)
        
        loss = objectiveFun(outputs, labels)

        loss.backward()

        optimiser.step()
        
        optimiser.zero_grad()

        cum_loss += loss.item()

    val_loss = validate(os.path.join('progress', f'epoch({str(epoch + 1).zfill(len(str(epochs)))}).jpg'))

    if val_loss < best_loss:
        best_loss = val_loss
        print('Saving model....................')
        torch.save(xModel.state_dict(), 'face_landmark_model.pt')

    print(f'Epoch({epoch + 1}/{epochs}) -> Training Loss: {cum_loss/batches:.8f} | Validation Loss: {val_loss:.8f}')
