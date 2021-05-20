import time
import torch
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

if __name__ == "__main__":
    
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('-f',
                           '--folder',
                           action='store',
                           help='Folder name')
    
    args = my_parser.parse_args()
    vars_ = vars(args)
    exp_folder = vars_['folder']

    exp_path = f'./experiments/{exp_folder}'

    # Load parameters from experiment config
    variables = {}
    exec(open(exp_path + '/config.py').read(), variables)

    print('============ Reading Experiment Config ============')

    SEED = variables['SEED']
    PIPELINE_TRAIN = variables['PIPELINE_TRAIN']
    PIPELINE_VAL = variables['PIPELINE_VAL']
    DATASET = variables['DATASET']
    BATCH_SIZE = variables['BATCH_SIZE']
    NUM_WORKERS = variables['NUM_WORKERS']
    CRITERION = variables['CRITERION']
    MODEL = variables['MODEL']
    OPTIMIZER = variables['OPTIMIZER']
    MAX_EPOCHS = variables['MAX_EPOCHS']

    print('============ Loading Data ============')
    # Load data
    df = pd.read_csv('./data/fer2013.csv')

    # Split data
    df_train = df[df['Usage'] == 'Training']
    df_test = df[df['Usage'] != 'PublicTest']

    # Load CUDA 
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # Create Dataloaders
    print('============ Creating Dataloaders ============')
    params_train = {'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS}
    params_val = {'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS}

    training_set = DATASET(df_train,  transform=PIPELINE_TRAIN)
    training_generator = torch.utils.data.DataLoader(training_set, **params_train)

    validation_set = DATASET(df_test,  transform=PIPELINE_VAL)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params_val)

    print('============ Model to CUDA ============')
    MODEL.cuda()
    optimizer = optim.Adam(MODEL.parameters(), lr=0.0001)

    # Train
    train_history = []
    val_history = []
    loss_stats = {
        'train': [],
        'val': []
    }

    print('============ Init Train ============')
    for epoch in range(MAX_EPOCHS):
        i = 0
        start_time = time.time()
        running_loss = 0.0
        train_epoch_loss = 0
        MODEL.train()
        for idx, (local_batch, local_labels) in enumerate(tqdm(training_generator)):

            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            # local_labels = local_labels.unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = MODEL(local_batch)
            loss = CRITERION(outputs, local_labels)

            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()
        
        # Validation
        with torch.set_grad_enabled(False):
            MODEL.eval()
            val_epoch_loss = 0
            val_preds = []
            val_labels = []
            # sigmoid = nn.Sigmoid()
            for local_batch, local_labels in tqdm(validation_generator):

                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                # local_labels = local_labels.unsqueeze(1).float()

                outputs = MODEL(local_batch)
                val_loss = CRITERION(outputs, local_labels)
                val_epoch_loss += val_loss.item()
                
                val_preds += torch.argmax(nn.functional.softmax(outputs), axis=1).tolist()
                # val_preds += torch.flatten(outputs).tolist()
                val_labels += torch.flatten(local_labels).tolist()

        loss_stats['train'].append(train_epoch_loss/len(training_generator))
        loss_stats['val'].append(val_epoch_loss/len(validation_generator))                              
        # val_preds = torch.round(sigmoid(torch.Tensor(val_preds))).tolist()

        print(f'Epoch Time: {time.time() - start_time} sec', f'Current timestamp: {time.time()}')
        print(f'Epoch {epoch+0:03}: | Train Loss: {train_epoch_loss/len(training_generator)} | Val Loss: {val_epoch_loss/len(validation_generator)}')
        
        # if f1_score(val_labels, val_preds) > 0.989:
        #     torch.save(vgg.state_dict(), f'good_performance_epoch_{epoch}.pth')

        # Save results to experiment folder
        if epoch % 10 == 0:
            torch.save(MODEL.state_dict(), f'./experiments/{exp_folder}/model_epoch_{epoch}.pth')

# Create report and save
plt.figure()
plt.plot(loss_stats['train'], label='Train')
plt.plot(loss_stats['val'], label='Val')
plt.title('Train and Validation Loss')
plt.legend()
plt.savefig(f'./experiments/{exp_folder}/train_val_loss.png')

# Saving predictions
with open(f'./experiments/{exp_folder}/val_preds.pkl', 'wb') as f:
    pickle.dump(val_preds, f)

with open(f'./experiments/{exp_folder}/val_labels.pkl', 'wb') as f:
    pickle.dump(val_labels, f)

# Save classification report
cr = classification_report(val_labels, val_preds, output_dict=True)
df_cr = pd.DataFrame(cr).transpose()
df_cr.to_csv(f'./experiments/{exp_folder}/classification_report.csv')

with open(f'./experiments/{exp_folder}/classification_report.tex', 'w') as f:
    f.write(df_cr.to_latex())

with open(f'./experiments/{exp_folder}/classification_report.txt', 'w') as f:
    f.write(df_cr.to_latex())