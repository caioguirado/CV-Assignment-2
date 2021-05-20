import torch
import argparse
import pandas as pd

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

    SEED = variables['SEED']
    RESHAPE_SIZE = variables['RESHAPE_SIZE']
    PIPELINE_TRAIN = variables['PIPELINE_TRAIN']
    PIPELINE_VAL = variables['PIPELINE_VAL']
    DATASET = variables['DATASET']
    BATCH_SIZE = variables['BATCH_SIZE']
    NUM_WORKERS = variables['NUM_WORKERS']
    CRITERION = variables['CRITERION']
    MODEL = variables['MODEL']
    OPTIMIZER = variables['OPTIMIZER']
    MAX_EPOCHS = variables['MAX_EPOCHS']
    DATASET = variables['DATASET']

    # Load data
    df = pd.read_csv('./data/fer2013.csv')

    # Split data
    df_train = df[df['Usage'] == 'Training']
    df_test = df[df['Usage'] != 'Training']

    # Load CUDA 
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # Create Dataloaders
    params_train = {'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS}
    training_set = DATASET(df_train,  transform=PIPELINE_TRAIN)
    training_generator = torch.utils.data.DataLoader(training_set, **params_train)

    validation_set = DATASET(partition['validation'],  transform=PIPELINE_VAL)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params_val)

    # Train

    # Save results to experiment folder