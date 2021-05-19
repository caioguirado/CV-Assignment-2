import argparse

if __name__ == "__main__":
    
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('-f',
                           '--folder',
                           action='store',
                           help='Folder name')
    
    args = my_parser.parse_args()
    vars_ = vars(args)
    exp_folder = vars_['folder']