# CV-Assignment-2

```
root/
  ├── .gitignore
  ├── utils.py
  ├── run_experiment.py
  ├── README.md
  ├── models/
  ├── data/
  ├── dataset/
  ├── preprocessing/
  └── experiments/
          └── experiment1/
              └── config.py
```

In order to create a new experiment, clone one of the experiments folder and edit the `config.py` file with your information. You must have a GPU for it to work otherwise it won't run.
```zsh
    python3 run_experiment.py -f <exp_folder_name>
```
