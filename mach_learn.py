# Module for Machine Learning Methods

## Package Import 
### Python Packages 
import numpy as np
import os 
import pickle
import random 
import re
import sys

import matplotlib.pyplot as plt
import sklearn
import skorch
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost 

from datetime import datetime

from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, EarlyStopping, Checkpoint
from skorch.dataset import Dataset
from skorch.helper import predefined_split

from sklearn.metrics import accuracy_score, log_loss, f1_score

from xgboost import XGBClassifier

### My Modules
from file_path import file_path

## Convolutional Neural Network function
def conv_nn(name, train, test, val, random_seed, colour_change): 
    
     # Make folder for CNN results
    try:  
        folder_path = file_path+f'Results/{name}/CNN/'
        os.mkdir(folder_path)  
    except OSError as error:
        pass
    
    # Make folder path based on colour 
    if colour_change in ['gray','grey','greyscale','grayscale']:
        try:  
            folder_path = folder_path+'Grayscale/'
            os.mkdir(folder_path)  
        except OSError as error:
            pass
    elif colour_change == "rgb":
        try:  
            folder_path = folder_path+'RGB/'
            os.mkdir(folder_path)  
        except OSError as error:
            pass
    
    # Set random seeds
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Create seed_worker for NeuralNetClassifier
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # Set generator
    g = torch.Generator()
    g.manual_seed(random_seed) 
    
    # Settings based on device used
    if torch.cuda.is_available:
        dev = 'cuda'
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        dev = 'cpu'
    
    # Set initial values
    NUM_EPOCHS = 500
    BATCH_SIZE = 128
    lr = 0.001
    
    # Extract features and labels from output
    train_features = train[0].to(torch.float32)
    train_labels = train[1].flatten()
    test_features = test[0].to(torch.float32)
    test_labels = test[1].flatten()
    val_features = val[0].to(torch.float32)
    val_labels = val[1].flatten()
      
    # Get number of classes
    n_classes = len(np.unique(train_labels))
    
    # Get number of channels 
    if len(train_features.size()) >= 4:
        n_channels = int(train_features.size()[-1])
    else:
        n_channels = 1
    
    # Permute all tensors to fit CNN
    if n_channels == 1:
        train_features = train_features.unsqueeze(n_channels)
        test_features = test_features.unsqueeze(n_channels)
        val_features = val_features.unsqueeze(n_channels)
    else:
        train_features = train_features.permute(0, 3, 1, 2)
        test_features = test_features.permute(0, 3, 1, 2)
        val_features = val_features.permute(0, 3, 1, 2)   
     
    # Extract resolution for CNN
    res = int(train_features.size()[-1])
        
    # Ensure accuracy metrics calculated correctly based on number of classes    
    if n_classes != 2: 
        f1_avg = 'f1_weighted'
    else:
        f1_avg = 'f1'       

    # Define CNN architecture
    class NeuralNet(nn.Module):
        def __init__(self, in_channels, num_classes, resolution):
            super(NeuralNet, self).__init__()

            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3),
                nn.BatchNorm2d(16),
                nn.ReLU())

            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

            self.layer3 = nn.Sequential(
                nn.Conv2d(16, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU())

            self.layer4 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU())
            
            
            if resolution == 28:
                self.layer5 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding = 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2))
                
            else:
                self.layer5 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2))
            
            if resolution in [64, 128, 224]:
                self.layer6 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU())
           
                self.layer7 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU())
                
                if resolution == 64:
                    self.layer8 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding = 1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))

                elif resolution == 128:
                    self.layer8 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))

                    self.layer9 = nn.Sequential(
                            nn.Conv2d(64, 64, kernel_size=3),
                            nn.BatchNorm2d(64),
                            nn.ReLU())

                    self.layer10 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding = 1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))

                else:
                    self.layer8 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding = 1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))

                    self.layer9 = nn.Sequential(
                            nn.Conv2d(64, 64, kernel_size=3),
                            nn.BatchNorm2d(64),
                            nn.ReLU())

                    self.layer10 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))
                
                    self.layer11 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())

                    self.layer12 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding = 1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))
                                      
            self.fc = nn.Sequential(
                nn.Linear(64 * 4 * 4, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes))
          
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)        
            if hasattr(self, 'layer6'):
                x = self.layer6(x)
                x = self.layer7(x)
                x = self.layer8(x)
            if hasattr(self, 'layer9'):
                x = self.layer9(x)
                x = self.layer10(x)
            if hasattr(self, 'layer11'):
                x = self.layer11(x)
                x = self.layer12(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Define Neural Network with input params
    CNN = NeuralNet(in_channels=n_channels, num_classes=n_classes, resolution = res)
    
    # Define Loss Function   
    loss = nn.CrossEntropyLoss()
    
    # Define validation set
    validation = Dataset(val_features, val_labels)
    
    # Define callbacks 
    epoch_score = EpochScoring(scoring = 'accuracy', lower_is_better = False)
    earlystop = EarlyStopping(patience = 10)
    check = Checkpoint(f_params=folder_path+"params.pt", f_history = folder_path+'history.json')
    
    # Write to file 
    path3 = folder_path+f'cnn_{colour_change.lower()}_train.txt'
    with open(path3, mode='w') as file:
        sys.stdout = file 
        print("CONVOLUTIONAL NEURAL NETWORK STATISTICS")
        print('-' * 50)
        print("TRAINING THE MODEL")
        print('-' * 30)
        sys.stdout = sys.__stdout__
    
    # Define model
    model = NeuralNetClassifier(CNN,
                                criterion = loss,
                                optimizer = optim.Adam,
                                optimizer__lr = lr,
                                max_epochs = NUM_EPOCHS,
                                batch_size = BATCH_SIZE, 
                                callbacks = [epoch_score,earlystop,check],
                                train_split = predefined_split(validation),
                                device = dev,
                                iterator_train__num_workers = 0,
                                iterator_valid__num_workers = 0,
                                iterator_train__worker_init_fn=seed_worker,
                                iterator_valid__worker_init_fn=seed_worker,
                                iterator_train__generator = g, 
                                iterator_valid__generator = g)
    
    with open(path3, mode='a+') as file:
        sys.stdout = file 
        start = datetime.now()
        model.fit(train_features,train_labels)
        end = datetime.now()
        time = (end - start).total_seconds()
        print("Model Training Time:", time, "seconds")
        print("Device used:", model.device)
        sys.stdout = sys.__stdout__
        
    ## Complete Predictions 
    pred = model.predict(test_features)
    pred_prob = model.predict_proba(test_features)
    
    ## Ensure accuracy metrics are correct
    if n_classes != 2: 
        f1_avg = 'weighted'
    else:
        f1_avg = 'binary'
    
    ## Calculate Accuracy Metrics
    acc_score = accuracy_score(test_labels, pred)
    f1score = f1_score(test_labels, pred, average=f1_avg)
    logloss = log_loss(test_labels, pred_prob)
    prediction_score = (f1score+acc_score)/(2*logloss)

    ## Write to log file 
    with open(path3, mode='a+') as file:
        sys.stdout = file 
        print('-' * 50)
        print("TESTING THE MODEL")
        print('-' * 30)
        print("Scores based on test set:")
        print("Accuracy Score:",acc_score)
        print("F1 Score:",f1score)
        print("Log Loss:", logloss)
        print("Prediction Score:", prediction_score)
        print('-' * 50)
        print()
        sys.stdout = sys.__stdout__

    # Remove ANSI values
    with open(path3, mode="r") as file:
        content = file.read()
        
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    # Perform regular expression substitution to remove ANSI escape sequences
    content = ansi_escape.sub('', content)
    
    with open(path3, mode="w") as file:
        sys.stdout = file
        print(content)
        sys.stdout = sys.__stdout__     
    
## XGBoost Function         
def xg_boost(name,train,test,val,stage,random_seed,colour_change):
    
    # Make folder for XGBoost results
    try:  
        folder_path = file_path+f'Results/{name}/XGBoost/'
        os.mkdir(folder_path)  
    except OSError as error:
        pass
    
    # Make folder path based on colour 
    if colour_change in ['gray','grey','greyscale','grayscale']:
        try:  
            folder_path = folder_path+'Grayscale/'
            os.mkdir(folder_path)  
        except OSError as error:
            pass
    elif colour_change == "rgb":
        try:  
            folder_path = folder_path+'RGB/'
            os.mkdir(folder_path)  
        except OSError as error:
            pass
        
    # Set random seeds
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Create seed_worker for NeuralNetClassifier
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # Set generator
    g = torch.Generator()
    g.manual_seed(random_seed) 
    
    # Extract features and labels from output
    train_features = train[0]
    train_labels = train[1]
    test_features = test[0]
    test_labels = test[1]
    val_features = val[0]
    val_labels = val[1]
    
    # Get number of classes
    n_classes = len(np.unique(train_labels))    
    
    # Set initial values
    iter_num = 500
    lr = 0.1
    
    # Remove spatial structure from features to work with sklearn
    train_features = train_features.view(train_features.size(0), -1)
    val_features = val_features.view(val_features.size(0), -1)
    test_features = test_features.view(test_features.size(0), -1)
            
    # Ensure correct accuracy metrics calculated
    if n_classes != 2: 
        f1_avg = 'f1_weighted'
        obj = "multi:softprob"
        metric = "mlogloss"
    else:
        f1_avg = 'f1'
        obj = "binary:logistic"
        metric = "logloss"
                
    # Start writing to file
    path = folder_path+f'xgboost_results_{colour_change.lower()}.txt'
    with open(path, mode='w') as file:
        sys.stdout = file   
        print("XGBOOST STATISTICS")
        print('-' * 50)
        print("TRAINING THE MODEL")
        print('-' * 30)
        print('  iteration    train_loss    valid_loss    cp')
        print('-----------  ------------  ------------  ----')       
        sys.stdout = sys.__stdout__    

    # Define XGBoost model
    model = XGBClassifier(objective = obj,
                          n_estimators = iter_num,
                          device = 'cpu',
                          learning_rate = lr,
                          eval_metric = metric,
                          random_state = random_seed,
                          early_stopping_rounds = 10)
    
    ## Fit Model
    start = datetime.now()
    model.fit(X = train_features, y = train_labels, eval_set = [(train_features,train_labels), (val_features, val_labels)], verbose = False)
    end = datetime.now()
    run_time = (end-start).total_seconds()
    
    # Save the best model
    with open('best_xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Load the best model for prediction
    with open('best_xgboost_model.pkl', 'rb') as f:
        best_model = pickle.load(f)

    ## Setting up Results file similar to CNN
    results = model.evals_result()
    max_iter = len(results['validation_0'][metric])
    step = 10**(len(str(abs(max_iter))) - 2)
    indexes = [0] + [i for i in range(step-1,max_iter, step)] 
    indexes = list(set(indexes))
    last_six = [i for i in range(max_iter-11,max_iter) if i not in indexes]
    indexes += last_six
    indexes = sorted(indexes)

    # Print results to the file
    for i in indexes:
        
        # Keeping track of min valid log loss
        current_value = results['validation_1'][metric][i]
        previous_values = results['validation_1'][metric][:i]
        if previous_values:
            is_minimum = all(current_value <= prev_value for prev_value in previous_values)
        else:
            is_minimum = True

        # Write to log file     
        with open(path, mode='a+') as file:
            sys.stdout = file   
            print("{:11}  {:12.5f}  {:12.5f}  {:4}".format(i+1, 
                                                results['validation_0'][metric][i], 
                                                results['validation_1'][metric][i],
                                                "   +" if is_minimum else ""))
            sys.stdout = sys.__stdout__  
            
    ## Prediction
    pred = best_model.predict(val_features)
    pred_prob = best_model.predict_proba(val_features)         
    
    ## Ensure correct accuracy metrics calculated
    if n_classes != 2:  
        f1_avg = 'weighted'
    else:
        f1_avg = 'binary'
        
    ## Accuracy Metrics
    acc_score = accuracy_score(val_labels, pred)
    f1score = f1_score(val_labels, pred, average=f1_avg)
    logloss = log_loss(val_labels, pred_prob)
    prediction_score = ((f1score+acc_score)/(2*logloss))

    ## Write to log file 
    with open(path, mode='a+') as file:
        # Redirect sys.stdout to the file
        sys.stdout = file   
        if int(model.best_iteration)+1 < iter_num:
            print("Stopping at iteration", int(model.best_iteration)+1, "since valid_loss has not improved in the last 10 iterations.")
        else:
            print("Stopping at iteration", int(iter_num), "since this is equal to the maximum number of estimators paramater provided to the model.")
        print("Model Training Time", run_time, "seconds")
        print("Device used:", model.device)
        print('-' * 50)
        print("TESTING THE MODEL")
        print('-' * 30)
        print("Scores based on test set:")
        print("Accuracy Score:",acc_score)
        print("F1 Score:",f1score)
        print("Log Loss:", logloss)
        print("Prediction Score:", prediction_score)
        print('-' * 50)
        print()
        sys.stdout = sys.__stdout__