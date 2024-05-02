# Module for Statistical Learning Methods

## Package Import
### Python Packages 
import numpy as np
import pandas as pd
import torch

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.feature_selection import mutual_info_classif as MI, VarianceThreshold as VarThres
from sklearn.metrics import accuracy_score, log_loss, f1_score

import torch.utils.data as data
import matplotlib.pyplot as plt

import skorch 

### Python Modules
from datetime import datetime
from functools import partial
from itertools import product
import math 
import os
import random
import re
import sys

### My Modules
from file_path import file_path

## Function to select the top k% of features
def select_best_features_pct(data, scores, percentage):
    num_feat = math.floor(data.shape[1]*(percentage/100))
    indices = np.argsort(scores)
    indices = indices[-num_feat:]
    data_new = data[:, indices]
    return data_new
 
## LDA Function
def lda(name,train,test,val,stage,random_seed, parameters,colour_change): 
       
    # Extract features and labels from output
    train_features = train[0]
    train_labels = train[1]
    test_features = test[0]
    test_labels = test[1]
    val_features = val[0]
    val_labels = val[1]
       
    # Make folder for LDA results
    try:  
        folder_path = file_path+f'Results/{name}/LDA/'
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
    
    # Extract number of classes
    n_classes = len(np.unique(train_labels))
    
    # Remove spatial structure from features to work with sklearn
    train_features = train_features.view(train_features.size(0), -1)
    val_features = val_features.view(val_features.size(0), -1)
    test_features = test_features.view(test_features.size(0), -1)
    
    # If the model is in the training stage: 
    if stage == "TRAIN":
        
        #Start writing the figures to the file
        path = folder_path+f'lda_results_{colour_change.lower()}.txt'
        with open(path, mode='w') as file:
                sys.stdout = file   
                print("LDA STATISTICS")
                sys.stdout = sys.__stdout__
        
        # Start dictionary of models for later comparison
        models = {}
        
        # Load or Generate MI Scores for training dataset
        score_path = file_path+f"Results/{name}/mutual_information_{colour_change}.npy"
        if os.path.exists(score_path):
            MI_Scores = np.load(score_path)
            print("Mutual Information Scores loaded")
        
        else: 
            MI_Scores = MI(train_features, train_labels, discrete_features = False, random_state = random_seed)
            np.save(score_path, MI_Scores)
        
        # FULL MODEL TEST
        ## Specify Full Model and Flags
        full_model = LDA()
        fs_flag = "N"
        fs_method = ""
        feat_num = train_features.size()[1]
        feat_pct = 1
        
        ## Fit Model
        start = datetime.now()
        full_model.fit(train_features, train_labels).transform(train_features)

        ## Hyperparameter tuning
        pred = full_model.predict(val_features)
        pred_prob = full_model.predict_proba(val_features)

        ## End model fitting time
        end = datetime.now()
        run_time = (end-start).total_seconds()

        ## Ensure correct parameters specified based on number of classes
        if n_classes != 2: 
            f1_avg = 'weighted'
        else:
            f1_avg = 'binary'

        ## Accuracy Metrics
        acc_score = accuracy_score(val_labels, pred)
        f1score = f1_score(val_labels, pred, average=f1_avg)
        logloss = log_loss(val_labels, pred_prob)
        prediction_score = ((f1score+acc_score)/(2*logloss))

        ## Enter data into the dictionary
        entry = {'full_model' : [None, run_time, colour_change.title(), fs_flag, fs_method, feat_num, feat_pct, acc_score, f1score, logloss, prediction_score]}
        models.update(entry)
        
        ## Write to log file 
        with open(path, mode='a+') as file:
            # Redirect sys.stdout to the file
            sys.stdout = file   
            print('-' * 50)
            print("TESTING THE FULL MODEL")
            print()
            print("Model: full_model")
            print()
            print("Scores based on validation set:")
            print("Execution Time:", run_time, "seconds")
            print("Accuracy Score:",acc_score)
            print("F1 Score:",f1score)
            print("Log Loss:", logloss)
            print("Prediction Score:", prediction_score)
            print('-' * 50)
            print()
            sys.stdout = sys.__stdout__
        
        # Test LDA with all input parameters 
        for config_name, config_params in parameters.items():
            
            model = LDA(**config_params)
            model_name = config_name
            with open(path, mode='a+') as file:
            # Redirect sys.stdout to the file
                sys.stdout = file   
                print('-' * 50)
                print(f"TESTING THE MODEL: {model_name}")
                print('-' * 30)
                sys.stdout = sys.__stdout__
            
            #Test model with all features present if not full model as already done this
            if config_name != 'baseline':
                fs_flag = "N"
                fs_method = ""
                new_feat_num = feat_num
                feat_pct = 1
                start = datetime.now()
                
                ## Fit the model
                model.fit(train_features, train_labels).transform(train_features)

                ## Hyperparameter tuning
                pred = model.predict(val_features)
                pred_prob = model.predict_proba(val_features)

                ## End model fitting time
                end = datetime.now()
                run_time = (end-start).total_seconds()

                ## Assign correct parameters based on number of classes 
                if n_classes != 2: 
                    f1_avg = 'weighted'
                else:
                    f1_avg = 'binary'

                ## Accuracy Metrics
                acc_score = accuracy_score(val_labels, pred)
                f1score = f1_score(val_labels, pred, average=f1_avg)
                logloss = log_loss(val_labels, pred_prob)
                prediction_score = ((f1score+acc_score)/(2*logloss))     

                ## Enter Data into Model Dictionary
                entry = {model_name : [config_params, run_time, colour_change.title(), fs_flag, fs_method, new_feat_num, feat_pct, acc_score, f1score, logloss, prediction_score]}
                models.update(entry)
                
                ## Write to log file 
                with open(path, mode='a+') as file:
                    # Redirect sys.stdout to the file
                    sys.stdout = file   
                    print(f"Model: {model_name}")
                    print(f"Parameters: {config_params}")                      
                    print()
                    print("Scores based on validation set:")
                    print("Execution Time:", run_time, "seconds")
                    print("Accuracy Score:",acc_score)
                    print("F1 Score:",f1score)
                    print("Log Loss:", logloss)
                    print("Prediction Score:", prediction_score)
                    print('-' * 20)
                    # Reset sys.stdout to the console
                    sys.stdout = sys.__stdout__
        
            # Remove zero variance features (constant features) 
            rem_const_feat = VarThres()
            train_features = rem_const_feat.fit_transform(train_features)
            test_features = rem_const_feat.fit_transform(test_features)
            val_features = rem_const_feat.fit_transform(val_features)
            num_con_feat_rem = abs(train_features.shape[1] - feat_num)
            with open(path, mode='a+') as file:
                    sys.stdout = file
                    print('FEATURE SELECTION')
                    print("Number of Constant Features Removed:",num_con_feat_rem)
                    sys.stdout = sys.__stdout__

            ## Test model with reduced features if necessary
            if num_con_feat_rem > 0: 
                fs_flag = "Y"
                fs_method = "Removal of Constant Features"
                model_name = f"{model_name}_no_const_feat"
                new_feat_num = feat_num - num_con_feat_rem
                feat_pct = new_feat_num/feat_num
                start = datetime.now()

                ## Fit the model
                model.fit(train_features, train_labels).transform(train_features)

                ## Hyperparameter tuning
                pred = model.predict(val_features)
                pred_prob = model.predict_proba(val_features)

                ## End model fitting time
                end = datetime.now()
                run_time = (end-start).total_seconds()

                ## Assign correct parameters based on number ofclasses 
                if n_classes != 2: 
                    f1_avg = 'weighted'
                else:
                    f1_avg = 'binary'

                ## Accuracy Metrics
                acc_score = accuracy_score(val_labels, pred)
                f1score = f1_score(val_labels, pred, average=f1_avg)
                logloss = log_loss(val_labels, pred_prob)
                prediction_score = ((f1score+acc_score)/(2*logloss))
           
                model_name = f'{config_name}_{feat_pct:.0%}'
                
                ## Enter Data into Model Dictionary
                entry = {model_name : [config_params, run_time, colour_change.title(), fs_flag, fs_method, new_feat_num, feat_pct, acc_score, f1score, logloss, prediction_score]}
                models.update(entry)
                
                ## Write to log file 
                with open(path, mode='a+') as file:
                    # Redirect sys.stdout to the file
                    sys.stdout = file   
                    print('-' * 20)
                    print(f"Model: {model_name}")
                    print(f"Parameters: {config_params}")
                    print(f"Feature Selection?: {fs_flag}")
                    print(f"Method of Feature Selection: {fs_method}")                    
                    print(f"New number of features: {new_feat_num}")
                    print(f"Percentage of kept features: {feat_pct:.0%}")    
                    print()
                    print("Scores based on validation set:")
                    print("Execution Time:", run_time, "seconds")
                    print("Accuracy Score:",acc_score)
                    print("F1 Score:",f1score)
                    print("Log Loss:", logloss)
                    print("Prediction Score:", prediction_score)
                    print('-' * 20)
                    # Reset sys.stdout to the console
                    sys.stdout = sys.__stdout__
                    
            else:
                feat_pct = 1

            ## Specify range for feature selection
            feat_pct = round((feat_pct*100),5)-5
            fs_pct = range(feat_pct,55,-5)

            # Feature Selection based on MI 
            for j in fs_pct: 

                ## Specify Model and Model Name
                model = LDA(**config_params)
                model_name = f"{config_name}_{j}"

                ## Start Timer
                start = datetime.now()

                ## Flags for Output Data
                fs_flag = "Y" 
                fs_method = "Mutual Information"
                train_features_fs = select_best_features_pct(train_features, MI_Scores, j)
                val_features_fs = select_best_features_pct(val_features, MI_Scores, j) 

                new_feat_num = train_features_fs.shape[1]
                feat_pct = j/100

                ## Fit the model
                model.fit(train_features_fs, train_labels).transform(train_features_fs)

                ## Hyperparameter tuning
                pred = model.predict(val_features_fs)
                pred_prob = model.predict_proba(val_features_fs)

                ## End model fitting time
                end = datetime.now()
                run_time = (end-start).total_seconds()

                ## Ensure scores accurate for number of classes
                if n_classes != 2: 
                    f1_avg = 'weighted'
                else:
                    f1_avg = 'binary'

                ## Accuracy Metrics
                acc_score = accuracy_score(val_labels, pred)
                f1score = f1_score(val_labels, pred, average=f1_avg)
                logloss = log_loss(val_labels, pred_prob)
                prediction_score = ((f1score+acc_score)/(2*logloss))

                ## Update models dictionary
                entry = {model_name : [config_params, run_time, colour_change.title(), fs_flag, fs_method, new_feat_num, feat_pct, acc_score, f1score, logloss, prediction_score]}
                models.update(entry)

                ## Write data to log file 
                with open(path, mode='a+') as file:
                    # Redirect sys.stdout to the file
                    sys.stdout = file   
                    print('-' * 20)
                    print(f"Model: {model_name}")
                    print(f"Parameters: {config_params}")
                    print(f"Feature Selection?: {fs_flag}")
                    print(f"Method of Feature Selection: {fs_method}")

                    if fs_flag == "Y":
                        print(f"New number of features: {new_feat_num}")
                        print(f"Percentage of kept features: {feat_pct:.0%}")

                    print()
                    print("Scores based on validation set:")
                    print("Execution Time:", run_time, "seconds")
                    print("Accuracy Score:",acc_score)
                    print("F1 Score:",f1score)
                    print("Log Loss:", logloss)
                    print("Prediction Score:", prediction_score)
                    print('-' * 20)
                    print()

                    # Reset sys.stdout to the console
                    sys.stdout = sys.__stdout__
         
        # Create graph showing percentage of features with accuracy score
        scoring = ['Accuracy','F1','Log Loss','Prediction']
        fs_pct = range(100,55,-5)
        for index, score in enumerate(scoring):
            i = int(index + 7)
            baseline_data = [value[i] for key, value in models.items() if 'full' in key] + [value[i] for key, value in models.items() if 'baseline' in key]
            plt.clf()
            plt.cla()
            plt.scatter(fs_pct[1:], baseline_data[1:], zorder = 2)
            plt.plot(fs_pct[1:], baseline_data[1:], zorder = 1, label = "Baseline")
            plt.scatter(fs_pct[0], baseline_data[0], zorder = 2, c='black', marker='D', s=50, label = "Full Model")
            plt.gca().invert_xaxis()
            plt.xlabel('Percentage of Original Features')
            plt.ylabel(f'{score} Score')
            plt.title(f'LDA {score} score calculated on the validation dataset \n as a result of feature selection for {name}')
            plt.legend(bbox_to_anchor=(0.3, -0.15), loc='upper left')
            plt.tight_layout()
            plt.savefig(folder_path+f'{name}_LDA_graph_{colour_change.lower()}_{score.replace(" ", "")}.jpeg')           

        #Order the optimal dictionary into a list displaying the optimal model at the top
        models = [[key] + value for key, value in models.items()]
        models = sorted(models, key=lambda item: (-item[11],item[10]))

        # Write results to an Excel File
        df = pd.DataFrame(models, columns = ['Model Name','Parameters','Run Time','Feature Colour','Feature Selection', 'Method of Feature Selection', 'Number of Features', 'Percentage of Original Features','Accuracy Score', 'F1 Score', 'Log Loss','Prediction Score'])
        df.to_csv(folder_path+f'model_training_{colour_change.lower()}.csv', index=False)

    elif stage == "TEST":

        for config_name, config_params in parameters.items():
            opt_model = LDA(**config_params)
            pct = int(input("What percentage of features would you like to keep? "))

            # Test the full model on test dataset
            ## Build and fit the model
            start = datetime.now()
            full_model = LDA() 
            full_model.fit(train_features, train_labels).transform(train_features)

            ## Complete Predictions 
            pred = full_model.predict(test_features)
            pred_prob = full_model.predict_proba(test_features)
            
             ## End model fitting time
            end = datetime.now()
            run_time = (end-start).total_seconds()

            ## Calculate Accuracy Metrics
            if n_classes != 2: 
                f1_avg = 'weighted'
            else:
                f1_avg = 'binary'

            acc_score = accuracy_score(test_labels, pred)
            f1score = f1_score(test_labels, pred, average=f1_avg)
            logloss = log_loss(test_labels, pred_prob)
            prediction_score = (f1score+acc_score)/(2*logloss)

            ## Write to log file 
            path3 = folder_path+f'lda_results_{colour_change.lower()}_optimal.txt'
            with open(path3, mode='w') as file:
                sys.stdout = file 
                print("LDA STATISTICS")
                print('-' * 50)
                print("TESTING THE FULL MODEL")
                print()
                print("Model: full_model")
                print()
                print("Scores based on test set:")
                print("Execution Time:", run_time, "seconds")
                print("Accuracy Score:",acc_score)
                print("F1 Score:",f1score)
                print("Log Loss:", logloss)
                print("Prediction Score:", prediction_score)
                print('-' * 50)
                print()
                sys.stdout = sys.__stdout__

            if pct != 100:
                # Load or Generate MI Scores for training dataset
                score_path = file_path+f"Results/{name}/mutual_information_{colour_change}.npy"
                if os.path.exists(score_path):
                    MI_Scores = np.load(score_path)
                    print("Mutual Information Scores loaded")

                else: 
                    MI_Scores = MI(train_features, train_labels, discrete_features = False, random_state = random_seed)
                    np.save(score_path, MI_Scores)

                # Flags for Output 
                fs_flag = "Y"
                fs_method = "Mutual Information"

                # Features Selection
                train_features = select_best_features_pct(train_features, MI_Scores, pct)
                test_features = select_best_features_pct(test_features, MI_Scores, pct)

            else:
                fs_flag = "N"
                fs_method = ""


            # Fit the optimal model
            start = datetime.now()
            opt_model.fit(train_features, train_labels).transform(train_features)

            pred = opt_model.predict(test_features)
            pred_prob = opt_model.predict_proba(test_features)
            
             ## End model fitting time
            end = datetime.now()
            run_time = (end-start).total_seconds()

            if n_classes != 2: 
                f1_avg = 'weighted'
            else:
                f1_avg = 'binary'

            acc_score = accuracy_score(test_labels, pred)
            f1score = f1_score(test_labels, pred, average=f1_avg)
            logloss = log_loss(test_labels, pred_prob)
            prediction_score = (f1score+acc_score)/(2*logloss)

            with open(path3, mode='a+') as file:
                sys.stdout = file 
                print('-' * 50)
                print("TESTING THE OPTIMAL MODEL")
                print()
                print(f"Model Name: {config_name}")
                print(f"Parameters: {config_params}")
                print(f"Feature Selection?: {fs_flag}")
                print(f"Method of Feature Selection: {fs_method}")

                if fs_flag == "Y":
                    print(f"Percentage of kept features: {pct}")

                print()
                print("Scores based on test set:")
                print("Execution Time:", run_time, "seconds")
                print("Accuracy Score:",acc_score)
                print("F1 Score:",f1score)
                print("Log Loss:", logloss)
                print("Prediction Score:", prediction_score)
                sys.stdout = sys.__stdout__
    
def logistic_regression(name, train, test, val, stage, random_seed, parameters,colour_change):

    # Extract features and labels from output
    train_features = train[0]
    train_labels = train[1]
    test_features = test[0]
    test_labels = test[1]
    val_features = val[0]
    val_labels = val[1]
    
    # Make folder for Logistic Regression results
    try:  
        folder_path = file_path+f'Results/{name}/Logistic Regression/'
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
    
    # Get number of classes
    n_classes = len(np.unique(train_labels))
    
    # Remove spatial structure from features to work with sklearn
    train_features = train_features.view(train_features.size(0), -1)
    val_features = val_features.view(val_features.size(0), -1)
    test_features = test_features.view(test_features.size(0), -1)
    
    # If module in the training stage
    if stage == "TRAIN":    
    
        #Start writing to file
        path = folder_path+f'logreg_results_{colour_change.lower()}.txt'
        with open(path, mode='w') as file:
            sys.stdout = file   
            print(f"LOGISTIC REGRESSION STATISTICS")
            sys.stdout = sys.__stdout__ 

        # Define dictionary of model scores
        models = {}

        # Load or Generate MI Scores for training dataset
        score_path = file_path+f"Results/{name}/mutual_information_{colour_change}.npy"
        if os.path.exists(score_path):
            MI_Scores = np.load(score_path)
            print("Mutual Information Scores loaded")

        else: 
            MI_Scores = MI(train_features, train_labels, discrete_features = False, random_state = random_seed)
            np.save(score_path, MI_Scores)
        
        # FULL MODEL TEST
        ## Specify Full Model and Flags
        full_model = LogReg(max_iter = 500, random_state = random_seed, multi_class = 'ovr', n_jobs = -1)
        fs_flag = "N"
        fs_method = ""
        feat_num = train_features.shape[1]
        feat_pct = 1
        
        ## Fit Model
        start = datetime.now()
        full_model.fit(train_features, train_labels)
        
        ## Hyperparameter tuning
        pred = full_model.predict(val_features)
        pred_prob = full_model.predict_proba(val_features)

        ## End model fitting time
        end = datetime.now()
        run_time = (end-start).total_seconds()

        ## Check convergence and ensure correct parameters specified based on number of classes
        if n_classes != 2: 
            f1_avg = 'weighted'
            if int(sum(full_model.n_iter_))/n_classes < int(full_model.max_iter):
                converg = "Y"
            else:
                converg = "N"

        else:
            f1_avg = 'binary'
            if int(full_model.n_iter_) < int(full_model.max_iter):
                converg = "Y"
            else:
                converg = "N"

        ## Ensure correct parameters specified based on number of classes
        if n_classes != 2: 
            f1_avg = 'weighted'
        else:
            f1_avg = 'binary'

        ## Accuracy Metrics
        acc_score = accuracy_score(val_labels, pred)
        f1score = f1_score(val_labels, pred, average=f1_avg)
        logloss = log_loss(val_labels, pred_prob)
        prediction_score = ((f1score+acc_score)/(2*logloss))

        ## Enter data into the dictionary
        entry = {'full_model' : [None, run_time, colour_change.title(), None, None, feat_num, feat_pct, converg, acc_score, f1score, logloss, prediction_score]}
        models.update(entry)
        
        ## Write to log file 
        with open(path, mode='a+') as file:
            # Redirect sys.stdout to the file
            sys.stdout = file   
            print('-' * 50)
            print("TESTING THE FULL MODEL")
            print()
            print("Model: full_model")
            print(f"Convergence?: {converg}")
            print()
            print("Scores based on validation set:")
            print("Execution Time:", run_time, "seconds")
            print("Accuracy Score:",acc_score)
            print("F1 Score:",f1score)
            print("Log Loss:", logloss)
            print("Prediction Score:", prediction_score)
            print('-' * 50)
            print()
            sys.stdout = sys.__stdout__
            
        # Test Logistic Regression with all input parameters 
        for config_name, config_params in parameters.items(): 

            model = LogReg(**config_params)
            model_name = config_name
            
            feat_num = train_features.shape[1]
            
            with open(path, mode='a+') as file:
            # Redirect sys.stdout to the file
                sys.stdout = file   
                print('-' * 50)
                print(f"TESTING THE MODEL: {model_name}")
                print('-' * 30)
                sys.stdout = sys.__stdout__
            
            #Test model with all features present if not full model as already done this
            if config_name not in ['Ridge_LBFGS', 'LASSO']:
                
                fs_flag = "N"
                fs_method = ""
                new_feat_num = feat_num
                feat_pct = 1
                
                start = datetime.now()
                model.fit(train_features, train_labels)
                
                ## Hyperparameter tuning
                pred = model.predict(val_features)
                pred_prob = model.predict_proba(val_features)

                ## End model fitting time
                end = datetime.now()
                run_time = (end-start).total_seconds()
                
                ## Check convergence and ensure correct parameters specified based on number of classes
                if n_classes != 2: 
                    f1_avg = 'weighted'
                    if int(sum(model.n_iter_))/n_classes < int(model.max_iter):
                        converg = "Y"
                    else:
                        converg = "N"
                    
                else:
                    f1_avg = 'binary'
                    if int(model.n_iter_) < int(model.max_iter):
                        converg = "Y"
                    else:
                        converg = "N"
                        
                ## Accuracy Metrics
                acc_score = accuracy_score(val_labels, pred)
                f1score = f1_score(val_labels, pred, average=f1_avg)
                logloss = log_loss(val_labels, pred_prob)
                prediction_score = ((f1score+acc_score)/(2*logloss))

                ## Enter Data into Model Dictionary
                entry = {model_name : [config_params, run_time, colour_change.title(), fs_flag, fs_method, new_feat_num, feat_pct, converg, acc_score, f1score, logloss, prediction_score]}
                models.update(entry)
                
                ## Write to log file 
                with open(path, mode='a+') as file:
                    # Redirect sys.stdout to the file
                    sys.stdout = file   

                    print(f"Model: {model_name}")
                    print(f"Parameters: {config_params}")
                    print(f"Convergence?: {converg}")
                    print()
                    print("Scores based on validation set:")
                    print("Execution Time:", run_time, "seconds")
                    print("Accuracy Score:",acc_score)
                    print("F1 Score:",f1score)
                    print("Log Loss:", logloss)
                    print("Prediction Score:", prediction_score)
                    print('-' * 30)
                    sys.stdout = sys.__stdout__
            
            # Remove zero variance features (constant features) 
            rem_const_feat = VarThres()
            train_features = rem_const_feat.fit_transform(train_features)
            test_features = rem_const_feat.fit_transform(test_features)
            val_features = rem_const_feat.fit_transform(val_features)
            num_con_feat_rem = abs(train_features.shape[1] - feat_num)
            with open(path, mode='a+') as file:
                sys.stdout = file
                print('FEATURE SELECTION')
                print("Number of Constant Features Removed:",num_con_feat_rem)
                sys.stdout = sys.__stdout__

            # Test model with reduced features if necessary 
            if num_con_feat_rem > 0: 
                fs_flag = "Y"
                fs_method = "Removal of Constant Features"
                model_name = f"{model_name}_no_const_feat"
                new_feat_num = feat_num - num_con_feat_rem
                feat_pct = new_feat_num/feat_num
            
                # Fit the model
                start = datetime.now()
                model.fit(train_features_fs, train_labels)

                # Hyperparameter tuning
                pred = model.predict(val_features_fs)
                pred_prob = model.predict_proba(val_features_fs)

                # End model fitting time
                end = datetime.now()
                run_time = (end-start).total_seconds()

                ## Check convergence and ensure correct parameters specified based on number of classes
                if n_classes != 2: 
                    f1_avg = 'weighted'
                    if int(sum(model.n_iter_))/n_classes < int(model.max_iter):
                        converg = "Y"
                    else:
                        converg = "N"
                    
                else:
                    f1_avg = 'binary'
                    if int(model.n_iter_) < int(model.max_iter):
                        converg = "Y"
                    else:
                        converg = "N"
                    
                ## Accuracy Metrics
                acc_score = accuracy_score(val_labels, pred)
                f1score = f1_score(val_labels, pred, average=f1_avg)
                logloss = log_loss(val_labels, pred_prob)
                prediction_score = ((f1score+acc_score)/(2*logloss))
                    
                model_name = f'{config_name}_{feat_pct:.0%}'
                
                ## Enter Data into Model Dictionary
                entry = {model_name : [config_params, run_time, colour_change.title(), fs_flag, fs_method, new_feat_num, feat_pct, converg, acc_score, f1score, logloss, prediction_score]}
                models.update(entry)
                
                ## Write to log file 
                with open(path, mode='a+') as file:
                # Redirect sys.stdout to the file
                    sys.stdout = file   
                    print('-' * 20)
                    print(f"Model: {model_name}")
                    print(f"Parameters: {config_params}")
                    print(f"Convergence?: {converg}")
                    print(f"Feature Selection?: {fs_flag}")
                    print(f"Method of Feature Selection: {fs_method}")

                    if "Y" in fs_flag:
                        print(f"New number of features: {new_feat_num}")
                        print(f"Percentage of kept features: {feat_pct:.0%}")
                            
                    print()
                    print("Scores based on validation set:")
                    print("Execution Time:", run_time, "seconds")
                    print("Accuracy Score:",acc_score)
                    print("F1 Score:",f1score)
                    print("Log Loss:", logloss)
                    print("Prediction Score:", prediction_score)
                    print('-' * 20)
                    print()
                    sys.stdout = sys.__stdout__

            else:
                feat_pct = 1
                
            ## Specify range for feature selection
            feat_pct = round((feat_pct*100),5)-5
            fs_pct = range(feat_pct,55,-5)
               
            if config_params.get('penalty') in ['l2', None]:

                for j in fs_pct:
                    ## Specify Model and Model Name
                    model = LogReg(**config_params)
                    model_name = f"{config_name}_{j}"

                    ## Start Timer
                    start = datetime.now()

                    ## Flags for Output Data
                    fs_flag = "Y" 
                    fs_method = "Mutual Information"
                    train_features_fs = select_best_features_pct(train_features, MI_Scores, j)
                    val_features_fs = select_best_features_pct(val_features, MI_Scores, j) 

                    new_feat_num = train_features_fs.shape[1]
                    feat_pct = j/100

                    model.fit(train_features_fs, train_labels)
                    
                    ## Hyperparameter tuning
                    pred = model.predict(val_features_fs)
                    pred_prob = model.predict_proba(val_features_fs)

                    ## End model fitting time
                    end = datetime.now()
                    run_time = (end-start).total_seconds()

                    ## Check convergence and ensure correct parameters specified based on number of classes
                    if n_classes != 2: 
                        f1_avg = 'weighted'
                        if int(sum(model.n_iter_))/n_classes < int(model.max_iter):
                            converg = "Y"
                        else:
                            converg = "N"

                    else:
                        f1_avg = 'binary'
                        if int(model.n_iter_) < int(model.max_iter):
                            converg = "Y"
                        else:
                            converg = "N"

                    ## Accuracy Metrics
                    acc_score = accuracy_score(val_labels, pred)
                    f1score = f1_score(val_labels, pred, average=f1_avg)
                    logloss = log_loss(val_labels, pred_prob)
                    prediction_score = ((f1score+acc_score)/(2*logloss))

                    model_name = f'{config_name}_{feat_pct:.0%}'

                    ## Enter Data into Model Dictionary
                    entry = {model_name : [config_params, run_time, colour_change.title(), fs_flag, fs_method, new_feat_num, feat_pct, converg, acc_score, f1score, logloss, prediction_score]}
                    models.update(entry)
                    
                    with open(path, mode='a+') as file:
                        # Redirect sys.stdout to the file
                        sys.stdout = file   
                        print('-' * 20)
                        print(f"Model: {model_name}")
                        print(f"Parameters: {config_params}")
                        print(f"Convergence?: {converg}")
                        print(f"Feature Selection?: {fs_flag}")
                        print(f"Method of Feature Selection: {fs_method}")

                        if "Y" in fs_flag:
                            print(f"New number of features: {new_feat_num}")
                            print(f"Percentage of kept features: {feat_pct:.0%}")

                        print()
                        print("Scores based on validation set:")
                        print("Execution Time:", run_time, "seconds")
                        print("Accuracy Score:",acc_score)
                        print("F1 Score:",f1score)
                        print("Log Loss:", logloss)
                        print("Prediction Score:", prediction_score)
                        print('-' * 20)
                        print()
                        sys.stdout = sys.__stdout__
                        
            else: 

                fs_flag = "Y"
                fs_method = f"{config_params['penalty'].title()} regularisation"               

                start = datetime.now()
                model.fit(train_features, train_labels)
                
                # Hyperparameter tuning
                pred = model.predict(val_features)
                pred_prob = model.predict_proba(val_features)

                # End model fitting time
                end = datetime.now()
                run_time = (end-start).total_seconds() 
                
                ## Check convergence and ensure correct parameters specified based on number of classes
                if n_classes != 2:
                    f1_avg = 'weighted'
                    if int(sum(model.n_iter_))/n_classes < int(model.max_iter):
                        converg = "Y"
                    else:
                        converg = "N"
                    new_feat_num = np.mean(train_features.shape[1] - np.sum(model.coef_ == 0, axis=1))
                    feat_pct = round(new_feat_num/feat_num, 1)
                else:
                    f1_avg = 'binary'
                    if int(model.n_iter_) < int(model.max_iter):
                        converg = "Y"
                    else:
                        converg = "N" 
                    new_feat_num = int(train_features.shape[1] - np.sum(model.coef_ == 0, axis=1))
                    feat_pct = round(new_feat_num/feat_num, 1)

                model_name = f"{config_name}"

                acc_score = accuracy_score(val_labels, pred)
                f1score = f1_score(val_labels, pred, average=f1_avg)
                logloss = log_loss(val_labels, pred_prob)
                prediction_score = (f1score+acc_score)/(2*logloss)

                entry = {model_name : [config_params, run_time, colour_change.title(), fs_flag, fs_method, new_feat_num, feat_pct, converg, acc_score, f1score, logloss, prediction_score]}
                models.update(entry)


                with open(path, mode='a+') as file:
                    # Redirect sys.stdout to the file
                    sys.stdout = file   
                    print('-' * 20)
                    print(f"Model: {model_name}")
                    print(f"Parameters: {config_params}")
                    print(f"Convergence?: {converg}")
                    print(f"Feature Selection?: {fs_flag}")
                    print(f"Method of Feature Selection: {fs_method}")

                    if "Y" in fs_flag:
                        print(f"Mean number of features: {new_feat_num}")
                        print(f"Mean percentage of kept features: {feat_pct:.0%}")

                    print()
                    print("Scores based on validation set:")
                    print("Execution Time:", run_time, "seconds")
                    print("Accuracy Score:",acc_score)
                    print("F1 Score:",f1score)
                    print("Log Loss:", logloss)
                    print("Prediction Score:", prediction_score)
                    print('-' * 20)
                    print()

                    # Reset sys.stdout to the console
                    sys.stdout = sys.__stdout__

        # Create graphs showing scores by percentage of features
        scoring = ['Accuracy','F1','Log Loss','Prediction']
        LASSO_pct = (models['LASSO'][6]*100)
        fs_pct = range(100,55,-5)
        for index, score in enumerate(scoring):
            i = int(index + 8)
            Ridge1_data = [value[i] for key, value in models.items() if 'full' in key] + [value[i] for key, value in models.items() if 'Ridge_LBFGS' in key] 
            Ridge2_data = [value[i] for key, value in models.items() if 'Ridge_SAG' in key and 'Ridge_SAGA' not in key]
            Ridge3_data = [value[i] for key, value in models.items() if 'Ridge_SAGA' in key]
            LASSO_data = [value[i] for key, value in models.items() if 'LASSO' in key]
            plt.clf()
            plt.cla()
            plt.scatter(fs_pct[1:], Ridge1_data[1:], zorder = 2)
            plt.plot(fs_pct[1:], Ridge1_data[1:], label = 'Baseline', zorder = 1)
            plt.scatter(fs_pct, Ridge2_data, zorder = 2)
            plt.plot(fs_pct, Ridge2_data, label = 'Ridge with SAG solver', zorder = 1)
            plt.scatter(fs_pct, Ridge3_data, zorder = 2)
            plt.plot(fs_pct, Ridge3_data, label = 'Ridge with SAGA solver', zorder = 1)
            plt.scatter(LASSO_pct, LASSO_data, zorder = 2, label = 'LASSO')
            plt.scatter(fs_pct[0], Ridge1_data[0], zorder = 2, c='black', marker='D', s=50, label = "Full Model")
            plt.gca().invert_xaxis()
            plt.xlabel('Percentage of Original Features')
            plt.ylabel(f'{score} Score')
            plt.title(f'Logistic Regression {score} score calculated on the validation dataset \n as a result of feature selection for {name}')
            plt.legend(bbox_to_anchor=(0.3, -0.15), loc='upper left')
            plt.tight_layout()
            plt.savefig(folder_path+f'{name}_LR_graph_{colour_change.lower()}_{score.replace(" ", "")}.jpeg')  
            
        #Order the optimal dictionary into a list displaying the optimal model at the top
        models = [[key] + value for key, value in models.items()]
        models = sorted(models, key=lambda item: (-item[12],item[11]))
         
        # Write results to an Excel File
        df = pd.DataFrame(models, columns = ['Model Name','Parameters','Run Time','Feature Colour','Feature Selection', 'Method of Feature Selection', 'Number of Features', 'Percentage of Original Features', 'Convergence', 'Accuracy Score', 'F1 Score', 'Log Loss', 'Prediction Score'])
        df.to_csv(folder_path+f'model_training_{colour_change.lower()}.csv', index=False)

    elif stage == "TEST":
        
        for config_name, config_params in parameters.items():
            feat_num = train_features.shape[1]
            if config_params['penalty'] != 'l1':
                pct = int(input("What percentage of features would you like to keep? "))
            else:
                pct = 100
            
            # Test the full model on test dataset
            ## Build and fit the model
            start = datetime.now()
            full_model = LogReg(max_iter = 500, multi_class = 'ovr', n_jobs = -1) 
            full_model.fit(train_features, train_labels)

            ## Complete Predictions 
            pred = full_model.predict(test_features)
            pred_prob = full_model.predict_proba(test_features)
            end = datetime.now()
            run_time = (end-start).total_seconds()

            ## Check convergence and ensure correct parameters specified based on number of classes
            if n_classes != 2: 
                f1_avg = 'weighted'
                if int(sum(full_model.n_iter_))/n_classes < int(full_model.max_iter):
                    converg = "Y"
                else:
                    converg = "N"

            else:
                f1_avg = 'binary'
                if int(full_model.n_iter_) < int(full_model.max_iter):
                    converg = "Y"
                else:
                    converg = "N"

            acc_score = accuracy_score(test_labels, pred)
            f1score = f1_score(test_labels, pred, average=f1_avg)
            logloss = log_loss(test_labels, pred_prob)
            prediction_score = (f1score+acc_score)/(2*logloss)

            ## Write to log file 
            path3 = folder_path+f'logreg_results_{colour_change.lower()}_optimal.txt'
            with open(path3, mode='w') as file:
                sys.stdout = file 
                print("LOGISTIC REGRESSION STATISTICS")
                print('-' * 50)
                print("TESTING THE FULL MODEL")
                print()
                print("Model: full_model")
                print(f"Convergence?: {converg}")
                print()
                print("Scores based on test set:")
                print("Execution Time",run_time,"seconds")
                print("Accuracy Score:",acc_score)
                print("F1 Score:",f1score)
                print("Log Loss:", logloss)
                print("Prediction Score:", prediction_score)
                print('-' * 50)
                print()
                sys.stdout = sys.__stdout__
                
            if pct != 100:
                # Load or Generate MI Scores for training dataset
                score_path = file_path+f"Results/{name}/mutual_information_{colour_change}.npy"
                if os.path.exists(score_path):
                    MI_Scores = np.load(score_path)
                    print("Mutual Information Scores loaded")

                else: 
                    MI_Scores = MI(train_features, train_labels, discrete_features = False, random_state = random_seed)
                    np.save(score_path, MI_Scores)

                # Flags for Output 
                fs_flag = "Y"
                fs_method = "Mutual Information"

                # Features Selection
                train_features = select_best_features_pct(train_features, MI_Scores, pct)
                test_features = select_best_features_pct(test_features, MI_Scores, pct)

            elif config_params['penalty'] == 'l1':
                fs_flag = "Y"
                fs_method = "L1 regularisation"
                 
            # Test optimal model
            ## Fit model
            start = datetime.now()
            opt_model = LogReg(**config_params)
            opt_model.fit(train_features, train_labels)

            ## Complete Predictions 
            pred = opt_model.predict(test_features)
            pred_prob = opt_model.predict_proba(test_features)
            end = datetime.now()
            run_time = (end-start).total_seconds()          

            ## Check convergence and ensure correct parameters specified based on number of classes
            if n_classes != 2: 
                f1_avg = 'weighted'
                if int(sum(opt_model.n_iter_))/n_classes < int(opt_model.max_iter):
                    converg = "Y"
                else:
                    converg = "N"

            else:
                f1_avg = 'binary'
                if int(opt_model.n_iter_) < int(opt_model.max_iter):
                    converg = "Y"
                else:
                    converg = "N"

            # Get optimal model feature number if LASSO
            if config_params['penalty'] == 'l1':
                new_feat_num = np.mean(train_features.shape[1] - np.sum(opt_model.coef_ == 0, axis=1))
                pct = round(new_feat_num/feat_num, 1)

            acc_score = accuracy_score(test_labels, pred)
            f1score = f1_score(test_labels, pred, average=f1_avg)
            logloss = log_loss(test_labels, pred_prob)
            prediction_score = (f1score+acc_score)/(2*logloss)

            with open(path3, mode='a+') as file:
                sys.stdout = file 
                print('-' * 50)
                print("TESTING THE OPTIMAL MODEL")
                print()
                print(f"Model Name: {config_name}")
                print(f"Parameters: {config_params}")
                print(f"Convergence?: {converg}")
                print(f"Feature Selection?: {fs_flag}")
                print(f"Method of Feature Selection: {fs_method}")

                if fs_flag == "Y":
                    print(f"Percentage of kept features: {pct:.0%}")

                print()
                print("Scores based on test set:")
                print("Execution Time:",run_time,"seconds")
                print("Accuracy Score:",acc_score)
                print("F1 Score:",f1score)
                print("Log Loss:", logloss)
                print("Prediction Score:", prediction_score)
                sys.stdout = sys.__stdout__
                
