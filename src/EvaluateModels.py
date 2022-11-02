import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.model_selection import KFold

class EvaluateModels:
    '''Base class to test text vectorizer and classifiers in a pipeline using GridSearchCV.
    
    Parameters
    ----------
    
    model_list: A list of tuples of models to be evaluated. 
        If this is a list of classifiers, use test_type = "classifier".
        If this is a list of vectorizers, use test_type = "vectorizer"
    constant_model: A tuple containing ('model name', model()) to be evaluated
    test_type: use either "vectorizer" or "classifier"
    params: A dictionary of parameters to be test using GridSearch. The 
            parameter key should start with the same str used for the 'name' 
            in the models tuple.
            Example: {'vectorizer': cvec, 'model__C': [0.001, 0.1, 1, 10, 100]}
    scoring: An sklearn scoring parameter
    '''

    def __init__(self, model_list, constant_model, test_type:str, scoring: str, params=None, num_folds: int = 5, seed: int = 23):
        if params is None:
            params = {}
        self.model_list = model_list
        self.constant_model = constant_model
        self.test_type = test_type
        self.scoring = scoring
        self.num_folds = num_folds
        self.seed = seed

        self.params = params
        self.cv_search = None # holds raw CV search results
        self.best_score = None
        self.best_model = None
        self.best_params = None
        # self.plot_labels = None
 
    def make_pipeline(self, model):
        '''Make pipeline with vectorizer and model'''
        if self.test_type == 'vectorizer':
            model_pipe = Pipeline([model, self.constant_model])           
        if self.test_type == 'classifier':
            model_pipe = Pipeline([self.constant_model, model]) 
        return model_pipe

    def store_best_results(self, model):
        '''Print the best cv test score and the optimal model'''
        if not self.best_score:
            self.best_score = np.mean(self.cv_search['test_score'])
            self.best_model = model
        elif np.mean(self.cv_search['test_score']) > self.best_score:
            self.best_score = np.mean(self.cv_search['test_score'])
            self.best_model = model
    
    # def get_plot_labels(self, grid_results):
    #     '''From GridSearchCV results, get list with x labels'''
    #     col_labels = [col for col in grid_results.columns if 'param_' in col]
    #     col_label = str(col_labels[0])
    #     plot_labels = grid_results[col_label].values.tolist()
    #     return [str(i).split('(')[0] for i in plot_labels]
    
    # def make_testscore_df(self, grid_results, plot_labels):
    #     '''From GridSearchCV results, get test scores to plot'''
    #     score_cols = [col for col in grid_results.columns if 'split' in col]
    #     cv_score_df = grid_results[score_cols].T
    #     nums = range(len(plot_labels))
    #     label_mapper = dict(zip(nums, plot_labels))
    #     return cv_score_df.rename(columns= label_mapper)
    
    def plot_grid_results(self, grid_search):
        '''Plot results from GridSearchCV for model parameters'''
        grid_results = pd.DataFrame(grid_search.cv_results_)
        
        col_labels = [col for col in grid_results.columns if 'param_' in col]
        col_label = str(col_labels[0])
        plot_labels = grid_results[col_label].values.tolist()
        clean_labels = [str(i).split('(')[0] for i in plot_labels]

        score_cols = [col for col in grid_results.columns if 'split' in col]
        cv_score_df = grid_results[score_cols].T
        nums = range(len(plot_labels))
        label_mapper = dict(zip(nums, clean_labels))
        cv_score_df = cv_score_df.rename(columns= label_mapper)

        _, ax = plt.subplots()
        cv_score_df.boxplot()
        plt.ylabel("cv_score")
        ax.set_xticklabels(clean_labels)
        plt.show()
    
    def print_best_cvresults(self):
        '''Print mean test score and corresponding best model from cross_validate results'''
        print('='*50)
        print(f'The best mean test score: {self.best_score}')
        print(f'The best model: {self.best_model}', end='\n')
    
    def print_cvresults(self, cv_results, cv_index):
        '''Get cross_validate results'''
        self.result_df = (pd.DataFrame(data = cv_results, index= cv_index,
                columns=['test_score', 'train_score', 'fit_time', 'score_time'], 
                ))
        print('='*50)
        print(self.result_df)

    def plot_testscores_box(self, test_scores, cv_index):
        '''Plot the test scores from cross_validate for each model as a boxplot'''
        nums = range(len(test_scores))
        label_mapper = dict(zip(nums, cv_index))
        data = pd.DataFrame(test_scores).T.rename(columns=label_mapper)
        # print('Dataframe: Test Scores for each Model Tested')
        # print(data)
        print('='*50)
        data.plot(kind='box', xlabel=cv_index)
        plt.ylabel("cv_score")
        plt.xticks(rotation=30)
        plt.show()

    # def plot_mean_results(self):
    #     '''Plot the results of the key parameter from grid search results
    #       Plots bar graph using mean value of scoring parameter.'''
    #     plot_data = self.result_df['test_score']
    #     plot_data.plot(kind='bar', use_index=True)
    #     plt.ylabel("cv_score")
    #     plt.show()
    
    def run(self, x_train, y_train):
        '''Creates pipeline then performs cross_validate.
        Prints best results, prints a table with cv results, 
        andn plots of all tested parameters'''
        
        X = x_train.values
        y = y_train.values

        cv_results = [] # mean model scores
        cv_std = []
        cv_index = [] # list of models
        test_scores = [] # list of model test scores

        for model_tuple in self.model_list: # for each tuple in list of tuples
            model = self.make_pipeline(model_tuple)

            if any(model_tuple[0] in key for key in self.params.keys()):
                print(f'Parameters found for {model_tuple[0]}')
                model = self.tune_hyperparameters(model_tuple, X,  y)
                cv_index.append(model_tuple[0])
                # print(f'Returned model: {model}')
                # returns model with best parameters
            else:
                kfold = KFold(n_splits=self.num_folds, random_state=self.seed, shuffle=True)
                self.cv_search = cross_validate(model, X, y, cv=kfold, scoring= self.scoring, return_train_score=True)        

                # get cv_search scores
                scores = {name:np.mean(value) for name, value in self.cv_search.items()}
                cv_results.append(scores)
                std = {name:np.std(value) for name, value in self.cv_search.items()}
                cv_std.append(std)
                cv_index.append(model_tuple[0])

                # store individual test scores
                model_test_scores = self.cv_search['test_score']
                test_scores.append(model_test_scores)

                # store best results
                self.store_best_results(model_tuple)
                print(f'Testing {model_tuple[0]} finished')

        self.print_best_cvresults()
        if not self.best_params:
            self.print_cvresults(cv_results, cv_index)
            self.plot_testscores_box(test_scores, cv_index)
    
    def tune_hyperparameters(self, model_tuple, X, y):
        '''Perform hyperparameter tuning and set best parameter for best performing model'''
        pipe = self.make_pipeline(model_tuple)
        kfold = KFold(n_splits=self.num_folds, random_state=self.seed, shuffle=True)
        grid_search = GridSearchCV(pipe, 
                                    param_grid = self.params, 
                                    cv=kfold, 
                                    scoring= self.scoring, 
                                    return_train_score=True)
        grid_search.fit(X, y)
        
        # store best score and parameters
        self.best_score = grid_search.best_score_
        self.best_params = grid_search.best_params_
        
        print(f'Best parameters found for {model_tuple[0]} : {self.best_params}')
        print(f'The best mean score is {self.best_score}')
        
        self.plot_grid_results(grid_search)

        pipe.set_params(**grid_search.best_params_)
        self.best_model = pipe.set_params(**grid_search.best_params_)
        return pipe

class EvaluatePreprocessors(EvaluateModels):
    '''Subclass that evaluates multiple preprocessors in a pipeline with one classifier'''
    def __init__(self, preprocessors: list, classifier, test_type, scoring, **kwargs):
        super().__init__(preprocessors, classifier, test_type, scoring=scoring, **kwargs)

class EvaluateClassifiers(EvaluateModels):
    '''Subclass that evaluates multiple classifiers in a pipeline with one preprocessor'''
    def __init__(self, classifiers: list, preprocessor, test_type, scoring, **kwargs):
        super().__init__(classifiers, preprocessor, test_type, scoring=scoring, **kwargs)
