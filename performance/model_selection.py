from typing import Dict

import pandas as pd
from sklearn.model_selection import ParameterGrid

from constants import *
from evaluate import accuracy, macro_f1, weighted_f1
from required_model import required_model


def evaluate_hyperparameters():
    """
    Execute GridSearch to evaluate hyperparameters.
    :return:
    """
    param_grid = {HYPERPARAM_VOCABULARY: [0, 1],
                  HYPERPARAM_NGRAM: [1],
                  HYPERPARAM_DELTA: [0.5, 1]}
    gs = GridSearch(estimator=required_model, param_grid=param_grid, load=False,
                    scoring=MODEL_SCORE_EVALUATION_F1_WEIGHTED)
    gs.fit()

    print('\nGrid search results:')
    print('Best score: {}'.format(gs.best_score()))
    print('Best params: {}'.format(gs.best_params()))

    print('\nFetching top 5 results')
    print(gs.top_k_results(5))


class GridSearch:
    """
    Hyperparameter optimization tool that helps improve the performance of a model
    by finding optimal combination of hyperparameter values.

    Produce DataFrame of all hyperparameter combinations and corresponding classification scores.
    By default scoring technique is `accuracy`.

    To simplify information retrieval, tool allows retrieval of the already generated DataFrame,
    which skips generation process.
    By default DataFrame is generated.
    """

    def __init__(self, estimator, param_grid, load=False, scoring=MODEL_SCORE_EVALUATION_ACCURACY):
        """
        Init GridSearch.
        :param estimator: Instance of the model. Our Main() function
        :param param_grid: List of parameters.
        :param scoring: Score used for parameters comparison.
        :param load: Load existing GridSearch DataFrame from the memory.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.load = load
        self.df = pd.DataFrame

    def fit(self):
        """
        Fit data according to the params
        :return: DataFrame
        """
        if self.load and os.path.exists(GRID_SEARCH_SERIALIZE_FILE):
            self.df = pd.read_pickle(GRID_SEARCH_SERIALIZE_FILE)
        else:
            grid = ParameterGrid(self.param_grid)
            rows = []
            for params in grid:
                model_result = self.estimator(params[HYPERPARAM_VOCABULARY], params[HYPERPARAM_NGRAM],
                                              params[HYPERPARAM_DELTA], TRAINING_TWEETS_FILE_LOCATION,
                                              TEST_TWEETS_FILE_LOCATION)
                score = self.__get_evaluation_score(model_result)
                rows.append(self.__assemble_row(params, score))
            self.df = pd.DataFrame(rows,
                                   columns=[HYPERPARAM_VOCABULARY, HYPERPARAM_NGRAM, HYPERPARAM_DELTA, self.scoring])
        self.__finalize_df()

    def best_score(self):
        """
        Get best score out of all param combinations based on scoring method
        :return:
        """
        return self.df[self.scoring].max()

    def best_params(self):
        """
        Get the best combination of params based on scoring method
        :return:
        """
        idx = self.df[self.scoring].idxmax()
        best_params = {param: self.df.loc[idx, param] for (param, _) in self.param_grid.items()}
        return best_params

    def top_k_results(self, k: int):
        """
        Fetch top K results based on scoring method
        :param k:
        :return:
        """
        return self.df.head(k).to_string(index=False)

    def __get_evaluation_score(self, model_result):
        if self.scoring == MODEL_SCORE_EVALUATION_ACCURACY:
            return accuracy(model_result)
        elif self.scoring == MODEL_SCORE_EVALUATION_F1_MACRO:
            return macro_f1(model_result)
        elif self.scoring == MODEL_SCORE_EVALUATION_F1_WEIGHTED:
            return weighted_f1(model_result)
        else:
            raise ValueError('Unsupported scoring strategy for model evaluation!')

    def __assemble_row(self, params: Dict[str, int], score: int):
        """
        Constcut row of the result DataFrame
        :param params: Combination of params
        :param score:
        :return:
        """
        row = {param: param_value for (param, param_value) in params.items()}
        row[self.scoring] = score
        return row

    def __finalize_df(self):
        """
        Final DataFrame processing:
        - Sort based on scoring method
        - serialize DataFrame
        - write results to the file
        :return:
        """
        self.df.sort_values(by=self.scoring, ascending=False, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.__serialize()
        self.__write_to_file()

    def __serialize(self):
        """
        Serialize df
        :return:
        """
        self.df.to_pickle(GRID_SEARCH_SERIALIZE_FILE)

    def __write_to_file(self):
        """
        Write to file df
        :return:
        """
        with open(GRID_SEARCH_OUTPUT_FILE, 'w') as f:
            self.df.to_string(f, index=False, col_space=OUTPUT_FILE_SPACE_COUNT)
