import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import cProfile
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from sklearn.utils import resample
from sklearn.impute import KNNImputer
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier

from concurrent.futures import ProcessPoolExecutor, as_completed

from model_functions import logistic_test, svm_test, nb_test, rf_test, gb_test, knn_test, ensemble_test, stacking_test, run_model
import warnings


def flavor_profile(ingredients_df, ingr_df, comp_df, ingr_comp_df):
    ingredients_df.reset_index(drop=True, inplace=True)
    ingredients_df['id'] = ingredients_df.index

    function_df = ingredients_df.copy()
    undesired_columns = ['region', 'country', 'id']  # Keep 'id' out of calculations
    function_df.drop(undesired_columns, axis=1, inplace=True)
    sorted_ingredients = function_df.columns

    ingr_total = ingr_comp_df.merge(ingr_df, how='right', on='ingredient_id')
    ingr_total = ingr_total.merge(comp_df, how='right', on='compound_id')

    ingr_pivot = pd.crosstab(ingr_total['ingredient_name'], ingr_total['compound_id'])
    ingr_flavor = ingr_pivot.reindex(sorted_ingredients).fillna(0)

    df_flavor = pd.DataFrame(np.dot(function_df.values, ingr_flavor.values), index=ingredients_df.index)
    df_flavor['region'] = ingredients_df['region']
    df_flavor['id'] = ingredients_df['id']  # Add 'id' column

    # Reorder columns to have 'id' first
    cols = ['id'] + [col for col in df_flavor.columns if col != 'id']
    df_flavor = df_flavor[cols]

    # Reorder columns to have 'region' and 'country' first
    column_order = ['region'] + [col for col in df_flavor.columns if col not in ['region']]
    df_flavor = df_flavor[column_order]

    return df_flavor

def add_flavour_profile(ingredients_df, flavour_df):
    # Ensure 'id' is included as a unique identifier
    ingredients_df['id'] = range(len(ingredients_df))
    flavour_df['id'] = ingredients_df['id']

    # Ensure that 'flavour_df' and 'ingredients_df' have the same index
    flavour_df.index = ingredients_df.index
    ingredients_df.index = ingredients_df.index

    # Create a new DataFrame
    region_recipe_df = pd.DataFrame()
    region_recipe_df['id'] = ingredients_df['id']
    region_recipe_df['region'] = ingredients_df['region']

    # Calculate the total count of compounds per recipe from flavour_df
    region_recipe_df['compound_count'] = np.count_nonzero(flavour_df.values, axis=1)

    # Calculate the total count of ingredients per recipe from ingredients_df
    ingredient_columns = [col for col in ingredients_df.columns if col not in ['id', 'region']]
    region_recipe_df['ingredient_count'] = np.count_nonzero(ingredients_df[ingredient_columns].values, axis=1)

    return region_recipe_df