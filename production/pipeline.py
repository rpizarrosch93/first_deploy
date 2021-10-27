from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler

from production.config.core import config
#from production..processing import features as pp

price_pipeline = Pipeline(
    [
        (
            "missing_imputation",
            CategoricalImputer(
                imputation_method = "missing",
                variables = config.model_config.categorical_vars_with_na_missing
            )
        ),
                (
            "frequent_imputation",
            CategoricalImputer(
                imputation_method = "frequent",
                variables = config.model_config.categorical_vars_with_na_frequent
            )
        )
    ]
)

