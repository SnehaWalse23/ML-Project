import os
import sys
from dataclasses import dataclass

import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import load_object


@dataclass
class CustomData:
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: float
    writing_score: float

    def get_data_as_data_frame(self):
        return pd.DataFrame([{
            'gender': self.gender,
            'race_ethnicity': self.race_ethnicity,
            'parental_level_of_education': self.parental_level_of_education,
            'lunch': self.lunch,
            'test_preparation_course': self.test_preparation_course,
            'reading_score': self.reading_score,
            'writing_score': self.writing_score,
        }])


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join(project_root, 'artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join(project_root, 'artifacts', 'proprocessor.pkl')

    def predict(self, features):
        model = load_object(self.model_path)
        preprocessor = load_object(self.preprocessor_path)

        transformed_features = preprocessor.transform(features)
        return model.predict(transformed_features)
