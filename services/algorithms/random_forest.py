from sklearn.ensemble import RandomForestClassifier
from services.algorithms.base import BaseScikitClassificationAlgorithm

class RandomForestAlgorithm(BaseScikitClassificationAlgorithm):
    def __init__(self):
        super().__init__(model=RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))

    def name(self) -> str:
        return "Random Forest"
