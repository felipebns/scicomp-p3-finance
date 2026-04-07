from sklearn.linear_model import LogisticRegression
from services.algorithms.base import BaseScikitClassificationAlgorithm

class LogisticRegressionAlgorithm(BaseScikitClassificationAlgorithm):
    def __init__(self):
        super().__init__(model=LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"))

    def name(self) -> str:
        return "Logistic Regression"
