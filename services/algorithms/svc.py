from sklearn.svm import SVC
from services.algorithms.base import BaseScikitClassificationAlgorithm

class SVCAlgorithm(BaseScikitClassificationAlgorithm):
    def __init__(self):
        super().__init__(model=SVC(random_state=42, probability=True, class_weight="balanced"))

    def name(self) -> str:
        return "Support Vector Classifier"
