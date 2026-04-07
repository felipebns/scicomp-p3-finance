from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from services.algorithms.base import BaseScikitClassificationAlgorithm

class EnsembleClassificationAlgorithm(BaseScikitClassificationAlgorithm):
    def __init__(self):
        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
        svc = SVC(random_state=42, probability=True, class_weight="balanced")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        
        ensemble = VotingClassifier(
            estimators=[('lr', lr), ('svc', svc), ('rf', rf)],
            voting='soft'
        )
        super().__init__(model=ensemble)

    def name(self) -> str:
        return "Hybrid Ensemble"
