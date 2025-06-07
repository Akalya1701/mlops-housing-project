import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from train import model, X_test, y_test

def test_model_accuracy_threshold():
    score = model.score(X_test, y_test)
    assert score > 0.6, f"Model accuracy too low: {score}"

