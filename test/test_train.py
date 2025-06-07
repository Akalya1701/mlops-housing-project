# You can run tests using pytest or unittest.

def test_model_accuracy_threshold():
    from train import model, X_test, y_test
    assert model.score(X_test, y_test) > 0.6
