from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data import load_activations

LAYER = 15

acts, labels = load_activations("cities")

X = acts[:, LAYER, :]

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=0
)

probe = LogisticRegression(max_iter=2000, C=0.1)
probe.fit(X_train, y_train)

print(f"layer {LAYER} test accuracy: {probe.score(X_test, y_test):.4f}")
