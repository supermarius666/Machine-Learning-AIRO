import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Dataset creato manualmente
df = pd.DataFrame({
    "Furniture":   ["no", "yes", "no", "no", "yes"],
    "Nr_Rooms":    [3, 3, 4, 3, 4],
    "New_Kitchen": ["yes", "no", "no", "no", "no"],
    "Acceptable":  ["yes", "no", "yes", "no", "yes"]
})

X = df[["Furniture", "Nr_Rooms", "New_Kitchen"]]
y = df["Acceptable"]

### Data Preparation
X = df[["Furniture", "Nr_Rooms", "New_Kitchen"]].copy()
X["Furniture"] = X["Furniture"].map({"no": 0, "yes": 1})
X["New_Kitchen"] = X["New_Kitchen"].map({"no": 0, "yes": 1})
y = df["Acceptable"].map({"no": 0, "yes": 1})

### Model training 
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)
model.fit(X, y)

### Plot the tree
plt.figure(figsize=(14, 8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["no", "yes"],
    filled=True,
    rounded=True,
    impurity=True
)

plt.show()

### Prediction on new data
new_house = pd.DataFrame({
    "Furniture": [0],
    "Nr_Rooms": [4],
    "New_Kitchen": [0]
})

prediction = model.predict(new_house)

print(prediction)
print("Accettabile:", "yes" if prediction[0] == 1 else "no")