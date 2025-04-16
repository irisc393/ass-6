# ass-6
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score

from ucimlrepo import fetch_ucirepo 

heart_disease = fetch_ucirepo(id=45) 
df = heart_disease.data.original.copy()


#1
The UCI Heart Disease dataset is a binary classification problem. The goal is to predict whether the patient has heart disease based on 13 clinical features such as age, sex, cp (chest pain type), trestbps (resting blood pressure), chol (serum cholesterol levels), fbs (fasting blood sugar), restecg (resting ECG results), thalach (maximum heart rate achieved), exang (exercise-induced angina), oldpeak, slope, ca (major vessels) and thal (Thalassemia blood disorder).

Target variable is 'num'
  - `0` means patient does not have heart disease  
  - `1` means patient has heart disease

Since the output is categorical, classification algorithms such as Logistic Regression, K-Nearest Neighbors, or Random Forests can be applied. These models learn patterns in the data to accurately predict the health status of new, unseen patients based on their clinical indicators.

#2
y = heart_disease.data.targets
sns.countplot(
    data=y,
    x='num'
)

plt.show()

The transformation is needed to be apply, since we need let 0 represent no heart dieases and 1–4 represent have heart disease. Also, there are some missing data, therefore the transformation is needed.

#3
print(df.shape)
print(df.dtypes)
print(df.describe())
- There are 303 observations and 14 features(13 features + target 'num') with 9 numerical and 5 categorical
- Missing values exist in ca (4 missing) and thal (2 missing).
- The dataset contain the wide for blood pressure(trestbps) measurements, showing a high variability. And Cholesterol(chol) also has a wide spread, as 126 to 564, since 564 is much higher than the 75th percentile (275) suggesting possible outliers.
- Variables like sex, fbs, and exang appear to be binary (as the values of 0 or 1)

#4
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
print(df['target'].value_counts())
X = heart_disease.data.features
df = X.copy()
df['target'] = y_binary


#5
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Features and Target")
plt.show()

- Interpretation 
1. **`Thalach` (maximum heart rate achieved)** shows the **strongest negative correlation** with the target variable (`target = 1`) at **-0.42**. This suggests that patients with **lower maximum heart rates** are more likely to have heart disease. 
2. Several features exhibit **moderate to strong positive correlations** with the target variable:
   - `cp` (chest pain type): **0.41**
   - `oldpeak` (ST depression): **0.42**
   - `exang` (exercise-induced angina): **0.43**
   - `ca` (number of major vessels): **0.46**
   - `thal` (thalassemia): **0.53**
    These variables are strongly associated with the likelihood of heart disease. 
Overall, these correlation insights help us identify the most relevant predictors for the target outcome. By prioritizing features like `thal`, `ca`, `cp`, and `thalach`, we can potentially improve model accuracy and reduce noise from weakly related variables.

#6
df.dropna(inplace=True)
print("Number of observations after dropping missing values:", df.shape[0])

- We dropped all rows containing missing values. After this step, the dataset contains 297 observations.

#7
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
X_numerical = df[numerical_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numerical)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=2, random_state=1)
clusters = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7,c=clusters, cmap='viridis')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Sub-group Visualization")
plt.grid()
plt.show()

- We performed PCA using only numerical features to explore underlying sub-groups in the dataset without using labels or categorical variables. The resulting scatter plot shows a mild spread in the data, which may suggest some underlying structure, although no clearly defined clusters appear.
  
#8
X = df.drop(columns='target')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

#9
- For this assignment, I chose **Logistic Regression** and **K-Nearest Neighbors (KNN)** as our two classification models.

#### 1. **Logistic Regression**
- Logistic Regression is a widely used for binary classification problems like this one, and goal is to predict whether a person has heart disease (1) or not (0). It works by modeling the relationship between the features and the probability of the target outcome.
- **Justification:** 
  - It’s simple, fast, and gives good performance on linearly separable data.
  - It also gives useful coefficients that help us understand which features are important.
  - Since this dataset is not too large and has both numerical and categorical data.

#### 2. **K-Nearest Neighbors (KNN)**
- KNN is a non-parametric, instance-based learning algorithm that makes predictions based on the majority label of the nearest neighbors.
- **Justification:**
  - It does not assume a linear relationship between the input and output, which helps if the pattern in the data is more complex.
  - Can capture non-linear relationships that logistic regression miss.
  - Useful for comparison to assess whether a distance-based classifier performs better on this dataset than a linear model.


These two classifiers complement each other: Logistic Regression tests a linear assumption, and meanwhile KNN evaluates local proximity in feature space. Comparing their performance helps us assess which approach better fits the characteristics of the heart disease dataset.

#10
**1. Accuracy**  
Accuracy measures the proportion of correct predictions made by the model.
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$
- **TP**: True Positives — correctly predicted heart disease  
- **TN**: True Negatives — correctly predicted no disease  
- **FP**: False Positives — wrongly predicted disease  
- **FN**: False Negatives — wrongly predicted no disease

Accuracy gives an overall view of how well the classifier is performing, but it can be misleading when the dataset is imbalanced.

---

**2. F1 Score**  
The F1 Score is the harmonic mean of precision and recall. It balances both false positives and false negatives, and is especially useful when the dataset is imbalanced.
$$
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

- **Precision** = $$ \frac{TP}{TP + FP} $$: How many predicted positives are actually correct.  
- **Recall** = $$ \frac{TP}{TP + FN} $$: How many actual positives were correctly predicted.

The F1 score is more informative than accuracy when both types of errors matter, which is important in medical contexts.


#11
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

param_grid = {'n_neighbors': range(1, 21)}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train) 

best_k = grid_search.best_params_['n_neighbors']
print(f"Best k: {best_k}")  

param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],  
    'penalty': ['l2'],              
    'solver': ['liblinear']          
}

lr = LogisticRegression(max_iter=1000, random_state=42)
grid_lr = GridSearchCV(lr, param_grid_lr, cv=5)
grid_lr.fit(X_train_scaled, y_train)  # Use scaled data for consistency

print("Best LR Parameters:", grid_lr.best_params_)

- Logistic Regression has tuning parameters is at **C=0.01**. For KNN, we identified the best-performing value as **K=7**.


#12
selector = SelectKBest(score_func=f_classif, k=8)
X_train_kbest = selector.fit_transform(X_train, y_train)
X_test_kbest = selector.transform(X_test)

selected_mask = selector.get_support()
selected_features = X_train.columns[selected_mask]
print("Selected features:", selected_features.tolist())

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Add scaling for Logistic Regression
    ('select', SelectKBest(score_func=f_classif, k=8)),
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
])

param_grid = {
    'select__k': [5, 8, 10],  # Also tune number of features
    'logreg__C': [0.01, 0.1, 1, 10, 100],
    'logreg__penalty': ['l1', 'l2'],  # Test both penalties
    'logreg__solver': ['liblinear']  # Required for L1/L2
}

grid = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
   
)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
best_model = grid.best_estimator_
print("Test accuracy:", best_model.score(X_test, y_test))


- We performed feature selection using `SelectKBest` with the ANOVA F-test (`f_classif`) to extract the 8 most predictive features from the training data. The selected features we choose includes: sex, cp (chest pain type), thalach (max heart rate), exang (exercise-induced angina), oldpeak, slope, ca, thal

- Using only these selected features, we trained a third classifier Logistic Regression, and the best-performing value of `C` is 0.01, and the final classifier was trained using this optimal parameter.


#13
X_test_scaled = scaler.transform(X_test)

- 1. Logistic
y_pred_lr = grid_lr.best_estimator_.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

- 2. KNN
knn = KNeighborsClassifier(n_neighbors=7)  # use best k=7 we get from (11)
knn.fit(X_train_scaled, y_train) 
y_pred_knn = knn.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

- 3. Logistic Regression with KBest
y_pred_kbest = best_model.predict(X_test)
accuracy_kbest = accuracy_score(y_test, y_pred_kbest)
f1_kbest = f1_score(y_test, y_pred_kbest)


print(f"- Logistic Regression: Accuracy = {accuracy_lr:.3f}, F1 = {f1_lr:.3f}")
print(f"- KNN (k=7): Accuracy = {accuracy_knn:.3f}, F1 = {f1_knn:.3f}")
print(f"- LR + KBest: Accuracy = {accuracy_kbest:.3f}, F1 = {f1_kbest:.3f}")

**Findings:**
1. Logistic Regression using all features achieved an accuracy of **0.844** and an F1 score of **0.821**, making it the best-performing model.
2. K-Nearest Neighbors performed significantly worse with an accuracy of **0.833** and F1 score of **0.805**, suggesting that it's more sensitive to noisy or high-dimensional data.
3. Feature selection using `KBest` slightly reduced performance (accuracy = **0.856**, F1 = **0.831**) but helped simplify the model. This can improve generalization and reduce overfitting, especially in larger datasets.

#14
for feature, coef in zip(selected_features, log_reg_kbest.coef_[0]):
    print(f"{feature}: {coef:.4f}")

**Sex (1 = male):**
- The feature sex has the largest positive coefficient (1.0147), indicating that being male is strongly associated with a higher risk of heart disease in this dataset.

**Exang (Exercise-Induced Angina):**
- With a high positive coefficient (0.8416), patients who experience exercise-induced chest pain are more likely to have heart disease, which is consistent with clinical intuition.


#15
- Standardize
X_numerical = df.drop(columns=['num', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numerical)

- PCA & Clustering
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=2, random_state=1)
clusters = kmeans.fit_predict(X_pca)

- Add cluster to original
X_clustered = X.copy()
X_clustered['cluster'] = clusters

- Train-test split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clustered, y, test_size=0.3, random_state=1, stratify=y
)

- Train logistic reg with the added cluster 
log_reg_cluster = LogisticRegression(max_iter=1000)
log_reg_cluster.fit(X_train_c, y_train_c)

- acc and f1
y_pred_cluster = log_reg_cluster.predict(X_test_c)
accuracy_cluster = accuracy_score(y_test_c, y_pred_cluster)
f1_cluster = f1_score(y_test_c, y_pred_cluster)

print("LogReg with Cluster Feature - Accuracy:", round(accuracy_cluster, 3))
print("LogReg with Cluster Feature - F1 Score:", round(f1_cluster, 3))

- We explored a subgroup-based strategy to enhance classifier performance by incorporating unsupervised structure from the data. Additionally, we applied PCA followed by KMeans clustering to identify potential latent sub-groups within the dataset.

- These cluster labels were then added as a new feature to the dataset. We trained a fourth model, Logistic Regression with this new cluster feature included, and evaluated its performance using the same metrics: Accuracy and F1 Score.


#### Performance Comparison

| Model                        | Accuracy | F1 Score |
|-----------------------------|----------|----------|
| Logistic Regression         | 0.856    | 0.835    |
| KNN                         | 0.611    | 0.507    |
| LogReg + KBest              | 0.867    | 0.846    |
| **LogReg + Cluster Feature**| 0.856    | 0.835    |

---

#### Conclusion

Although the addition of the cluster-based subgroup feature did not lead to a performance increase in this case, it still demonstrates a **valid data-driven strategy** for model enhancement. Sub-group features can be useful for more complex or nonlinear datasets where hidden patterns may be more influential.



