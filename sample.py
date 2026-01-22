import pandas as pd
import numpy as np
import dtale
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import math

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

#pip install dtale
#pip install pymysql

#pip install tensorflow
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.model_selection import StratifiedKFold, learning_curve
from skimage import io, color, filters, morphology, measure
import glob
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score





# Load your dataset
df = pd.read_csv(r"dt_ERA_corrected - Copy.csv")

# Display the DataFrame using D-Tale
d = dtale.show(df, host = 'localhost', port = 8000)

# Open the browser to view the interactive D-Tale dashboard
d.open_browser()


# Create a SQLAlchemy engine to connect to the MySQL database
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = "root", pw = "Vamsi%40235", db = "embryr24"))

# Display the first few rows of the DataFrame
df.head()

# Write the data from the DataFrame to the MySQL database table named 'groceries'
df.to_sql('embryo', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# Read data from the 'embryo' table in the database into a pandas DataFrame
sql = 'select * from embryo;'
df= pd.read_sql_query(sql, con = engine)

# Display the first few rows of the DataFrame
df.head()

df.info()

df.head()
df.describe()

df.isnull().sum()



# Step 2: Define feature groups
# ===================================
X = df[['endometrial_thickness_mm', 'cycle_day']]
y = df['ERA_status']

target_col = "ERA_status"
implantation_col = "implantation_outcome"

# ===================================
# Step 3: Handle missing values (EDA check)
# ===================================
print(df.isnull().sum().sort_values(ascending=False).head(10))

# Strategy:
# - Numeric → median imputation
# - Categorical → most frequent imputation

# ===================================
# Step 4: Custom transformer for outliers
# ===================================
class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip numeric values outside IQR bounds (winsorization style)."""
    def __init__(self, k=1.5):
        self.k = k
        self.bounds_ = {}

    def fit(self, X, y=None):
        # If numpy array, convert to DataFrame with generic column names
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - self.k * IQR, Q3 + self.k * IQR
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        for col in X.columns:
            low, high = self.bounds_[col]
            X[col] = np.clip(X[col], low, high)
        return X.values  # return as array for sklearn


# ===================================
# Step 5: Ordinal encoding for embryo scores
# ===================================
def map_scores(X):
    mapping = {"A": 3, "B": 2, "C": 1}
    cols = ["endometrium_pattern", "embryo_stage", "embryo_grade", "ICM_score", "TE_score"]

    # Rebuild DataFrame with correct column names
    X = pd.DataFrame(X, columns=cols)

    if "ICM_score" in X:
        X["ICM_score"] = X["ICM_score"].map(mapping)
    if "TE_score" in X:
        X["TE_score"] = X["TE_score"].map(mapping)

    return X

# ===================================
# Step 6: Build preprocessing pipelines
# ===================================

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, ['endometrial_thickness_mm', 'cycle_day'])
])


numeric_cols = [
    "age", "BMI", "endometrial_thickness_mm",
    "E2_pg_ml(Estradiol)", "P4_ng_ml(Progesterone)",
    "blastocoel_expansion"
]

categorical_cols = [
    "endometrium_pattern", "embryo_stage",
    "embryo_grade", "ICM_score", "TE_score"
]






#Load ResNet50 for feature extraction
# Pretrained ResNet50 (without top classification layer)
IMG_SIZE = (224, 224)
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
print("ResNet50 loaded for image feature extraction")


#Define the image feature extraction function
def extract_image_features(image_paths, model, img_size=IMG_SIZE):
    features, ids = [], []
    for path in image_paths:
        try:
            img = load_img(path, target_size=img_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            feat = model.predict(img_array, verbose=0)
            features.append(feat.flatten())
            ids.append(os.path.basename(path))  # keep filename only
        except Exception as e:
            print("Error:", path, e)
    return np.array(features), ids


#Collect all image files in the folder
image_folder = r"C:\Users\vamsi\Downloads\EMBRYO PROJECT\Dataset\ebimg"  # adjust if different
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


#Extract image features
X_image_features, image_ids = extract_image_features(image_files, resnet_model)
print("Image features shape:", X_image_features.shape)


#Create DataFrame for image features
# ✅ Use this instead
pca = PCA(n_components=50, random_state=42)
X_image_pca = pca.fit_transform(X_image_features)

image_feat_df = pd.DataFrame(X_image_pca, columns=[f"img_feat_{i}" for i in range(50)])
image_feat_df["Image_Name"] = [os.path.basename(x).lower() for x in image_ids]




#Clean metadata image column

df = df.rename(columns={"image_file": "Image_Name"})

df["Image_Name"] = df["Image_Name"].astype(str).apply(
    lambda x: os.path.basename(x) if x != "None" else None
)
df["Image_Name"] = df["Image_Name"].str.lower()


#Merge tabular metadata with image features
merged_df = df.merge(image_feat_df, on="Image_Name", how="inner")
print("Merged dataset shape:", merged_df.shape)

#to check they are merge or not
print(merged_df[['patient_id', 'Image_Name']])


# ===================================
# Step 7: Fit + Transform data
# ===================================
# Use merged_df including PCA image features
image_feature_cols = [c for c in merged_df if c.startswith("img_feat_")]
X = df[['endometrial_thickness_mm', 'cycle_day']]
y = df['ERA_status']


X_pre = preprocessor.fit_transform(X)
print("Transformed shape:", X_pre.shape)

#balancing the er_status
df["ERA_status"].value_counts(normalize=True) * 100
df["implantation_outcome"].value_counts(normalize=True) * 100


# Apply SMOTE oversampling once
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_pre, y)

print("Before balancing:\n", y.value_counts())
print("\nAfter balancing:\n", y_res.value_counts())

# Data Visualization PART
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# BEFORE balancing
sns.countplot(x=y, palette="pastel", ax=ax[0], order=y.value_counts().index)
ax[0].set_title("ERA_status Before Balancing", fontsize=14, fontweight="bold")
ax[0].set_ylabel("Count")
ax[0].set_xlabel("ERA Status")
for p in ax[0].patches:
    ax[0].annotate(f"{p.get_height()}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='bottom', fontsize=10, color="black")

# AFTER balancing
sns.countplot(x=y_res, palette="muted", ax=ax[1], order=y.value_counts().index)
ax[1].set_title("ERA_status After Balancing (SMOTE)", fontsize=14, fontweight="bold")
ax[1].set_ylabel("Count")
ax[1].set_xlabel("ERA Status")
for p in ax[1].patches:
    ax[1].annotate(f"{p.get_height()}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='bottom', fontsize=10, color="black")

plt.suptitle("Effect of SMOTE Balancing on ERA_status", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()



'''Step 11: Compare Numeric Features by ERA_status'''
# Plot distributions grouped by ERA_status
sns.set(style="whitegrid", palette="Set2")

# Number of columns for the grid
n_cols = 3
n_rows = math.ceil(len(numeric_cols) / n_cols)  # Calculate number of rows needed

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
axes = axes.flatten()  # Flatten in case of multiple rows/columns

for i, col in enumerate(numeric_cols):
    sns.boxplot(x="ERA_status", y=col, data=df, palette="Set2", ax=axes[i], showfliers=False)
    sns.stripplot(x="ERA_status", y=col, data=df, color="black", size=4, jitter=True, ax=axes[i])
    
    axes[i].set_title(f"{col} by ERA_status", fontsize=14, fontweight="bold")
    axes[i].set_xlabel("ERA Status", fontsize=12)
    axes[i].set_ylabel(col, fontsize=12)
    axes[i].tick_params(axis='x', labelsize=10)
    axes[i].tick_params(axis='y', labelsize=10)
    axes[i].grid(True, linestyle='--', alpha=0.7)

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Correlation heatmap (numeric features)
# ==============================
plt.figure(figsize=(8,6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()



# ERA_status vs Implantation Outcome Analysis
# ==============================
# Countplot: how implantation outcome differs by ERA_status
plt.figure(figsize=(6,4))
sns.countplot(x=implantation_col, hue=target_col, data=df, palette="Set2")
plt.title("Implantation Outcome by ERA_status")
plt.ylabel("Count")
plt.show()


# Numeric features by both ERA_status & implantation outcome
plt.figure(figsize=(18, 10))
for i, col in enumerate(numeric_cols):
    ax = plt.subplot(2, 3, i+1)
    
    # Boxplot without hue
    sns.boxplot(x=target_col, y=col, data=df, palette="Set2", showfliers=False, ax=ax)
    
    # Stripplot with hue, dodge to separate implantation outcomes
    sns.stripplot(x=target_col, y=col, hue=implantation_col, data=df,
                  dodge=True, color="black", size=4, jitter=0.25, alpha=0.7, ax=ax)
    
    ax.set_title(f"{col} by ERA_status & Implantation Outcome", fontsize=12, fontweight="bold")
    ax.set_xlabel("ERA Status", fontsize=10)
    ax.set_ylabel(col, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Only show legend on the first subplot to avoid duplicates
    if i == 0:
        ax.legend(title="Implantation Outcome", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.get_legend().remove()

plt.tight_layout()
plt.suptitle("Numeric Features by ERA_status and Implantation Outcome", fontsize=16, fontweight="bold", y=1.02)
plt.show()



#Distribution of numeric features
numeric_cols = ["age", "BMI", "endometrial_thickness_mm",
                "E2_pg_ml(Estradiol)", "P4_ng_ml(Progesterone)",
                "blastocoel_expansion"]

plt.figure(figsize=(15,5))
for i, col in enumerate(numeric_cols):
    plt.subplot(1, len(numeric_cols), i+1)
    sns.histplot(merged_df[col], kde=True, color='skyblue')
    plt.title(col)
plt.tight_layout()
plt.show()





#Image Feature Visualization
# Select columns that are image features (assuming feature columns start with numbers)
image_features_cols = [c for c in merged_df.columns if c.startswith("img_feat_")]

image_features = merged_df[image_features_cols].values

# PCA
pca = PCA(n_components=2)
image_pca = pca.fit_transform(image_features)

# Plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=image_pca[:,0], y=image_pca[:,1], hue=merged_df["ERA_status"], palette="Set1")
plt.title("PCA of Image Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()



# t-SNE transform
tsne = TSNE(n_components=2, random_state=42)
image_tsne = tsne.fit_transform(image_features)

# Plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=image_tsne[:,0], y=image_tsne[:,1], hue=merged_df["ERA_status"], palette="Set1")
plt.title("t-SNE of Image Features")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()



plt.figure(figsize=(8,6))
scatter = plt.scatter(
    image_pca[:,0], image_pca[:,1], 
    c=merged_df["age"], cmap="viridis", s=50
)
plt.colorbar(scatter, label="Age")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Image Features Colored by Age")
plt.show()


#visualize the relationship between age and ERA_status
plt.figure(figsize=(8,5))
sns.kdeplot(data=df, x="age", hue="ERA_status", fill=True, common_norm=False, alpha=0.4)
plt.title("Age Distribution Across ERA_status Groups", fontsize=14, fontweight="bold")
plt.xlabel("Age")
plt.ylabel("Density")
plt.show()


plt.figure(figsize=(6,4))
sns.swarmplot(x="ERA_status", y="age", data=df, palette="Set1", size=6)
plt.title("Age Distribution by ERA_status (Swarm Plot)", fontsize=14, fontweight="bold")
plt.xlabel("ERA_status")
plt.ylabel("Age")
plt.show()


sns.histplot(df, x="age", hue="ERA_status", multiple="stack",
             palette=["#4A90E2", "#50C878", "#A0A0A0"], bins=15)
plt.title("Age Distribution by ERA_status"); plt.xlabel("Age"); plt.ylabel("Count")
plt.show()






le = LabelEncoder()
y_encoded = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
# Preprocess training and test data
X_train_pre = preprocessor.fit_transform(X_train)
X_test_pre = preprocessor.transform(X_test)

# Random Forest with GridSearchCV
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 8]
}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train_pre, y_train)

# Best estimator evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_pre)

# Metrics output
print(f"Best Params: {grid_search.best_params_}")
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Visualization of Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for ERA_status Classification')
plt.show()


def get_patient_details(patient_id, merged_df):
    matched = merged_df[merged_df['patient_id'] == patient_id]
    if matched.empty:
        print(f"No data found for patient id: {patient_id}")
        return None
    else:
        # Exclude image feature columns before printing
        non_image_cols = [col for col in matched.columns if not col.startswith("img_feat_")]
        # Print only columns excluding image features
        for idx, row in matched.iterrows():
            print(f"Details for patient_id: {row['patient_id']}")
            for col in non_image_cols:
                print(f"{col}: {row[col]}")
            print("\n" + "-"*40 + "\n")

# Usage example
patient_id_input = "patient_id_0014"
get_patient_details(patient_id_input, merged_df)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))





cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_model,
    X=X,
    y=y_encoded,              # encoded labels
    cv=cv,
    scoring="f1_macro",       # or "accuracy"
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=-1
)

# Compute mean ± std
train_mean = np.mean(train_scores, axis=1)
train_std  = np.std(train_scores, axis=1)
val_mean   = np.mean(val_scores, axis=1)
val_std    = np.std(val_scores, axis=1)


plt.figure(figsize=(10, 7))
plt.plot(train_sizes, train_mean, marker='o', linestyle='-', color='blue', label="Train F1_macro")
plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.2, color='blue')

plt.plot(train_sizes, val_mean, marker='s', linestyle='-', color='green', label="CV F1_macro")
plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.2, color='green')

# Highlight best validation point in dark green
best_idx = np.argmax(val_mean)
plt.scatter(train_sizes[best_idx], val_mean[best_idx], color="darkgreen", marker="*", s=250, label="Best CV F1")

plt.xlabel("Training Samples", fontsize=13)
plt.ylabel("F1_macro", fontsize=13)
plt.title("Learning Curve for ERA_status Classification\n(Random Forest, Stratified 5-Fold CV)", fontsize=16)
plt.legend(fontsize=13)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()





# Assuming 'best_model' is your trained RandomForestClassifier
# and 'X_train' or 'X' contains your feature columns

# Get feature importances from the model
importances = best_model.feature_importances_

# If your features are from the preprocessed 'X' DataFrame or original feature names:
# List of feature names used in the model (adjust based on your features)
feature_names = X.columns.tolist()

# Create a sorted index to plot features in ascending order of importance
indices = np.argsort(importances)

plt.figure(figsize=(8, 6))
sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette="viridis")
plt.title("Feature Importance from Random Forest")
plt.xlabel("Relative Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()




# 3. Classification Report as Heatmap (Precision, Recall, F1)
report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report_dict).iloc[:-1, :-1].T  # exclude 'accuracy' row

plt.figure(figsize=(10, 5))
sns.heatmap(report_df, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Classification Report (Precision, Recall, F1-score)")
plt.show()



acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

plt.figure(figsize=(5,5))
sns.barplot(x=["Accuracy", "Macro F1"], y=[acc, macro_f1], palette="pastel")
plt.title("Overall Model Performance")
plt.ylabel("Score")
plt.ylim(0,1)
for i, v in enumerate([acc, macro_f1]):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)
plt.show()





















import os
import glob
import numpy as np
import pandas as pd
from skimage import io, color, filters, morphology, measure
import matplotlib.pyplot as plt

# Load the dataset with image_file and thickness
df = pd.read_csv(r"C:\Users\vamsi\Downloads\fg\dt_ERA_corrected - Copy.csv")

# Normalize the image file names to just basename lowercase for matching
df["Image_Name"] = df["image_file"].apply(lambda x: os.path.basename(str(x)).lower())

def load_rgb_image(image_path):
    """Load image and convert RGBA to RGB if needed."""
    img = io.imread(image_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = color.rgba2rgb(img)  # Converts to float [0,1]
        img = (img * 255).astype(np.uint8)  # Convert to uint8 [0,255]
    return img

def segment_and_lookup_thickness(image_path, dataset_df, pixel_to_mm_factor=0.0158, show_plot=True):
    """Segment endometrium image and lookup thickness from dataset."""
    img = load_rgb_image(image_path)
    if img is None:
        print(f"⚠️ Could not read {image_path}")
        return None

    gray = color.rgb2gray(img) if img.ndim == 3 else img
    thresh = filters.threshold_otsu(gray)
    binary = gray > thresh
    clean_mask = morphology.remove_small_objects(binary, min_size=200)
    clean_mask = morphology.remove_small_holes(clean_mask, area_threshold=300)
    labeled = measure.label(clean_mask)
    regions = measure.regionprops(labeled)

    if not regions:
        print(f"No endometrium found in {image_path}")
        return None

    region = max(regions, key=lambda r: r.area)
    coords = region.coords

    img_name = os.path.basename(image_path).lower()

    # Lookup thickness from dataset for this image name
    matched_row = dataset_df[dataset_df["Image_Name"] == img_name]
    if not matched_row.empty:
        thickness_mm_dataset = matched_row.iloc[0]["endometrial_thickness_mm"]
    else:
        thickness_mm_dataset = None

    if show_plot:
        plt.figure(figsize=(6,6))
        plt.imshow(gray, cmap='gray')
        plt.plot(coords[:, 1], coords[:, 0], 'b.', markersize=1)
        title = f"{img_name}"
        if thickness_mm_dataset is not None:
            title += f"\nEndometrial Thickness: {thickness_mm_dataset:.2f} mm"
        plt.title(title)
        plt.axis('off')
        plt.show()

    return {
        "Image_Name": img_name,
        "endometrial_thickness_mm": thickness_mm_dataset
    }

image_folder = r"C:\Users\vamsi\Downloads\EMBRYO PROJECT\Dataset\ebimg"
image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + \
              glob.glob(os.path.join(image_folder, "*.png")) + \
              glob.glob(os.path.join(image_folder, "*.jpeg"))

pixel_to_mm_factor = 0.0158

results = []
for image_path in image_paths:
    res = segment_and_lookup_thickness(image_path, df, pixel_to_mm_factor, show_plot=True)
    if res:
        results.append(res)

combined_df = pd.DataFrame(results)
#print(combined_df)




# Export all relevant dataframes to CSV for Power BI visualization

# 1. Export preprocessed merged dataset with image features and metadata
merged_df.to_csv(r"C:\Users\vamsi\Downloads\powerbi_merged_dataset.csv", index=False)

# 2. Export classification results (true vs predicted labels)
results_df = pd.DataFrame({
    "Patient_ID": X_test.index,
    "True_ERA_status": le.inverse_transform(y_test),
    "Predicted_ERA_status": le.inverse_transform(y_pred)
})
results_df.to_csv(r"C:\Users\vamsi\Downloads\powerbi_results.csv", index=False)

# 3. Export feature importance from the trained Random Forest
feat_imp = pd.DataFrame({
    "Feature": X.columns.tolist(),
    "Importance": importances
})
feat_imp.to_csv(r"C:\Users\vamsi\Downloads\powerbi_feature_importance.csv", index=False)

# 4. Export classification report metrics (precision, recall, f1 per class)
report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "Class"})
report_df.to_csv(r"C:\Users\vamsi\Downloads\powerbi_classification_report.csv", index=False)

# 5. Export learning curve data (training sizes, mean and std F1 scores)
lc_df = pd.DataFrame({
    "Training_Size": train_sizes,
    "Train_F1_mean": train_mean,
    "Train_F1_std": train_std,
    "Val_F1_mean": val_mean,
    "Val_F1_std": val_std
})
lc_df.to_csv(r"C:\Users\vamsi\Downloads\powerbi_learning_curve.csv", index=False)


print("✅ Exported datasets for Power BI successfully.")









