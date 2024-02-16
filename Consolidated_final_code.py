# Import required functions from packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import seaborn as sb
from statsmodels.stats.outliers_influence import variance_inflation_factor
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV
import xgboost 
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

dataraw_n = pd.read_csv("players_22.csv")
dataraw_n.head()

################################ Data Preprocessing ################################

### Drop columns that have no significance with index numbers 
columns_to_drop = [0,1,2,10,11,13,15,40,41,42,43,44]
df_n = dataraw_n.drop(dataraw_n.columns[columns_to_drop], axis=1, inplace=False)
df_ncopy= df_n.copy()
df_ncopy.shape

# Value ditribution before cleaning
plt.figure(1, figsize=(18, 7))
sb.set(style="whitegrid")
sb.countplot( x= 'value_eur', data=df_ncopy)
plt.title('Value distribution of all players')
plt.show()

### Removing Null Values
df_n.isnull().sum()
df_null_removed = df_n.dropna()
df_null_removed.shape

# Fix Categorical Col work_rate
column_name = "work_rate"
print(df_null_removed["work_rate"].unique())
string_to_float_mapping = {'High/Medium': 2.5, 'Medium/Medium': 2, 'Medium/High': 2.5,'High/High': 3, 'High/Low': 2, 'Medium/Low': 1.5,'Low/High': 2, 'Low/Medium': 1.5, 'Low/Low':1}

df_null_removed.loc[df_null_removed[column_name] == 'High/Medium', column_name] = string_to_float_mapping['High/Medium']
df_null_removed.loc[df_null_removed[column_name] == 'Medium/Medium', column_name] =string_to_float_mapping['Medium/Medium']
df_null_removed.loc[df_null_removed[column_name] == 'Medium/High', column_name] = string_to_float_mapping['Medium/High']
df_null_removed.loc[df_null_removed[column_name] == 'High/High', column_name] = string_to_float_mapping['High/High']
df_null_removed.loc[df_null_removed[column_name] == 'High/Low', column_name] = string_to_float_mapping['High/Low']
df_null_removed.loc[df_null_removed[column_name] == 'Medium/Low', column_name] = string_to_float_mapping['Medium/Low']
df_null_removed.loc[df_null_removed[column_name] == 'Low/Medium', column_name] = string_to_float_mapping['Low/Medium']
df_null_removed.loc[df_null_removed[column_name] == 'Low/High', column_name] = string_to_float_mapping['Low/High']
df_null_removed.loc[df_null_removed[column_name] == 'Low/Low', column_name] = string_to_float_mapping['Low/Low']


df_null_removed[column_name] = df_null_removed[column_name].astype(float)
df_null_removed.isnull().sum()

#### Removing Outliers

# Calculate z-scores for each column
z_scores = (df_null_removed - df_null_removed.mean()) / df_null_removed.std()

# Set a z-score threshold (e.g., 3 or -3)
z_threshold = 3

# Remove rows with z-scores greater than the threshold
df_no_outliers = df_null_removed[(z_scores < z_threshold).all(axis=1)]
###

print(df_no_outliers.shape)

# fixing contract years column
print(df_no_outliers["club_contract_valid_until"].unique())
year = 2022
df_no_outliers['club_contract_valid_years'] = df_no_outliers['club_contract_valid_until'] - year 
# Drop multiple columns
columns_to_drop = ['league_level']
df_no_outliers = df_no_outliers.drop(columns_to_drop, axis=1)
columns_to_drop = ['club_contract_valid_until']
df_no_outliers = df_no_outliers.drop(columns_to_drop, axis=1)

# Re-arranging cols-target variable as first column followed by features
dep_var_col = 'value_eur'
other_cols = [col for col in df_no_outliers.columns if col!=dep_var_col]
desired_order = [dep_var_col] + other_cols
df_ordered_cleaned = df_no_outliers[desired_order]
df_ordered_cleaned.shape
###

# distribution of value_eur
# Value ditribution
plt.figure(1, figsize=(50, 25))
sb.set(style="whitegrid")
sb.countplot( x= 'value_eur', data=df_ordered_cleaned)
plt.title('Value distribution of all players')
plt.show()

#-------------------------------------------------------------------------------

### Stepwise Regression
# Separate features and target variable
X_stpreg = df_ordered_cleaned.drop(columns=["value_eur"])
y_stpreg = df_ordered_cleaned["value_eur"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_stpreg, y_stpreg, test_size=0.2, random_state=52)

# Perform stepwise feature selection using statsmodels
def stepwise_selection(X_stpreg, y_stpreg, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        excluded = list(set(X_stpreg.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y_stpreg, sm.add_constant(pd.DataFrame(X_stpreg[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        model = sm.OLS(y_stpreg, sm.add_constant(pd.DataFrame(X_stpreg[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

selected_features = stepwise_selection(X_train, y_train)

# dependent variable and features split
# Select all columns except the first one
X_features = df_ordered_cleaned.iloc[:, 1:]
Y_target = df_ordered_cleaned.iloc[:,0]

# Normaliziation of the target variable
# Reshape the Series to a 2D array
Y_target_2D = Y_target.values.reshape(-1, 1)

# Fit and transform the data using the scaler
Y_target_norm = scaler.fit_transform(Y_target_2D)

# Convert the normalized data back to a pandas Series
Y_target_norm = pd.Series(Y_target_norm.flatten())
###

# Stepwise Regression was conducted for feature selection in a separate file
# | and the results for selected columns obtained were implemented here 
# dependent variable and features split
# Select specific columns by their names
X_selected_features = df_ordered_cleaned[['overall','wage_eur','release_clause_eur','international_reputation','power_stamina','potential','age','club_contract_valid_years','defending','power_stamina','mentality_vision','weight_kg','movement_balance','skill_dribbling','movement_sprint_speed','physic']]
# Y_target = df_ordered_cleaned.iloc[:,0]
# Normalization of the features


# Create a MinMaxScaler object
# scaler = MinMaxScaler()
# Fit and transform the data to perform Min-Max Scaling
X_selected_features_norm = scaler.fit_transform(X_selected_features)


############## Experiment with random split and keeping a set of data unseen ############

#Create a split of the df_ordered_cleaned of seen and unseen to test with model
from sklearn.utils import shuffle
shuffled_data = shuffle(df_ordered_cleaned, random_state=12)  # You can provide any random state value
# distribution plot for value

# Split the dataset into two parts (80% and 20%)
df_part1, df_part2 = train_test_split(shuffled_data, test_size=0.2, random_state=2)

# SEEN
X_seen = df_part1.iloc[:, 1:]
Y_seen = df_part1.iloc[:,0]
# selected features
X_seen = X_seen[['overall','wage_eur','release_clause_eur','international_reputation','power_stamina','potential','age','club_contract_valid_years','defending','power_jumping','mentality_vision','weight_kg','movement_balance','skill_dribbling','movement_sprint_speed','physic']]

# UNSEEN
X_unseen = df_part2.iloc[:, 1:]
Y_unseen = df_part2.iloc[:,0]
# selected features
X_unseen = X_unseen[['overall','wage_eur','release_clause_eur','international_reputation','power_stamina','potential','age','club_contract_valid_years','defending','power_jumping','mentality_vision','weight_kg','movement_balance','skill_dribbling','movement_sprint_speed','physic']]

#######

### Normalization of the features
# Create a MinMaxScaler object
scaler = MinMaxScaler()
# Fit and transform the data to perform Min-Max Scaling
X_seen_norm = scaler.fit_transform(X_seen)
X_unseen_norm = scaler.fit_transform(X_unseen)

# Normaliziation of the target variable
# Reshape the Series to a 2D array
Y_seen_2D = Y_seen.values.reshape(-1, 1)
Y_unseen_2D = Y_unseen.values.reshape(-1, 1)

# Fit and transform the data using the scaler
Y_seen_norm = scaler.fit_transform(Y_seen_2D)
Y_unseen_norm = scaler.fit_transform(Y_unseen_2D)

# Convert the normalized data back to a pandas Series
Y_seen_norm = pd.Series(Y_seen_norm.flatten())
Y_unseen_norm = pd.Series(Y_unseen_norm.flatten())

#---------------------------------------------------------------------------------------

################ Algorithm Implementation ##################

### XgBoost ###

# Hyperparameter tuning

# Define Parameter Grid
param_grid = {
    'n_estimators': [100, 200, 300], #300
    'learning_rate': [0.01, 0.05, 0.1, 0.2], #0.01
    'max_depth': [3, 4, 5], #3
    'min_child_weight': [1, 2, 3], #1
    'gamma': [0.5,0.7] #0.5
    # Add more hyperparameters as needed
} 

#model_xgb = XGBRegressor(n_estimators=300, learning_rate=0.2, max_depth=4, min_child_weight=3)
model_xgb = XGBRegressor()

# grid search
grid_search = GridSearchCV(estimator=XGBRegressor(),
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           cv=5,  # Number of folds in cross-validation
                           verbose=2,
                           n_jobs=-1)  # Number of CPU cores to use (-1 uses all cores)


# Define number of folds for cross-validation
num_folds = 5

# Initialize KFold with the desired number of folds
kf = KFold(n_splits=num_folds, shuffle=True, random_state=32)

# Initialize lists to store evaluation results
#accuracy_scores = []
mse_scores = []
r2_scores = []

#df_X = df_X.values
# Perform k-fold cross-validation


for train_index, test_index in kf.split(X_selected_features_norm):
    X_train, X_test = X_selected_features_norm[train_index], X_selected_features_norm[test_index]
    y_train, y_test = Y_target_norm[train_index], Y_target_norm[test_index]

    # Fit the grid search to the log-transformed data
    grid_search.fit(X_train, y_train)

    # Get the best SVR model with tuned hyperparameters
    best_model_svr = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_model_svr.predict(X_test)

    # Evaluate the model
    #accuracy = accuracy_score(y_test, y_pred)  # Note: Accuracy might not be suitable for regression tasks
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    #accuracy_scores.append(accuracy)
    mse_scores.append(mse)
    r2_scores.append(r2)

# Now you can analyze the lists of scores as needed
# Print the hyperparameters of the best model
print("Best Model Parameters:", best_model_svr.get_params())
# Calculate average scores
#avg_accuracy = np.mean(accuracy_scores)
avg_mse = np.mean(mse_scores)
avg_r2 = np.mean(r2_scores)

#print("Average Accuracy:", avg_accuracy)
print("Average Mean Squared Error:", round(avg_mse,5))
print("R2 Error:", round(avg_r2,5))

### SVR ###
# Define the number of folds for k-fold cross-validation
n_splits = 5

# Initialize lists to store evaluation metrics
rmse_scores = []
r2_scores = []
# Initialize lists to store evaluation metrics
rmse_scores_train = []
r2_scores_train = []

# Create k-fold cross-validation iterator
kf = KFold(n_splits=n_splits, shuffle=True, random_state=22)

# Define hyperparameters and their potential values for tuning
param_grid = {
    #'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10], # 1
    'epsilon': [0.01, 0.1, 1], # 0.1
    #'gamma': ['scale', 'auto']
    # Add more hyperparameters to tune
}

# Create an instance of the SVR model
model_svr = SVR()


# Create GridSearchCV object
grid_search = GridSearchCV(estimator=model_svr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=n_splits)

# Split the data into training and test sets (80% training, 20% test)

# Iterate over the folds
for train_index, test_index in kf.split(X_seen_norm):
    X_train, X_test = X_seen_norm[train_index], X_seen_norm[test_index]
    y_train, y_test = Y_seen_norm[train_index], Y_seen_norm[test_index]

    # Fit the grid search to the log-transformed data
    grid_search.fit(X_train, y_train)

    # Get the best SVR model with tuned hyperparameters
    best_model_svr = grid_search.best_estimator_

    # Make predictions on the train set
    y_train_pred = best_model_svr.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    rmse_scores_train.append(train_rmse)
    r2_scores_train.append(train_r2)


    # Make predictions on the test set
    y_pred = best_model_svr.predict(X_test) 
    # Evaluate model performance
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    rmse_scores.append(rmse)
    r2_scores.append(r2)

# Print the hyperparameters of the best model
print("Best Model Parameters:", best_model_svr.get_params())
# Calculate average metrics
average_rmse_train = np.mean(rmse_scores_train)
average_r2_train = np.mean(r2_scores_train)

print("Average Root Mean Squared Error TRAIN SET:", average_rmse_train)
print("Average R-squared TRAIN SET:", average_r2_train)


average_rmse = np.mean(rmse_scores)
average_r2 = np.mean(r2_scores)

print("Average Root Mean Squared Error TEST SET:", average_rmse)
print("Average R-squared TEST SET:", average_r2)



# Make predictions on the test set
y_pred_outer = best_model_svr.predict(X_unseen_norm) 
# Evaluate model performance
rmse_outer = np.sqrt(mean_squared_error(Y_unseen_norm, y_pred_outer))
r2_outer = r2_score(Y_unseen_norm, y_pred_outer)

print("Root Mean Squared Error UNSEEN SET:", rmse_outer)
print("R-squared UNSEEN SET:", r2_outer)


### ANN ###
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)


rmse_scores = []
r2_scores = []

train_losses = []
val_losses = []
for fold, (train_index, test_index) in enumerate(kf.split(X_selected_features_norm)):
    X_train, X_test = X_selected_features_norm[train_index], X_selected_features_norm[test_index]
    y_train, y_test = Y_target_norm[train_index], Y_target_norm[test_index]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(X_selected_features_norm.shape[1],)),
        tf.keras.layers.Dropout(0.5),  # Adding dropout
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print(f"Fold {fold+1}/{num_folds}")
    
    for epoch in range(25):
        history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0, validation_split=0.2)
        train_loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Epoch {epoch + 1}:")
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    
    rmse_scores.append(rmse)
    r2_scores.append(r2)

average_rmse = np.mean(rmse_scores)
print("Average RMSE: {:.4f}".format(average_rmse))

average_r2 = np.mean(r2_scores)
print("Average R2: {:.4f}".format(average_r2))

# Increase figure size and resolution
plt.figure(figsize=(10, 6), dpi=100)

# Plot training loss and validation loss
plt.plot(range(1, 126), train_losses, label='Train Loss')
plt.plot(range(1, 126), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)  # Add grid lines for clarity
plt.tight_layout()  # Adjust spacing for better layout
plt.show()
