##############################
# Telco Customer Churn Feature Engineering
##############################

# https://www.kaggle.com/datasets/blastchar/telco-customer-churn?datasetId=13996&sortBy=voteCount

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings

warnings.simplefilter(action="ignore")

# Set pandas display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load the Telco Customer Churn dataset
df = pd.read_csv("datasets/Telco-Customer-Churn.csv")

def inspect_data(df):
    """
    Prints the first few rows, shape, and basic info of the dataframe.
    
    Parameters:
    df (pd.DataFrame): The dataframe to be inspected.
    """
    print(df.head())
    print(f"Shape: {df.shape}")
    print(df.info())

# Inspect the dataset
inspect_data(df)

# Convert 'TotalCharges' column to numeric format
# Some values might be non-numeric, so they will be coerced into NaN

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Encode 'Churn' column as a binary variable (1 for Yes, 0 for No)
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

##################################
# Explarotory  Data Analysis
##################################


def check_df(dataframe, head=5):
    """
    Provides a summary of a given Pandas DataFrame by displaying its shape, data types, head, tail, 
    missing values, and quantiles.
    
    Parameters:
    dataframe (pd.DataFrame): The DataFrame to be analyzed.
    head (int, optional): The number of rows to display for head and tail. Default is 5.
    
    Returns:
    None: Prints various descriptive statistics about the DataFrame.
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


##################################
#Identifying Numerical and Categorical Variables
###################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Identifies categorical, numerical, and categorical but cardinal variables in the given DataFrame.
    Note: Categorical variables include numerically represented categorical variables.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame for which variable names are to be identified.
    cat_th : int, optional (default=10)
        Threshold value for numerical but categorical variables.
    car_th : int, optional (default=20)
        Threshold value for categorical but cardinal variables.

    Returns
    -------
    cat_cols : list
        List of categorical variables.
    num_cols : list
        List of numerical variables.
    cat_but_car : list
        List of categorical but cardinal variables.

    Examples
    --------
    import seaborn as sns
    df = sns.load_dataset("iris")
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    Notes
    -----
    cat_cols + num_cols + cat_but_car = total number of variables in the dataset.
    num_but_cat variables are included in cat_cols.
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car



##################################
# Numerical variable analysis for the target variable
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    """
    Calculates and prints the mean of a numerical variable grouped by a target categorical variable.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset containing the variables.
    target : str
        The name of the categorical target variable.
    numerical_col : str
        The name of the numerical variable to be analyzed.
    
    Returns
    -------
    None: Prints the mean values of the numerical variable grouped by the target variable.
    
    Example
    -------
    target_summary_with_num(df, "Churn", "MonthlyCharges")
    """
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)


##################################
# Categorical variable analysis for the target variable
##################################


def target_summary_with_cat(dataframe, target, categorical_col):
    """
    Calculates and prints the mean of the target variable for each category of a given categorical variable,
    along with the count and percentage ratio of each category.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset containing the variables.
    target : str
        The name of the target variable.
    categorical_col : str
        The name of the categorical variable to be analyzed.
    
    Returns
    -------
    None: Prints a DataFrame containing the mean target value, count, and ratio of each category.
    
    Example
    -------
    target_summary_with_cat(df, "Churn", "Contract")
    """
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)



##################################
# Correlation
##################################

df[num_cols].corr()

# Correlation Matrix (Heat Map)
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

# It seems that TOTALCHARGERS are highly correlated with monthly fees and tenures

df.corrwith(df["Churn"]).sort_values(ascending=False)

##################################
# GÖREV 2: FEATURE ENGINEERING
##################################

##################################
# Missing Values
##################################

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    """
    Identifies missing values in the dataset and provides a summary table with the count and percentage 
    of missing values for each column.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset to analyze for missing values.
    na_name : bool, optional (default=False)
        If True, returns the list of columns with missing values.
    
    Returns
    -------
    list (optional)
        If na_name is True, returns a list of column names with missing values.
    
    Example
    -------
    na_columns = missing_values_table(df, na_name=True)
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

df[df["TotalCharges"].isnull()]["tenure"]
df["TotalCharges"].fillna(0, inplace=True)

df.isnull().sum()



##################################
# BASE MODEL
##################################

# Copying the original DataFrame
dff = df.copy()

# Removing the target variable from categorical columns
cat_cols = [col for col in cat_cols if col not in ["Churn"]]

# One-Hot Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """
    Applies one-hot encoding to categorical variables in the given DataFrame.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset containing categorical variables.
    categorical_cols : list
        List of categorical column names to be encoded.
    drop_first : bool, optional (default=False)
        Whether to drop the first level of encoded columns to avoid multicollinearity.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one-hot encoded categorical variables.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

dff = one_hot_encoder(dff, cat_cols, drop_first=True)

# Splitting features and target
y = dff["Churn"]
X = dff.drop(["Churn", "customerID"], axis=1)

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

# Training CatBoost Classifier
catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

# Model Evaluation
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred, y_test),4)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 4)}")
print(f"F1: {round(f1_score(y_pred,y_test), 4)}")
print(f"AUC: {round(roc_auc_score(y_pred,y_test), 4)}")

# Sample Output:
# Accuracy: 0.7837
# Recall: 0.6333
# Precision: 0.4843
# F1: 0.5489
# AUC: 0.7282


##################################
# OUTLIER ANALYSIS
##################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Calculates the lower and upper limits for detecting outliers in a given numerical column using 
    interquartile range (IQR) method.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset containing the numerical variable.
    col_name : str
        The name of the numerical column to compute thresholds for.
    q1 : float, optional (default=0.05)
        The first quantile threshold.
    q3 : float, optional (default=0.95)
        The third quantile threshold.
    
    Returns
    -------
    tuple
        Lower and upper limits for outlier detection.
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    """
    Checks if a given numerical column contains outliers based on calculated thresholds.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset containing the numerical variable.
    col_name : str
        The name of the numerical column to check for outliers.
    
    Returns
    -------
    bool
        True if outliers are present, False otherwise.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    """
    Replaces outliers in a numerical column with the calculated lower and upper limits.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset containing the numerical variable.
    variable : str
        The name of the numerical column to handle outliers.
    q1 : float, optional (default=0.05)
        The first quantile threshold.
    q3 : float, optional (default=0.95)
        The third quantile threshold.
    
    Returns
    -------
    None: Modifies the DataFrame in place.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=q1, q3=q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Checking and replacing outliers
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)





##################################
# Feature Exraction
##################################


# Creating a categorical variable based on tenure

df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"

# Creating new feature: Engaged customers

df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# Creating new feature: Lack of protection services

df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Identifying young customers who are not engaged

df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)

# Counting the total number of services used

df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

# Flagging customers who use any streaming service

df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Flagging customers who use automatic payment methods

df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# Calculating average charges per tenure

df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Calculating increase ratio in charges

df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Calculating average service fee per service used

df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)

# Displaying dataset details
df.head()
df.shape


##################################
# ENCODING
##################################

# Identifying categorical and numerical columns
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    """
    Applies Label Encoding to binary categorical columns.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset containing categorical variables.
    binary_col : str
        The name of the binary categorical column to be encoded.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with the binary column label-encoded.
    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# Identifying binary categorical columns
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

# Applying Label Encoding
for col in binary_cols:
    df = label_encoder(df, col)

# One-Hot Encoding
# Updating categorical columns
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """
    Applies One-Hot Encoding to categorical variables.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset containing categorical variables.
    categorical_cols : list
        List of categorical column names to be encoded.
    drop_first : bool, optional (default=False)
        Whether to drop the first level of encoded columns to avoid multicollinearity.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one-hot encoded categorical variables.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# Applying One-Hot Encoding
df = one_hot_encoder(df, cat_cols, drop_first=True)

# Displaying encoded dataset
df.head()


##################################
# MODELING
##################################

# Defining target and features
y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

# Training CatBoost Classifier
catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

# Model Evaluation
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),2)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"AUC: {round(roc_auc_score(y_pred,y_test), 2)}")

# Final Model Results
# Accuracy: 0.8
# Recall: 0.66
# Precision: 0.51
# F1: 0.58
# AUC: 0.75

# Base Model Results
# Accuracy: 0.7837
# Recall: 0.6333
# Precision: 0.4843
# F1: 0.5489
# AUC: 0.7282


