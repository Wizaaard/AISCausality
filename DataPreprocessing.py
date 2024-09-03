import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Function to normalize column names
def normalize_columns(df):
    df.columns = df.columns.str.replace(r'@Pre-OP', '', regex=True)
    df.columns = df.columns.str.replace(r'@6-month-PO', '', regex=True)
    return df


def loadData(path):
	df = pd.read_csv(path)
	data = df[df['SRS-22\\21. (SRS)@6-month-PO'] >= 3.4]
	target = data[['SRS-22\\21. (SRS)@6-month-PO']]
	treatment_df = data.filter(like='@6-month')
	control_df = data.filter(like='@Pre-OP')

	columns_to_remove = [
		'SRS Normalized Scores\\General Function (SRS 22)@6-month-PO',
		'SRS Normalized Scores\\Mental Health (SRS 22)@6-month-PO',
		'SRS Normalized Scores\\Pain (SRS 22)@6-month-PO',
		'SRS Normalized Scores\\Satisfaction (SRS 22)@6-month-PO',
		'SRS Normalized Scores\\Self-Image (SRS 22)@6-month-PO',
		'SRS Normalized Scores\\Total (SRS 22)@6-month-PO',
		'SRS-22\\1. (SRS)@6-month-PO',
		'SRS-22\\10. (SRS)@6-month-PO',
		'SRS-22\\11. (SRS)@6-month-PO',
		'SRS-22\\12. (SRS)@6-month-PO',
		'SRS-22\\13. (SRS)@6-month-PO',
		'SRS-22\\14. (SRS)@6-month-PO',
		'SRS-22\\15. (SRS)@6-month-PO',
		'SRS-22\\16. (SRS)@6-month-PO',
		'SRS-22\\17. (SRS)@6-month-PO',
		'SRS-22\\18. (SRS)@6-month-PO',
		'SRS-22\\19. (SRS)@6-month-PO',
		'SRS-22\\2. (SRS)@6-month-PO',
		'SRS-22\\20. (SRS)@6-month-PO',
		'SRS-22\\22. (SRS)@6-month-PO',
		'SRS-22\\3. (SRS)@6-month-PO',
		'SRS-22\\4. (SRS)@6-month-PO',
		'SRS-22\\5. (SRS)@6-month-PO',
		'SRS-22\\6. (SRS)@6-month-PO',
		'SRS-22\\7. (SRS)@6-month-PO',
		'SRS-22\\8. (SRS)@6-month-PO',
		'SRS-22\\9. (SRS)@6-month-PO'
	]

	columns_to_r = ['SRS Normalized Scores\\General Function (SRS 22)@Pre-OP',
	'SRS Normalized Scores\\Mental Health (SRS 22)@Pre-OP',
	'SRS Normalized Scores\\Pain (SRS 22)@Pre-OP',
	'SRS Normalized Scores\\Satisfaction (SRS 22)@Pre-OP',
	'SRS Normalized Scores\\Self-Image (SRS 22)@Pre-OP',
	'SRS Normalized Scores\\Total (SRS 22)@Pre-OP',
	'SRS-22\\1. (SRS)@Pre-OP',
	'SRS-22\\10. (SRS)@Pre-OP',
	'SRS-22\\11. (SRS)@Pre-OP',
	'SRS-22\\12. (SRS)@Pre-OP',
	'SRS-22\\13. (SRS)@Pre-OP',
	'SRS-22\\14. (SRS)@Pre-OP',
	'SRS-22\\15. (SRS)@Pre-OP',
	'SRS-22\\16. (SRS)@Pre-OP',
	'SRS-22\\17. (SRS)@Pre-OP',
	'SRS-22\\18. (SRS)@Pre-OP',
	'SRS-22\\19. (SRS)@Pre-OP',
	'SRS-22\\2. (SRS)@Pre-OP',
	'SRS-22\\20. (SRS)@Pre-OP',
	'SRS-22\\21. (SRS)@Pre-OP',
	'SRS-22\\22. (SRS)@Pre-OP',
	'SRS-22\\3. (SRS)@Pre-OP',
	'SRS-22\\4. (SRS)@Pre-OP',
	'SRS-22\\5. (SRS)@Pre-OP',
	'SRS-22\\6. (SRS)@Pre-OP',
	'SRS-22\\7. (SRS)@Pre-OP',
	'SRS-22\\8. (SRS)@Pre-OP',
	'SRS-22\\9. (SRS)@Pre-OP']
	treatment_df_ = treatment_df.drop(columns=columns_to_remove)
	control_df_ = control_df.drop(columns=columns_to_r)
      
	df1_normalized = normalize_columns(control_df_.copy())
	df2_normalized = normalize_columns(treatment_df_.copy())


	# Create a mapping from normalized names to original names
	df1_column_mapping = dict(zip(df1_normalized, control_df_.columns))
	df2_column_mapping = dict(zip(df2_normalized, treatment_df_.columns))
      
	# Find common columns
	common_columns = df1_normalized.columns.intersection(df2_normalized.columns)

	# Map back to the original columns
	common_columns_df1 = [df1_column_mapping[col] for col in common_columns]
	common_columns_df2 = [df2_column_mapping[col] for col in common_columns]

	# Filter both DataFrames to keep only common columns
	df1_filtered = control_df_[common_columns_df1]
	df2_filtered = treatment_df_[common_columns_df2]
      
	# List of columns to select
	columns_to_select = [
		'Race_White', 'Race_Black', 'Race_Other', 'Race_Hispanic',
		'Race_Asian', 'Gender', 'Age', 'Comorbidity'
	]

	dem = data[columns_to_select]
      
	# Merge the DataFrames along columns
	control = pd.concat([dem, df1_filtered ], axis=1)
	treatment = pd.concat([dem, df2_filtered], axis=1)
      
	# Combine DataFrames
	# Split the combined DataFrame
	train_combined, test_combined = train_test_split(data, test_size=0.3, random_state=35)

	y_train = train_combined[target.columns]

	# Remove the 'ID' column
	train_combined = train_combined.drop(columns=['PID'])
      
	# Apply SMOTE to balance the classes
	smote = SMOTE(random_state=35)
	X_train_smote, y_train_smote = smote.fit_resample(train_combined, np.array(y_train.astype(int)).ravel())
      
	print(len(X_train_smote), len(y_train_smote), len(control.columns), len(treatment.columns))

	# Separate the combined DataFrames back into original forms
	control_train = X_train_smote[control.columns]
	treatment_train = X_train_smote[treatment.columns]

	control_test = test_combined[control.columns]
	treatment_test = test_combined[treatment.columns]
	y_test = test_combined[target.columns]

	# Map target values to binary [0, 1] after splitting
	# y_train = y_train['SRS-22\\21. (SRS)@6-month-PO'].map({4: 0, 5: 1})
	y_train_smote = pd.Series(y_train_smote)
	y_train_smote = y_train_smote.map({4: 0, 5: 1})
	y_test = pd.Series(np.array(y_test.astype(int)).ravel())

	y_test = y_test.map({4: 0, 5: 1})

	return control_train, treatment_train, control_test, treatment_test, y_train_smote, y_test

# Based on how causualML calculates binary propensity it only calculates
# the probability the data point is in the treatment group not control
def process_binary_propensity(propensity):
  propensity[0] = 1 - propensity[1.0]
  return propensity

