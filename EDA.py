# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

from sklearn import gaussian_process as gp
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import seaborn as sns
import matplotlib.pyplot as plt

#outlier analysis
df = pd.read_csv('covidcast_new.csv')
df = df.drop('geo_value', axis = 'columns')
df = df.drop('time_value', axis = 'columns')
means = df.mean()
sigmas = df.std()

drop_dict = dict()

for col in df.columns:
    mu = means[col]
    sigma = sigmas[col]
    #print(col)
    #print(str(df[(df[col] > mu + 3*sigma)].shape[0]) + '/' + str(df.shape[0]) + '=' + str(df[(df[col] > mu + 3*sigma)].shape[0]/df.shape[0]) + '\n')
    #print(str(df[(df[col] < mu - 3*sigma)].shape[0]) + '/' + str(df.shape[0]) + '=' + str(df[(df[col] < mu - 3*sigma)].shape[0]/df.shape[0]) + '\n\n')
    drops = df[(df[col] > mu + 3*sigma)].index.tolist()
    drops.extend(df[(df[col] < mu - 3*sigma)].index.tolist())

    drop_dict[col] = drops

outlier_perc = [len(x)/df.shape[0] for x in drop_dict.values()]
plt.bar(drop_dict.keys(),outlier_perc)
plt.xticks(rotation=45, ha='right')
plt.title('Percentage of Outliers by Variable')
plt.show()

df = pd.read_csv('covidcast_new.csv')
for col in df.columns:
    if col in drop_dict.keys():
        df.drop(drop_dict[col])


# population density
fips_map = pd.read_csv('US_FIPS_Codes.csv')
fips_dictionary = dict()

for i in range(len(fips_map)):

    fips_st = str(fips_map.loc[i,'FIPS State'])
    fips_cty = str(fips_map.loc[i, 'FIPS County'])

    while(len(fips_cty) < 3):
        fips_cty = '0' + fips_cty

    fips_code = fips_st+fips_cty

    state = fips_map.loc[i,'State']
    county = fips_map.loc[i,'County Name']

    loc = [county, state]
    
    fips_dictionary[fips_code] = loc

for i in range(1,57):
    j = str(i)
    if((j+'013') in fips_dictionary.keys()):
        count += 1
        fips_dictionary[j + '000'] = ['MEGA', fips_dictionary[j+'013'][1]]
    elif((j+'003') in fips_dictionary.keys()):
        count += 1
        fips_dictionary[j + '000'] = ['MEGA', fips_dictionary[j+'003'][1]]

def get_st_from_fips(fips_code):
    fips_string = str(fips_code)
    return fips_dictionary[fips_string][1]

def get_county_from_fips(fips_code):
    fips_string = str(fips_code)
    return fips_dictionary[fips_string][0]

df['county'] = df['geo_value'].apply(get_county_from_fips)
df['state'] = df['geo_value'].apply(get_st_from_fips)

## defining population density per state

pop_state = pd.read_csv('NST-EST2023-POP.csv')

area_state = pd.read_csv('state_area.csv')

state_data = pd.DataFrame(pop_state).dropna()

state_data['area'] = area_state['Sq. Mi.'].apply(lambda x: str.replace(x,',','')).astype(float)
state_data['population'] = state_data['population'].apply(lambda x: str.replace(x,',','')).astype(float)

state_data['pop_den'] = state_data['population']/state_data['area']

pop_den_dict = dict()

for i in range(len(state_data)):
    pop_den_dict[state_data.loc[i,'state']] = state_data.loc[i, 'pop_den']


def get_state_pop_den(state):
    return pop_den_dict[state]

def get_days_since_new_year(date_string):
    date = dt.datetime.strptime(date_string, '%m/%d/%Y')
    nyd = dt.datetime(2021, 1, 1,0,0)
    delta = date - nyd
    return delta.days


df['pop_density'] = df['state'].apply(get_state_pop_den)

df['days'] = df['time_value'].apply(get_days_since_new_year)

df = df.drop('time_value', axis = 'columns')
df = df.drop('geo_value', axis = 'columns')
df = df.drop('county', axis = 'columns')
df = df.drop('state', axis = 'columns')


# Compute the correlation matrix
corr_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

# Extract correlations with the target variable
target_corr = (corr_matrix["smoothed_wtested_positive_14d"]).abs().sort_values(ascending=False)
print("\nCorrelation with target variable (smoothed_wtested_positive_14d):")
print(target_corr)

target_corr = (corr_matrix["smoothed_wcovid_vaccinated"]).abs().sort_values(ascending=False)
print("\nCorrelation with target variable (smoothed_wcovid_vaccinated):")
print(target_corr)

#splitting the datasets

#smoothed_wtested_positive_14d
df_tp = df.drop('smoothed_wcovid_vaccinated', axis = 'columns')
df_tp = df_tp.drop('smoothed_wvaccine_likely_govt_health', axis = 'columns')
df_tp = df_tp.drop('smoothed_wwearing_mask', axis = 'columns')


#smoothed_wcovid_vaccinated
df_cv = df.drop('smoothed_wtested_positive_14d', axis = 'columns')
df_cv = df_cv.drop('smoothed_wvaccine_likely_govt_health', axis = 'columns')
df_cv = df_cv.drop('smoothed_wothers_masked', axis = 'columns')


# analysis of NAs

isNA_dict_tp = dict()

for col in df_tp.columns:
    nas = df_tp[df_tp[col].isna()].index.tolist()
    isNA_dict_tp[col] = nas

nas_perc = [len(x)/df_tp_nd.shape[0] for x in isNA_dict_tp.values()]
plt.bar(isNA_dict_tp.keys(),nas_perc)
plt.xticks(rotation=45, ha='right')
plt.title('Percentage of NAs by Variable')
plt.show()


isNA_dict_cv = dict()

for col in df_cv.columns:
    nas = df_cv[df_cv[col].isna()].index.tolist()
    isNA_dict_cv[col] = nas

nas_perc = [len(x)/df_cv.shape[0] for x in isNA_dict_cv.values()]
plt.bar(isNA_dict_cv.keys(),nas_perc)
plt.xticks(rotation=45, ha='right')
plt.title('Percentage of NAs by Variable')
plt.show()



#mean imputation for covid_vaccinated dataset

cv_target = df_cv['smoothed_wcovid_vaccinated']
cv_features = df_cv.drop('smoothed_wcovid_vaccinated', axis = 'columns')

cv_features.columns = cv_features.columns.astype(str)
imp = SimpleImputer()

cv_features = imp.fit_transform(cv_features)

cv_features = pd.DataFrame(cv_features, columns = imp.feature_names_in_)

missing_target = pd.DataFrame([1 if pd.notna(x) else 0 for x in df_cv['smoothed_wcovid_vaccinated']])

cv_features_impute_drop = cv_features.loc[missing_target.squeeze().astype(bool), :]
cv_target_impute_drop = cv_target.loc[missing_target.squeeze().astype(bool)]
cv_features_final = cv_features_impute_drop
cv_target_final = cv_target_impute_drop
cv_final = cv_features_final.join(cv_target_final)

cv_train, cv_test = train_test_split(cv_final, test_size = 0.20, random_state = 95828)

cv_train.to_csv('cv_train.csv')
cv_test.to_csv('cv_test.csv')

# mean imputation for features in tested_positive dataset

#imputing the mean value into the features columns
tp_target = df_tp['smoothed_wtested_positive_14d']
tp_features = df_tp.drop('smoothed_wtested_positive_14d', axis = 'columns')
tp_features.columns = tp_features.columns.astype(str)
imp = SimpleImputer()

tp_features = imp.fit_transform(tp_features)
tp_features = pd.DataFrame(tp_features, columns = imp.feature_names_in_)

df_tp = tp_features.join(tp_target)

print('features imputed')

#splitting the data into the data which has the target has the target missing and the data which has the target present
tp_target_notna = df_tp.dropna(subset=['smoothed_wtested_positive_14d'])
tp_target_isna = df_tp.drop(tp_target_notna.index.tolist())

#splitting into target and features again
#for the data that has the target, splitting into target and features
tp_notna_target = tp_target_notna['smoothed_wtested_positive_14d']
tp_notna_features = tp_target_notna.drop('smoothed_wtested_positive_14d', axis = 'columns')

#for the data that doesn't have the target, splitting into target and features
tp_isna_target = tp_target_isna['smoothed_wtested_positive_14d']
tp_isna_features = tp_target_isna.drop('smoothed_wtested_positive_14d', axis = 'columns')


#imputing values from Bayesian Linear Regression using Matern Kernel
#This cell takes about 10 minutes to run
#training a GP on the data that has the target, then drawing from the GP to assign values to the data that doesn't have the target
mat = gp.kernels.Matern(nu = .5)
gp_reg = gp.GaussianProcessRegressor(kernel = mat, normalize_y = True)
gp_reg.fit(tp_notna_features,tp_notna_target)

tp_isna_target = gp_reg.predict(tp_isna_features)

plt.hist(tp_isna_target,  label = 'imputed values', bins = 50)
plt.hist(tp_notna_target, alpha = 0.85, label = 'non-imputed values', bins = 50)

plt.ylabel('count')
plt.xlabel('smoothed_wtested_positive_14d')
plt.legend()
plt.show()

bayesian_imputed_targets = pd.DataFrame(tp_isna_target)

test = tp_target_isna.to_numpy()

test = np.concatenate((test, bayesian_imputed_targets), axis = 1)
test = pd.DataFrame(test)

column_list = tp_target_notna.columns.tolist()
column_list.append('imputed_targets')

column_dict = dict()

for i in range(17):
    column_dict[i] = column_list[i]

test = test.rename(columns = column_dict)

test = test.drop('smoothed_wtested_positive_14d', axis = 'columns')

tp_original = tp_target_notna.rename(columns = {'smoothed_wtested_positive_14d': 'tested_pos'})
tp_imputed = test2.rename(columns = {'imputed_targets': 'tested_pos'})

tp_original_train, tp_original_test = train_test_split(tp_original, test_size= 0.2, random_state = 95828)
tp_wimputed_train = pd.concat([tp_original_train, tp_imputed], axis = 0)


tp_original_train.to_csv('tp_original_train.csv')
tp_original_test.to_csv('tp_original_test.csv')
tp_wimputed_train.to_csv('tp_wimputed_train.csv')