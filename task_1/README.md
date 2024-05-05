# Age Estimation from Brain Image Data

## 1. Task

This project focuses on estimating a person's age based on features derived from brain image data. 
Accurately estimating brain age could enhance early diagnosis of age-related neurodegenerative diseases.


## 2. Data

**Disclaimer:** The data, containing sensitive information, is not publicly available.

The dataset comprises features extracted from MRI brain scans using FreeSurfer, rather than raw MRI images. To simulate real-life challenges, the data was modified as follows:
- Inclusion of useless features: of the 800 covariates, only about 200 are actual anatomical features.
- Inclusion of outliers.
- Perturbations such as missing values (i.e., NaN).


## 3. Methodology

### 3.1. Preprocessing

1. **Outlier Detection and Removal**: Outliers are identified using the Isolation Forest method. Initially, NaN values are handled through median imputation, as Isolation Forest cannot process NaN values directly.
2. **Feature Selection**: Correlation analysis between each covariate and the target age is performed to identify relevant features. A threshold is set to select features with significant correlations.
3. **Data Imputation**: Following outlier and useless feature removal, K-Nearest Neighbors (KNN) imputation is applied to address missing values.
4. **Scaling**: Data is scaled using RobustScaler from Scikit-learn.


### 3.2. Model

A Gaussian Process Regressor with a Rational Quadratic kernel is employed for age estimation. This choice is motivated by the model's ability to provide uncertainty estimates — valuable in medical settings — and its effectiveness with small datasets, which is common in medical imaging due to the high costs associated with MRI technology.


## 4. Evaluation

The coefficient of determination (also known as $R^2$ score) is used to assess how well the model predicts the outcome of the dependent variable.