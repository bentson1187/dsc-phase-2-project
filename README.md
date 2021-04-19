<img src='Images/Rennovation README Title.png'>

# Evaluating Home Renovation Opportunities in King County, WA

# Overview

This project analyzes historic home prices in King County, Washington in order to understand which home features drive the value of the home. The analysis uses a linear regression model in order to determine the link between particular home features and sales price. This information will better inform homeowners on smart ways to rennovate their homes to maximize value.

# Business Problem

Most people's largest asset is their home which acts as the foundation for their net worth. Therefore, it is imperative that the value of this asset improves over time through either property value inflation or smart renovations. Since property values are largely based on location and current market conditions, which are outside of your control, renovations are one of the only controllable factors when trying to improve a homes value. In this analysis, I will explore which factors in a house are most correlated to higher value by looking at historical sales of homes in King County, Washington. I will then make recommendations to prospective home rennovators to help them make smart decisions to improve their homes value.

# Data

I obtained data of home sales from King County, Washington between 2014-2015. The data initially contained 20,000 home sales and was narrowed down to 14,000 after cleaning up bad data and removing outliers. The data had many important home features and I will highlight some below:
- Price 
- Total Home Square Feet
- Bedrooms
- Bathrooms
- Grade (construction quality and design)
- View
- Condition
- Home Age

<img src='Images/Zipcode Map of Median Home Prices.png' width=80%>

# Methods

This project focuses on utilizing the historic pricing data and home features to model which features can most accurately predict the value of a home. Most home features were utilized in developing a linear regression model to best predict the homes value, however, I will focus on only features a homeowner can control when providing my recommendations. Considerations were made on the data to ensure the final model is accurate and adheres to the requirements of linearity between features and price, multicollinearity between features and normality and homoscedasticity of the model's residuals.

# Results

The model analyzes how specific features of a home affect its value and uses this analysis to then determine a "best fit" prediction when trying to take in new home features and produce a new home value. Here is how the model classified each home feature with respect to how it influenced a homes value:

<img src='Images/Model Feature Influence.png'>

The feature which affects a home's value most positively is Total Home Square Feet. The grade of construction is also an important feature to predicting a home's value. The model utilizes the impact of the home features to give insights into rennovation opportunities.

## Rennovation Opportunity: Finish a 700 sqft Basement 

The model shows that having a basement negatively impacts a home's value. However, if a home has an unfinished basement with over 350 square feet then the extra square feet to the house may offset the negative affect of the basement. The median basement size for homes in this area is 700 square feet. If a homeowner would like to finish their basement which would add 700 square feet and use a high quality grade of construction, the predicted increase in a home's value is $81,000:

<img src='Images/Basement Rennovation.png'>

## Rennovation Opportunity: Add a Full Size Bathroom

The model shows that adding a bathroom positively impacts a home's value. If a homeowner were to add a full size bathroom (60 sqft) with high construction quality the model predicts an increase in a home's value of $43,200:

<img src='Images/Bathroom Rennovation.png'>

# Conclusions

For a homeowner looking to improve their home's value through rennovation, finishing a large basement or adding a full size bathroom will significantly improve a home's value. It is extremly important that the amount of additional square feet is large enough to offset costs and that the construction quality is high enough to last many years to come. Lastly, a few words of caution: The model suggests that adding an entire floor or adding a bedroom will most likely negatively impact a home's value unless the additional square feet is large enough to offset the negative impact and cost.

## **For More Information**

Please review our full analysis in my [Jupyter Notebook](https://github.com/bentson1187/dsc-phase-2-project/blob/231d7829a588150c9e801101b9ef99029747c3db/Bentson,%20Brian%20Phase%202%20Project.ipynb) or my [presentation](https://github.com/bentson1187/dsc-phase-2-project/blob/231d7829a588150c9e801101b9ef99029747c3db/Stakeholder%20Presentation.pptx).

For any additional questions, please contact **Brian Bentson, bentson.brian@gmail.com**

## Repository Structure

Describe the structure of your repository and its contents, for example:

```
├── README.md                           <- The top-level README for reviewers of this project
├── Bentson,Brian Phase 2 Project.ipynb <- Narrative documentation of analysis in Jupyter notebook
├── Jupyter Notebook.pdf                <- PDF version of the analysis in Jupyter notebook
├── Microsoft Presentation.pdf          <- PDF version of project presentation
├── data                          <- Both sourced externally and generated from code
└── images                              <- Both sourced externally and generated from code
```



# Table of Contents

*Click to jump to matching Markdown Header.*<br><br>
 
- **[INTRODUCTION](#Introduction)<br>**
- **[OBTAIN](#Obtain)**<br>
- **[SCRUB](#Scrub)**<br>
- **[EXPLORE](#Explore)**<br>
- **[MODEL](#Model)**<br>
- **[INTERPRET](#Interpret)**<br>
- **[RECOMMENDATIONS/CONCLUSIONS](#Recommendations-and-Conclusions)<br>**

# Introduction

## Business Statement


## Analysis Methodology

I will be analyzing historic home sales from King County, Washington in order to see which factors affect home price and how a model can be built to predict good estimates for home listing prices. This model will give insights into what a current home owner could do in order to improve their home value. I will focus only on features which a home owner has control over.

# Obtain

## Import Packages


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
plt.style.use('fivethirtyeight')

import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
import statsmodels.stats.api as sms
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
```


```python
pd.set_option("display.max_columns", 30)
pd.options.display.float_format = '{:,}'.format
```

## Global Functions


```python
#function to look at plots and stats of column with or without outliers
def get_plots(df, x_col, y_col='price', outlier='none'):
    
    """This function takes in a dataframe and a column, removes outliers using
       standard deviations or iqr and produces a histogram, scatter plot and
       boxplot of the values with descriptive statistics"""
    
    #plots for std
    if outlier == 'std':
        #create variables
        col_mean = df[x_col].mean()
        col_std = df[x_col].std()
        upper_thresh_std = col_mean + 3*col_std
        lower_thresh_std = col_mean - 3*col_std
        
        #create new df
        idx_std_outliers = (df[x_col] > lower_thresh_std) & (df[x_col] < upper_thresh_std)
        std_df = df.loc[idx_std_outliers]
        
        #plots
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,10));
        histogram = std_df[x_col].hist(ax=ax[0,0]);
        ax[0,0].set_title(f'Distribution of {x_col}');

        scatter = std_df.plot(kind='scatter', x=x_col, y=y_col,ax=ax[0,1]);
        ax[0,1].set_title(f'{y_col} vs {x_col}');

        boxplot = std_df.boxplot(column=x_col, ax=ax[1,0]);
        ax[1,0].set_title(f'Boxplot of {x_col}');
        
        sm.graphics.qqplot(std_df[x_col], dist=stats.norm, line='45', fit=True, ax=ax[1,1])
        ax[1,1].set_title(f'QQ plot of {x_col}');
        
        #stats
        rows_removed = len(df) - len(std_df)
        print(f'The number of rows removed is {rows_removed}')
        desc_stats = std_df[x_col].describe()
        plt.tight_layout()
        
    elif outlier == 'iqr':
        #create variables
        q25 = df[x_col].quantile(0.25)
        q75 = df[x_col].quantile(0.75)
        iqr = q75-q25
        upper_thresh_iqr = q75 + 1.5*iqr
        lower_thresh_iqr = q25 - 1.5*iqr
        
        #create new df
        idx_iqr_outliers = (df[x_col] > lower_thresh_iqr) & (df[x_col] < upper_thresh_iqr)
        iqr_df = df.loc[idx_iqr_outliers]
        
        #plots
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,10));
        histogram = iqr_df[x_col].hist(ax=ax[0,0]);
        ax[0,0].set_title(f'Distribution of {x_col}');

        scatter = iqr_df.plot(kind='scatter', x=x_col, y=y_col,ax=ax[0,1]);
        ax[0,1].set_title(f'{y_col} vs {x_col}');

        boxplot = iqr_df.boxplot(column=x_col, ax=ax[1,0]);
        ax[1,0].set_title(f'Boxplot of {x_col}');
        
        sm.graphics.qqplot(iqr_df[x_col], dist=stats.norm, line='45', fit=True, ax=ax[1,1])
        ax[1,1].set_title(f'QQ plot of {x_col}');
        
        #stats
        rows_removed = len(df) - len(iqr_df)
        print(f'The number of rows removed is {rows_removed}')
        desc_stats = df[x_col].describe()
        plt.tight_layout()
    
    elif outlier == 'none':
        #plots
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,10));
        histogram = df[x_col].hist(ax=ax[0,0]);
        ax[0,0].set_title(f'Distribution of {x_col}');

        scatter = df.plot(kind='scatter', x=x_col, y=y_col,ax=ax[0,1]);
        ax[0,1].set_title(f'{y_col} vs {x_col}');

        boxplot = df.boxplot(column=x_col, ax=ax[1,0]);
        ax[1,0].set_title(f'Boxplot of {x_col}');

        sm.graphics.qqplot(df[x_col], dist=stats.norm, line='45', fit=True, ax=ax[1,1])
        ax[1,1].set_title(f'QQ plot of {x_col}');
        
        #stats
        desc_stats = df[x_col].describe()
        plt.tight_layout()

        
    print(desc_stats)
    plt.show()
    
    return
```


```python
#function to preprocess and create a new model
def fit_new_model(df, x_cols=None, y_col=None, norm=False, diagnose=True):
    '''This function takes in a dataframe, a list of independent and dependent
       variables and whether or not you want to normalize the columns. The 
       output is a multiple linear regression model with checks for 
       multicollinearity, normality and homoscedasticity.'''
         
    #step 1: normalize columns
    if norm == True:
        for col in x_cols:
            df[col] = (df[col] - df[col].mean())/df[col].std()
        #display the normalized df
        display(df.head())
        print('\n')
    else:
        #display the df
        display(df.head())
        print('\n')
    
    #step 2: create model
    
    #set up model parameters
    x_cols = x_cols
    outcome = y_col
    predictors = '+'.join(x_cols)
    formula = outcome + '~' + predictors
    #fit the model
    model = ols(formula=formula, data=df).fit()
    print(model.summary())
    print('\n')
    
    if diagnose == True:
        #step 3: check multicollinearity
        print('VIF Multicollinearity Test Results')
        print('======================================================================================================')
        #run VIF test
        X = df[x_cols]
        vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        display(list(zip(x_cols, vif)))
        print('\n')

        #step 4: check normality
        print('Normality Test Results')
        print('======================================================================================================')
        #plot qqplot
        fig, ax = plt.subplots(figsize=(15,10))
        sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True, ax=ax);
        ax.set_title('QQPlot for Model Residuals')
        plt.show()
        print('\n')

        #step 5: check homoscedasticity
        print('Homoscedasticity Test Results')
        print('======================================================================================================')
        #scatter plot
        fig, ax = plt.subplots(figsize=(15,10))
        plt.scatter(model.predict(df[x_cols]), model.resid)
        plt.plot(model.predict(df[x_cols]), [0 for i in range(len(df))])
        ax.set_title('Model Residuals vs Model Predictions')
        plt.show()
    else:
        pass
    return model
```


```python
#function to delete outliers using either iqr or std
def outliers(df, col, outlier='std'):
    '''This function takes in a dataframe, a column in the dataframe and 
       whether or not to remove outliers via standard deviations or
       interquartile range.'''
        
    if outlier == 'std':
        #create outlier variables
        col_mean = df[col].mean()
        col_std = df[col].std()
        upper_thresh_std = col_mean + 3*col_std
        lower_thresh_std = col_mean - 3*col_std
        #update dataframe
        df_new = df.loc[(df[col] > lower_thresh_std) & (df[col] < upper_thresh_std)]
        print(f'There were {len(df) - len(df_new)} outliers removed.')
    elif outlier == 'iqr':
        #create outlier variables
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75-q25
        upper_thresh_iqr = q75 + 1.5*iqr
        lower_thresh_iqr = q25 - 1.5*iqr
        #create new dataframe with outliers removed
        df_new = df.loc[(df[col] > lower_thresh_iqr) & (df[col] < upper_thresh_iqr)]
        print(f'There were {len(df) - len(df_new)} outliers removed.')

    return df_new
```

## Import Data into Pandas

I will be importing a csv dataset which provides me with the information necessary to begin the analysis.


```python
#import the dataset from local csv
df_original = pd.read_csv('Data/kc_house_data.csv')
df_original
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221,900.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>nan</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538,000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1,991.0</td>
      <td>98125</td>
      <td>47.721000000000004</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180,000.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>nan</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.23299999999999</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604,000.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.39299999999999</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510,000.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>263000018</td>
      <td>5/21/2014</td>
      <td>360,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1530</td>
      <td>0.0</td>
      <td>2009</td>
      <td>0.0</td>
      <td>98103</td>
      <td>47.6993</td>
      <td>-122.346</td>
      <td>1530</td>
      <td>1509</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>6600060120</td>
      <td>2/23/2015</td>
      <td>400,000.0</td>
      <td>4</td>
      <td>2.5</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2310</td>
      <td>0.0</td>
      <td>2014</td>
      <td>0.0</td>
      <td>98146</td>
      <td>47.5107</td>
      <td>-122.36200000000001</td>
      <td>1830</td>
      <td>7200</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>1523300141</td>
      <td>6/23/2014</td>
      <td>402,101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>0.0</td>
      <td>2009</td>
      <td>0.0</td>
      <td>98144</td>
      <td>47.5944</td>
      <td>-122.29899999999999</td>
      <td>1020</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>291310100</td>
      <td>1/16/2015</td>
      <td>400,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>nan</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1600</td>
      <td>0.0</td>
      <td>2004</td>
      <td>0.0</td>
      <td>98027</td>
      <td>47.5345</td>
      <td>-122.069</td>
      <td>1410</td>
      <td>1287</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>1523300157</td>
      <td>10/15/2014</td>
      <td>325,000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>0.0</td>
      <td>2008</td>
      <td>0.0</td>
      <td>98144</td>
      <td>47.5941</td>
      <td>-122.29899999999999</td>
      <td>1020</td>
      <td>1357</td>
    </tr>
  </tbody>
</table>
<p>21597 rows × 21 columns</p>
</div>



## Data Schema

**Taken from https://rstudio-pubs-static.s3.amazonaws.com/155304_cc51f448116744069664b35e7762999f.html**

id - Unique ID for each home sold 

date - Date of the home sale 

price - Price of each home sold 

bedrooms - Number of bedrooms 

bathrooms - Number of bathrooms, where .5 accounts for a room with a toilet but no shower 

sqft_living - Square footage of the apartments interior living space 

sqft_lot - Square footage of the land space 

floors - Number of floors 

waterfront - A dummy variable for whether the apartment was overlooking the waterfront or not 

view - An index from 0 to 4 of how good the view of the property was 

condition - An index from 1 to 5 on the condition of the home

grade - An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design. 

sqft_above - The square footage of the interior housing space that is above ground level 

sqft_basement - The square footage of the interior housing space that is below ground level 

yr_built - The year the house was initially built 

yr_renovated - The year of the house’s last renovation 

zipcode - What zipcode area the house is in 

lat - Lattitude 

long - Longitude 

sqft_living15 - The square footage of interior housing living space for the nearest 15 neighbors 

sqft_lot15 - The square footage of the land lots of the nearest 15 neighbors 

## Investigate Data

I will preliminarily investigate the data to identify any glaring issues to fix later.


```python
#column names
df_original.columns
```




    Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
           'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
           'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
           'lat', 'long', 'sqft_living15', 'sqft_lot15'],
          dtype='object')




```python
#view df info to inspect data types
df_original.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 21 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   id             21597 non-null  int64  
     1   date           21597 non-null  object 
     2   price          21597 non-null  float64
     3   bedrooms       21597 non-null  int64  
     4   bathrooms      21597 non-null  float64
     5   sqft_living    21597 non-null  int64  
     6   sqft_lot       21597 non-null  int64  
     7   floors         21597 non-null  float64
     8   waterfront     19221 non-null  float64
     9   view           21534 non-null  float64
     10  condition      21597 non-null  int64  
     11  grade          21597 non-null  int64  
     12  sqft_above     21597 non-null  int64  
     13  sqft_basement  21597 non-null  object 
     14  yr_built       21597 non-null  int64  
     15  yr_renovated   17755 non-null  float64
     16  zipcode        21597 non-null  int64  
     17  lat            21597 non-null  float64
     18  long           21597 non-null  float64
     19  sqft_living15  21597 non-null  int64  
     20  sqft_lot15     21597 non-null  int64  
    dtypes: float64(8), int64(11), object(2)
    memory usage: 3.5+ MB


> **OBSERVATIONS**
> - `waterfront` values should be updated to a binary categorical data type
> - `yr_renovated` values should be updated to binary categorical data type
> - `sqft_basement` values should be updated to a binary categorical data type


```python
#check for null values
df_original.isna().sum()/len(df_original)*100
```




    id                              0.0
    date                            0.0
    price                           0.0
    bedrooms                        0.0
    bathrooms                       0.0
    sqft_living                     0.0
    sqft_lot                        0.0
    floors                          0.0
    waterfront        11.00152798999861
    view            0.29170718155299347
    condition                       0.0
    grade                           0.0
    sqft_above                      0.0
    sqft_basement                   0.0
    yr_built                        0.0
    yr_renovated      17.78950780200954
    zipcode                         0.0
    lat                             0.0
    long                            0.0
    sqft_living15                   0.0
    sqft_lot15                      0.0
    dtype: float64



> **OBSERVATIONS**
> - `waterfront` has 11% null values which is a large number to simply drop. Will evaluate options.
> - `view` should be explored further to see what it means
> - `yr_renovated` has 18% null values which is a large number to simply drop. Will evaluate options.
> - All other columns have 0 nulls


```python
#check numeric data
df_original.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>21,597.0</td>
      <td>21,597.0</td>
      <td>21,597.0</td>
      <td>21,597.0</td>
      <td>21,597.0</td>
      <td>21,597.0</td>
      <td>21,597.0</td>
      <td>19,221.0</td>
      <td>21,534.0</td>
      <td>21,597.0</td>
      <td>21,597.0</td>
      <td>21,597.0</td>
      <td>21,597.0</td>
      <td>17,755.0</td>
      <td>21,597.0</td>
      <td>21,597.0</td>
      <td>21,597.0</td>
      <td>21,597.0</td>
      <td>21,597.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4,580,474,287.770987</td>
      <td>540,296.5735055795</td>
      <td>3.3731999814789093</td>
      <td>2.1158262721674306</td>
      <td>2,080.3218502569803</td>
      <td>15,099.408760475992</td>
      <td>1.4940964022780943</td>
      <td>0.007595858696217679</td>
      <td>0.23386272870808952</td>
      <td>3.4098254387183404</td>
      <td>7.657915451220076</td>
      <td>1,788.5968421540028</td>
      <td>1,970.9996758809093</td>
      <td>83.6367783722895</td>
      <td>98,077.95184516368</td>
      <td>47.56009299439733</td>
      <td>-122.21398249756845</td>
      <td>1,986.6203176367087</td>
      <td>12,758.283511598833</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2,876,735,715.74778</td>
      <td>367,368.1401013936</td>
      <td>0.926298894542015</td>
      <td>0.7689842966527002</td>
      <td>918.1061250800823</td>
      <td>41,412.63687550209</td>
      <td>0.5396827909775687</td>
      <td>0.08682484570055837</td>
      <td>0.76568620117451</td>
      <td>0.6505456356724978</td>
      <td>1.1731996637757696</td>
      <td>827.7597611646777</td>
      <td>29.37523413244173</td>
      <td>399.94641387879193</td>
      <td>53.51307235352649</td>
      <td>0.1385517681730714</td>
      <td>0.1407235288294722</td>
      <td>685.2304719001584</td>
      <td>27,274.441950386576</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1,000,102.0</td>
      <td>78,000.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>370.0</td>
      <td>520.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>370.0</td>
      <td>1,900.0</td>
      <td>0.0</td>
      <td>98,001.0</td>
      <td>47.1559</td>
      <td>-122.51899999999999</td>
      <td>399.0</td>
      <td>651.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2,123,049,175.0</td>
      <td>322,000.0</td>
      <td>3.0</td>
      <td>1.75</td>
      <td>1,430.0</td>
      <td>5,040.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>1,190.0</td>
      <td>1,951.0</td>
      <td>0.0</td>
      <td>98,033.0</td>
      <td>47.4711</td>
      <td>-122.32799999999999</td>
      <td>1,490.0</td>
      <td>5,100.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3,904,930,410.0</td>
      <td>450,000.0</td>
      <td>3.0</td>
      <td>2.25</td>
      <td>1,910.0</td>
      <td>7,618.0</td>
      <td>1.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>1,560.0</td>
      <td>1,975.0</td>
      <td>0.0</td>
      <td>98,065.0</td>
      <td>47.5718</td>
      <td>-122.23100000000001</td>
      <td>1,840.0</td>
      <td>7,620.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7,308,900,490.0</td>
      <td>645,000.0</td>
      <td>4.0</td>
      <td>2.5</td>
      <td>2,550.0</td>
      <td>10,685.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>8.0</td>
      <td>2,210.0</td>
      <td>1,997.0</td>
      <td>0.0</td>
      <td>98,118.0</td>
      <td>47.678000000000004</td>
      <td>-122.125</td>
      <td>2,360.0</td>
      <td>10,083.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9,900,000,190.0</td>
      <td>7,700,000.0</td>
      <td>33.0</td>
      <td>8.0</td>
      <td>13,540.0</td>
      <td>1,651,359.0</td>
      <td>3.5</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>9,410.0</td>
      <td>2,015.0</td>
      <td>2,015.0</td>
      <td>98,199.0</td>
      <td>47.7776</td>
      <td>-121.315</td>
      <td>6,210.0</td>
      <td>871,200.0</td>
    </tr>
  </tbody>
</table>
</div>



# Scrub

I will make a new dataframe which is a copy of the `df_original` dataframe to begin making changes. 


```python
#create a copy of the original dataframe
df_scrub = df_original.copy()
df_scrub
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221,900.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>nan</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538,000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1,991.0</td>
      <td>98125</td>
      <td>47.721000000000004</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180,000.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>nan</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.23299999999999</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604,000.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.39299999999999</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510,000.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>263000018</td>
      <td>5/21/2014</td>
      <td>360,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1530</td>
      <td>0.0</td>
      <td>2009</td>
      <td>0.0</td>
      <td>98103</td>
      <td>47.6993</td>
      <td>-122.346</td>
      <td>1530</td>
      <td>1509</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>6600060120</td>
      <td>2/23/2015</td>
      <td>400,000.0</td>
      <td>4</td>
      <td>2.5</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2310</td>
      <td>0.0</td>
      <td>2014</td>
      <td>0.0</td>
      <td>98146</td>
      <td>47.5107</td>
      <td>-122.36200000000001</td>
      <td>1830</td>
      <td>7200</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>1523300141</td>
      <td>6/23/2014</td>
      <td>402,101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>0.0</td>
      <td>2009</td>
      <td>0.0</td>
      <td>98144</td>
      <td>47.5944</td>
      <td>-122.29899999999999</td>
      <td>1020</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>291310100</td>
      <td>1/16/2015</td>
      <td>400,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>nan</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1600</td>
      <td>0.0</td>
      <td>2004</td>
      <td>0.0</td>
      <td>98027</td>
      <td>47.5345</td>
      <td>-122.069</td>
      <td>1410</td>
      <td>1287</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>1523300157</td>
      <td>10/15/2014</td>
      <td>325,000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>0.0</td>
      <td>2008</td>
      <td>0.0</td>
      <td>98144</td>
      <td>47.5941</td>
      <td>-122.29899999999999</td>
      <td>1020</td>
      <td>1357</td>
    </tr>
  </tbody>
</table>
<p>21597 rows × 21 columns</p>
</div>



## Feature Engineering

### `basement` Column

In the dataset we have 3 related columns:
- `sqft_above`
- `sqft_basement`
- `sqft_living`

These columns are related in that `sqft_living` equals `sqft_above` plus `sqft_basement`. I do not think the square footage of the basement is as important as just knowing that a house has one. Therefore, I will create a new column which shows whether or not a house has a basement.


```python
#investigate values in sqft_basement
display(df_scrub['sqft_basement'].value_counts(),len(df_scrub), df_scrub['sqft_basement'].dtype)
```


    0.0      12826
    ?          454
    600.0      217
    500.0      209
    700.0      208
             ...  
    243.0        1
    946.0        1
    508.0        1
    935.0        1
    417.0        1
    Name: sqft_basement, Length: 304, dtype: int64



    21597



    dtype('O')


> **ACTIONS**
> - '?' impedes the ability to create a new column. Will drop convert this to a 0 to indicate that the house does not have a basement.


```python
#convert rows with a '?' to a 0
df_scrub.loc[df_scrub['sqft_basement'] == '?', ['sqft_basement']] = '0.0'
display(df_scrub['sqft_basement'].value_counts(),len(df_scrub))
```


    0.0       13280
    600.0       217
    500.0       209
    700.0       208
    800.0       201
              ...  
    792.0         1
    2850.0        1
    2350.0        1
    1481.0        1
    652.0         1
    Name: sqft_basement, Length: 303, dtype: int64



    21597



```python
#prove that these columns are related
df_scrub['sqft_basement'] = df_scrub['sqft_basement'].astype(float).astype(int)
sqft = df_scrub[['sqft_living', 'sqft_above', 'sqft_basement']]
(sqft['sqft_above'] + sqft['sqft_basement'] == sqft['sqft_living']).value_counts()
```




    True     21427
    False      170
    dtype: int64




```python
#investigate the Falses
df_scrub.loc[(sqft['sqft_above'] + sqft['sqft_basement'] == sqft['sqft_living']) == False]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>112</th>
      <td>2525310310</td>
      <td>9/16/2014</td>
      <td>272,500.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>1540</td>
      <td>12600</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>7</td>
      <td>1160</td>
      <td>0</td>
      <td>1980</td>
      <td>0.0</td>
      <td>98038</td>
      <td>47.3624</td>
      <td>-122.031</td>
      <td>1540</td>
      <td>11656</td>
    </tr>
    <tr>
      <th>115</th>
      <td>3626039325</td>
      <td>11/21/2014</td>
      <td>740,500.0</td>
      <td>3</td>
      <td>3.5</td>
      <td>4380</td>
      <td>6350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2780</td>
      <td>0</td>
      <td>1900</td>
      <td>nan</td>
      <td>98117</td>
      <td>47.6981</td>
      <td>-122.368</td>
      <td>1830</td>
      <td>6350</td>
    </tr>
    <tr>
      <th>309</th>
      <td>3204800200</td>
      <td>1/8/2015</td>
      <td>665,000.0</td>
      <td>4</td>
      <td>2.75</td>
      <td>3320</td>
      <td>10574</td>
      <td>2.0</td>
      <td>nan</td>
      <td>0.0</td>
      <td>5</td>
      <td>8</td>
      <td>2220</td>
      <td>0</td>
      <td>1960</td>
      <td>0.0</td>
      <td>98056</td>
      <td>47.5376</td>
      <td>-122.18</td>
      <td>2720</td>
      <td>8330</td>
    </tr>
    <tr>
      <th>384</th>
      <td>713500030</td>
      <td>7/28/2014</td>
      <td>1,350,000.0</td>
      <td>5</td>
      <td>3.5</td>
      <td>4800</td>
      <td>14984</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3</td>
      <td>11</td>
      <td>3480</td>
      <td>0</td>
      <td>1998</td>
      <td>0.0</td>
      <td>98006</td>
      <td>47.5543</td>
      <td>-122.148</td>
      <td>4050</td>
      <td>19009</td>
    </tr>
    <tr>
      <th>508</th>
      <td>5113400431</td>
      <td>5/8/2014</td>
      <td>615,000.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>1540</td>
      <td>6872</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>7</td>
      <td>820</td>
      <td>0</td>
      <td>1946</td>
      <td>0.0</td>
      <td>98119</td>
      <td>47.6454</td>
      <td>-122.37299999999999</td>
      <td>1420</td>
      <td>5538</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21000</th>
      <td>291310180</td>
      <td>6/13/2014</td>
      <td>379,500.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>1410</td>
      <td>1287</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1290</td>
      <td>0</td>
      <td>2005</td>
      <td>0.0</td>
      <td>98027</td>
      <td>47.5344</td>
      <td>-122.068</td>
      <td>1490</td>
      <td>1435</td>
    </tr>
    <tr>
      <th>21109</th>
      <td>3438500250</td>
      <td>6/23/2014</td>
      <td>515,000.0</td>
      <td>5</td>
      <td>3.25</td>
      <td>2910</td>
      <td>5027</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2040</td>
      <td>0</td>
      <td>2013</td>
      <td>0.0</td>
      <td>98106</td>
      <td>47.5543</td>
      <td>-122.359</td>
      <td>2910</td>
      <td>5027</td>
    </tr>
    <tr>
      <th>21210</th>
      <td>3278600680</td>
      <td>6/27/2014</td>
      <td>235,000.0</td>
      <td>1</td>
      <td>1.5</td>
      <td>1170</td>
      <td>1456</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1070</td>
      <td>0</td>
      <td>2007</td>
      <td>0.0</td>
      <td>98126</td>
      <td>47.5493</td>
      <td>-122.37200000000001</td>
      <td>1360</td>
      <td>1730</td>
    </tr>
    <tr>
      <th>21356</th>
      <td>6169901185</td>
      <td>5/20/2014</td>
      <td>490,000.0</td>
      <td>5</td>
      <td>3.5</td>
      <td>4460</td>
      <td>2975</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3</td>
      <td>10</td>
      <td>3280</td>
      <td>0</td>
      <td>2015</td>
      <td>nan</td>
      <td>98119</td>
      <td>47.6313</td>
      <td>-122.37</td>
      <td>2490</td>
      <td>4231</td>
    </tr>
    <tr>
      <th>21442</th>
      <td>3226049565</td>
      <td>7/11/2014</td>
      <td>504,600.0</td>
      <td>5</td>
      <td>3.0</td>
      <td>2360</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1390</td>
      <td>0</td>
      <td>2008</td>
      <td>0.0</td>
      <td>98103</td>
      <td>47.6931</td>
      <td>-122.33</td>
      <td>2180</td>
      <td>5009</td>
    </tr>
  </tbody>
</table>
<p>170 rows × 21 columns</p>
</div>



>**OBSERVATIONS**
> - It seems that 170 homes have a difference between the `sqft_living` and `sqft_above` that were originially classified as a '?'.

>**ACTIONS**
> - I will now assume that the difference in these 170 homes is due to having sq_ft in the basement. I will change from a 0 to the difference in `sqft_living` and `sqft_above`


```python
#replace sqft_basement with the desscrepency between sqft_living and sqft_above
df_scrub['sqft_basement'] = df_scrub['sqft_living']- df_scrub['sqft_above']
display(df_scrub['sqft_basement'].value_counts(), len(df_scrub))
```


    0       13110
    600       221
    700       218
    500       214
    800       206
            ...  
    792         1
    2590        1
    935         1
    2390        1
    248         1
    Name: sqft_basement, Length: 306, dtype: int64



    21597



```python
#check previous id where sqft_basement was a '?'
df_scrub.loc[df_scrub['id'] == 2525310310]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>112</th>
      <td>2525310310</td>
      <td>9/16/2014</td>
      <td>272,500.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>1540</td>
      <td>12600</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>7</td>
      <td>1160</td>
      <td>380</td>
      <td>1980</td>
      <td>0.0</td>
      <td>98038</td>
      <td>47.3624</td>
      <td>-122.031</td>
      <td>1540</td>
      <td>11656</td>
    </tr>
  </tbody>
</table>
</div>




```python
#check the rows have been dropped
df_scrub.loc[df_scrub['sqft_basement'] == '?']['sqft_basement'].count()
```




    0




```python
#check to ensure all descrepencies are gone
df_scrub.loc[(df_scrub['sqft_above'] + df_scrub['sqft_basement'] == df_scrub['sqft_living']) == False]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
#what is the median basement sqft
df_scrub.loc[df_scrub['sqft_basement'] > 0,'sqft_basement'].median()
```




    700.0



> **ACTIONS**
> - Will now create new column named `basement` which represents whether or not a house has a basement.


```python
#create new column for basement and verify 
df_scrub['basement'] = np.where(df_scrub['sqft_basement'] > 0, 1,0)
df_scrub[['sqft_basement','basement']].value_counts()
```




    sqft_basement  basement
    0              0           13110
    600            1             221
    700            1             218
    500            1             214
    800            1             206
                               ...  
    2360           1               1
    475            1               1
    2350           1               1
    1930           1               1
    4820           1               1
    Length: 306, dtype: int64




```python
#how much more sqft does a house have when they have a basement?
df_scrub.groupby(by='basement')['sqft_living'].median()
```




    basement
    0    1740
    1    2100
    Name: sqft_living, dtype: int64



### `renovated` Column

I want to reconfigure the `yr_renovated` column so that it is compatible with the model. I will convert null rows and create a new column which indicates whether or not a house has been renovated.


```python
#check values in yr_renovated column
df_scrub['yr_renovated'].value_counts(dropna=False).head(20)
```




    0.0        17011
    nan         3842
    2,014.0       73
    2,003.0       31
    2,013.0       31
    2,007.0       30
    2,005.0       29
    2,000.0       29
    1,990.0       22
    2,004.0       22
    2,009.0       21
    1,989.0       20
    2,006.0       20
    2,002.0       17
    1,991.0       16
    1,998.0       16
    1,984.0       16
    1,999.0       15
    2,001.0       15
    2,008.0       15
    Name: yr_renovated, dtype: int64



> **ACTIONS**
> - I will set the null values to 0 which will be converted to a No


```python
#convert null to 0 and verify
df_scrub.loc[df_scrub['yr_renovated'].isna()] = 0
df_scrub['yr_renovated'].isna().sum()
```




    0



> **ACTIONS**
> - Create new `renovated` column which gives a 0 if false and 1 if true


```python
#create new column based on yr_renovated
df_scrub['renovated'] = np.where(df_scrub['yr_renovated'] == 0, 0, 1)
df_scrub[['yr_renovated','renovated']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yr_renovated</th>
      <th>renovated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1,991.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>21597 rows × 2 columns</p>
</div>



### `home_age` Column

I want to create a column named `home_age` which represents the homes age which I believe will be more informative in a model. I will take the sale `date` and subtract the `yr_built` from it to get the home age.


```python
#explore data values
df_scrub['yr_built'].value_counts()
```




    0       3842
    2014     457
    2006     380
    2005     378
    2004     364
            ... 
    1901      23
    1933      22
    1902      21
    1934      16
    1935      16
    Name: yr_built, Length: 117, dtype: int64



> **OBSERVATIONS**
> - There are 3842 homes which have a `yr_built` of 0

> **ACTIONS**
> - I will investigate this further to see how to proceed


```python
#view value_counts of all columns when yr_built is 0
df_scrub.loc[df_scrub['yr_built'] == 0].count()
```




    id               3842
    date             3842
    price            3842
    bedrooms         3842
    bathrooms        3842
    sqft_living      3842
    sqft_lot         3842
    floors           3842
    waterfront       3842
    view             3842
    condition        3842
    grade            3842
    sqft_above       3842
    sqft_basement    3842
    yr_built         3842
    yr_renovated     3842
    zipcode          3842
    lat              3842
    long             3842
    sqft_living15    3842
    sqft_lot15       3842
    basement         3842
    renovated        3842
    dtype: int64



>**OBSERVATIONS**
> - All columns are 0 for these 3,842 homes.

>**ACTIONS**
> - I will drop these rows


```python
#drop rows with 0's for values and check
df_scrub.drop(df_scrub.loc[df_scrub['yr_built'] == 0].index, inplace=True)
df_scrub.loc[df_scrub['yr_built'] == 0].count()
```




    id               0
    date             0
    price            0
    bedrooms         0
    bathrooms        0
    sqft_living      0
    sqft_lot         0
    floors           0
    waterfront       0
    view             0
    condition        0
    grade            0
    sqft_above       0
    sqft_basement    0
    yr_built         0
    yr_renovated     0
    zipcode          0
    lat              0
    long             0
    sqft_living15    0
    sqft_lot15       0
    basement         0
    renovated        0
    dtype: int64




```python
#create yr_sold column first
df_scrub['date'] = pd.to_datetime(df_scrub['date'])
df_scrub['yr_sold'] = df_scrub['date'].dt.year
df_scrub[['date','yr_built','yr_sold']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>yr_built</th>
      <th>yr_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-10-13</td>
      <td>1955</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-12-09</td>
      <td>1951</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-12-09</td>
      <td>1965</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-02-18</td>
      <td>1987</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2014-05-12</td>
      <td>2001</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>2014-05-21</td>
      <td>2009</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>2015-02-23</td>
      <td>2014</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>2014-06-23</td>
      <td>2009</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>2015-01-16</td>
      <td>2004</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>2014-10-15</td>
      <td>2008</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
<p>17755 rows × 3 columns</p>
</div>




```python
#create new column
df_scrub['home_age'] = df_scrub['yr_sold'] - df_scrub['yr_built']
#plot distribution
fig, ax = plt.subplots(figsize=(15,5))
df_scrub['home_age'].hist(ax=ax);
ax.set_title('Distribution of Home Age');
```


    
![png](output_65_0.png)
    


## Change Data Types


```python
#check data types
df_scrub.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 17755 entries, 0 to 21596
    Data columns (total 25 columns):
     #   Column         Non-Null Count  Dtype         
    ---  ------         --------------  -----         
     0   id             17755 non-null  int64         
     1   date           17755 non-null  datetime64[ns]
     2   price          17755 non-null  float64       
     3   bedrooms       17755 non-null  int64         
     4   bathrooms      17755 non-null  float64       
     5   sqft_living    17755 non-null  int64         
     6   sqft_lot       17755 non-null  int64         
     7   floors         17755 non-null  float64       
     8   waterfront     15809 non-null  float64       
     9   view           17704 non-null  float64       
     10  condition      17755 non-null  int64         
     11  grade          17755 non-null  int64         
     12  sqft_above     17755 non-null  int64         
     13  sqft_basement  17755 non-null  int64         
     14  yr_built       17755 non-null  int64         
     15  yr_renovated   17755 non-null  float64       
     16  zipcode        17755 non-null  int64         
     17  lat            17755 non-null  float64       
     18  long           17755 non-null  float64       
     19  sqft_living15  17755 non-null  int64         
     20  sqft_lot15     17755 non-null  int64         
     21  basement       17755 non-null  int64         
     22  renovated      17755 non-null  int64         
     23  yr_sold        17755 non-null  int64         
     24  home_age       17755 non-null  int64         
    dtypes: datetime64[ns](1), float64(8), int64(16)
    memory usage: 3.5 MB


>**OBSERVATIONS**
> - All data types seem good for the model

## Null Values


```python
#check for null values
df_scrub.isna().sum()
```




    id                  0
    date                0
    price               0
    bedrooms            0
    bathrooms           0
    sqft_living         0
    sqft_lot            0
    floors              0
    waterfront       1946
    view               51
    condition           0
    grade               0
    sqft_above          0
    sqft_basement       0
    yr_built            0
    yr_renovated        0
    zipcode             0
    lat                 0
    long                0
    sqft_living15       0
    sqft_lot15          0
    basement            0
    renovated           0
    yr_sold             0
    home_age            0
    dtype: int64



### `waterfront` Column


```python
#view values in waterfront column
df_scrub['waterfront'].value_counts(dropna=False)
```




    0.0    15688
    nan     1946
    1.0      121
    Name: waterfront, dtype: int64



> **OBSERVATIONS:**
> - `waterfront` has 11% null values.

> **ACTION**
> - I will explore how I can fill the nulls in the `waterfront` values


```python
#correlation of waterfront
df_scrub.corr()['waterfront']
```




    id              -0.0007271151406019631
    price              0.28061779508670853
    bedrooms        -0.0037972470805466155
    bathrooms          0.06895849489521352
    sqft_living        0.11490850577128048
    sqft_lot           0.02592756366361523
    floors            0.018991071659567576
    waterfront                         1.0
    view                0.4097734225678934
    condition         0.017035059923196618
    grade              0.08520732172037616
    sqft_above         0.07953782583182091
    sqft_basement      0.08954329499890055
    yr_built         -0.023441718330979473
    yr_renovated        0.0872436795030493
    zipcode           0.029702414769883254
    lat              -0.015771099373141872
    long              -0.04205519654265491
    sqft_living15      0.09252912301185473
    sqft_lot15         0.02969156480538231
    basement          0.042247678022296786
    renovated          0.08763626037688341
    yr_sold          -0.007725185091591844
    home_age          0.023317154750412086
    Name: waterfront, dtype: float64



> **OBSERVATIONS**
> - `waterfront` correlates most closely with `view` at a coefficient of 0.40

> **ACTIONS**
> - I will determine how i can utilize the `view` column to fill out the nulls in the `waterfall` column


```python
#number of waterfront properties in each view category
df_scrub.groupby('view')['waterfront'].sum()
```




    view
    0.0     0.0
    1.0     1.0
    2.0     6.0
    3.0    10.0
    4.0   103.0
    Name: waterfront, dtype: float64



> **OBSERVATIONS**
> - It seems that most of the waterfront homes also have a view ranking of 3 or 4


```python
#there are 19 null values in waterfront with a view of 4
df_scrub.loc[(df_scrub['waterfront'].isna()) & (df_scrub['view'] == 4)].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>basement</th>
      <th>renovated</th>
      <th>yr_sold</th>
      <th>home_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>582</th>
      <td>2998800125</td>
      <td>2014-07-01</td>
      <td>730,000.0</td>
      <td>2</td>
      <td>2.25</td>
      <td>2130</td>
      <td>4920</td>
      <td>1.5</td>
      <td>nan</td>
      <td>4.0</td>
      <td>4</td>
      <td>7</td>
      <td>1530</td>
      <td>600</td>
      <td>1941</td>
      <td>0.0</td>
      <td>98116</td>
      <td>47.573</td>
      <td>-122.40899999999999</td>
      <td>2130</td>
      <td>4920</td>
      <td>1</td>
      <td>0</td>
      <td>2014</td>
      <td>73</td>
    </tr>
    <tr>
      <th>1732</th>
      <td>913000340</td>
      <td>2015-01-02</td>
      <td>252,000.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>680</td>
      <td>1638</td>
      <td>1.0</td>
      <td>nan</td>
      <td>4.0</td>
      <td>1</td>
      <td>6</td>
      <td>680</td>
      <td>0</td>
      <td>1910</td>
      <td>1,992.0</td>
      <td>98116</td>
      <td>47.5832</td>
      <td>-122.399</td>
      <td>1010</td>
      <td>3621</td>
      <td>0</td>
      <td>1</td>
      <td>2015</td>
      <td>105</td>
    </tr>
    <tr>
      <th>2563</th>
      <td>7856400240</td>
      <td>2015-02-11</td>
      <td>1,650,000.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>3900</td>
      <td>9750</td>
      <td>1.0</td>
      <td>nan</td>
      <td>4.0</td>
      <td>5</td>
      <td>10</td>
      <td>2520</td>
      <td>1380</td>
      <td>1972</td>
      <td>0.0</td>
      <td>98006</td>
      <td>47.5605</td>
      <td>-122.15799999999999</td>
      <td>3410</td>
      <td>9450</td>
      <td>1</td>
      <td>0</td>
      <td>2015</td>
      <td>43</td>
    </tr>
    <tr>
      <th>3825</th>
      <td>8550001515</td>
      <td>2014-10-01</td>
      <td>429,592.0</td>
      <td>2</td>
      <td>2.75</td>
      <td>1992</td>
      <td>10946</td>
      <td>1.5</td>
      <td>nan</td>
      <td>4.0</td>
      <td>5</td>
      <td>6</td>
      <td>1288</td>
      <td>704</td>
      <td>1903</td>
      <td>0.0</td>
      <td>98070</td>
      <td>47.3551</td>
      <td>-122.475</td>
      <td>1110</td>
      <td>8328</td>
      <td>1</td>
      <td>0</td>
      <td>2014</td>
      <td>111</td>
    </tr>
    <tr>
      <th>4422</th>
      <td>7781600100</td>
      <td>2014-09-05</td>
      <td>1,340,000.0</td>
      <td>3</td>
      <td>2.75</td>
      <td>2730</td>
      <td>38869</td>
      <td>1.5</td>
      <td>nan</td>
      <td>4.0</td>
      <td>3</td>
      <td>9</td>
      <td>1940</td>
      <td>790</td>
      <td>1963</td>
      <td>2,001.0</td>
      <td>98146</td>
      <td>47.4857</td>
      <td>-122.361</td>
      <td>2630</td>
      <td>28188</td>
      <td>1</td>
      <td>1</td>
      <td>2014</td>
      <td>51</td>
    </tr>
  </tbody>
</table>
</div>




```python
#there are 54 null values in waterfront with a view of 3
df_scrub.loc[(df_scrub['waterfront'].isna()) & (df_scrub['view'] == 3)].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>basement</th>
      <th>renovated</th>
      <th>yr_sold</th>
      <th>home_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>60</th>
      <td>1516000055</td>
      <td>2014-12-10</td>
      <td>650,000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2150</td>
      <td>21235</td>
      <td>1.0</td>
      <td>nan</td>
      <td>3.0</td>
      <td>4</td>
      <td>8</td>
      <td>1590</td>
      <td>560</td>
      <td>1959</td>
      <td>0.0</td>
      <td>98166</td>
      <td>47.4336</td>
      <td>-122.339</td>
      <td>2570</td>
      <td>18900</td>
      <td>1</td>
      <td>0</td>
      <td>2014</td>
      <td>55</td>
    </tr>
    <tr>
      <th>216</th>
      <td>46100204</td>
      <td>2015-02-21</td>
      <td>1,510,000.0</td>
      <td>5</td>
      <td>3.0</td>
      <td>3300</td>
      <td>33474</td>
      <td>1.0</td>
      <td>nan</td>
      <td>3.0</td>
      <td>3</td>
      <td>9</td>
      <td>1870</td>
      <td>1430</td>
      <td>1957</td>
      <td>1,991.0</td>
      <td>98040</td>
      <td>47.5673</td>
      <td>-122.21</td>
      <td>3836</td>
      <td>20953</td>
      <td>1</td>
      <td>1</td>
      <td>2015</td>
      <td>58</td>
    </tr>
    <tr>
      <th>527</th>
      <td>3225079035</td>
      <td>2014-06-18</td>
      <td>1,600,000.0</td>
      <td>6</td>
      <td>5.0</td>
      <td>6050</td>
      <td>230652</td>
      <td>2.0</td>
      <td>nan</td>
      <td>3.0</td>
      <td>3</td>
      <td>11</td>
      <td>6050</td>
      <td>0</td>
      <td>2001</td>
      <td>0.0</td>
      <td>98024</td>
      <td>47.6033</td>
      <td>-121.943</td>
      <td>4210</td>
      <td>233971</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>13</td>
    </tr>
    <tr>
      <th>707</th>
      <td>4022907770</td>
      <td>2014-10-14</td>
      <td>550,000.0</td>
      <td>4</td>
      <td>1.75</td>
      <td>2480</td>
      <td>14782</td>
      <td>1.0</td>
      <td>nan</td>
      <td>3.0</td>
      <td>3</td>
      <td>8</td>
      <td>1460</td>
      <td>1020</td>
      <td>1958</td>
      <td>0.0</td>
      <td>98155</td>
      <td>47.7646</td>
      <td>-122.271</td>
      <td>2910</td>
      <td>10800</td>
      <td>1</td>
      <td>0</td>
      <td>2014</td>
      <td>56</td>
    </tr>
    <tr>
      <th>830</th>
      <td>2061100570</td>
      <td>2015-02-10</td>
      <td>595,000.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>1910</td>
      <td>5753</td>
      <td>1.0</td>
      <td>nan</td>
      <td>3.0</td>
      <td>3</td>
      <td>8</td>
      <td>1110</td>
      <td>800</td>
      <td>1941</td>
      <td>0.0</td>
      <td>98115</td>
      <td>47.6898</td>
      <td>-122.32700000000001</td>
      <td>1630</td>
      <td>5580</td>
      <td>1</td>
      <td>0</td>
      <td>2015</td>
      <td>74</td>
    </tr>
  </tbody>
</table>
</div>



> **ACTIONS**
> - Fill in the null waterfront value when the view is 3 or 4


```python
#fill in null where view is 4
df_scrub.loc[(df_scrub['waterfront'].isna()) & (df_scrub['view'] == 4),['waterfront']] = 1
```


```python
#fill in null where view is 3
df_scrub.loc[(df_scrub['waterfront'].isna()) & (df_scrub['view'] == 3),['waterfront']] = 1
```


```python
#check the changes
df_scrub.loc[(df_scrub['waterfront'].isna()) & (df_scrub['view'] == 4)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>basement</th>
      <th>renovated</th>
      <th>yr_sold</th>
      <th>home_age</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
#check the changes
df_scrub.loc[(df_scrub['waterfront'].isna()) & (df_scrub['view'] == 3)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>basement</th>
      <th>renovated</th>
      <th>yr_sold</th>
      <th>home_age</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
#view values in waterfront column
df_scrub['waterfront'].value_counts(dropna=False)/len(df_scrub)
```




    0.0     0.8835820895522388
    nan    0.10537876654463531
    1.0   0.011039143903125881
    Name: waterfront, dtype: float64



>**OBSERVATIONS**
> - The number of nulls in the `waterfront` column is still 10.5%

>**ACTIONS**
> - I will convert the rest of the nulls to zeros as they do not seem to have any other indicators of being a waterfront property


```python
#convert waterfront null values to 0
df_scrub.loc[df_scrub['waterfront'].isna(),['waterfront']] = 0
```


```python
#check waterfront values
df_scrub['waterfront'].value_counts(dropna=False)
```




    0.0    17559
    1.0      196
    Name: waterfront, dtype: int64




```python
#print number of rows in dataframe
print(f'The dataframe now has {len(df_scrub)} many rows.')
```

    The dataframe now has 17755 many rows.


### `view` Column


```python
#view the values of the view column
df_scrub['view'].value_counts(dropna=False)
```




    0.0    15972
    2.0      792
    3.0      403
    1.0      277
    4.0      260
    nan       51
    Name: view, dtype: int64



> **ACTIONS**
> - I will drop the 39 null values


```python
#drop rows
df_scrub.dropna(subset=['view'], inplace=True)
df_scrub['view'].isna().sum()
```




    0




```python
#check the view column
df_scrub['view'].value_counts(dropna=False)
```




    0.0    15972
    2.0      792
    3.0      403
    1.0      277
    4.0      260
    Name: view, dtype: int64




```python
#recheck null values in dataframe
df_scrub.isna().sum()
```




    id               0
    date             0
    price            0
    bedrooms         0
    bathrooms        0
    sqft_living      0
    sqft_lot         0
    floors           0
    waterfront       0
    view             0
    condition        0
    grade            0
    sqft_above       0
    sqft_basement    0
    yr_built         0
    yr_renovated     0
    zipcode          0
    lat              0
    long             0
    sqft_living15    0
    sqft_lot15       0
    basement         0
    renovated        0
    yr_sold          0
    home_age         0
    dtype: int64



## Duplicates

### Duplicates for `id`


```python
#check for duplicates
duplicate_id = df_scrub.loc[df_scrub.duplicated(subset='id', keep=False) == True].sort_values('id').head(50)
duplicate_id.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>basement</th>
      <th>renovated</th>
      <th>yr_sold</th>
      <th>home_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2494</th>
      <td>1000102</td>
      <td>2014-09-16</td>
      <td>280,000.0</td>
      <td>6</td>
      <td>3.0</td>
      <td>2400</td>
      <td>9373</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2400</td>
      <td>0</td>
      <td>1991</td>
      <td>0.0</td>
      <td>98002</td>
      <td>47.3262</td>
      <td>-122.214</td>
      <td>2060</td>
      <td>7316</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>23</td>
    </tr>
    <tr>
      <th>2495</th>
      <td>1000102</td>
      <td>2015-04-22</td>
      <td>300,000.0</td>
      <td>6</td>
      <td>3.0</td>
      <td>2400</td>
      <td>9373</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2400</td>
      <td>0</td>
      <td>1991</td>
      <td>0.0</td>
      <td>98002</td>
      <td>47.3262</td>
      <td>-122.214</td>
      <td>2060</td>
      <td>7316</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>24</td>
    </tr>
    <tr>
      <th>11421</th>
      <td>109200390</td>
      <td>2014-08-20</td>
      <td>245,000.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>1480</td>
      <td>3900</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>7</td>
      <td>1480</td>
      <td>0</td>
      <td>1980</td>
      <td>0.0</td>
      <td>98023</td>
      <td>47.2977</td>
      <td>-122.367</td>
      <td>1830</td>
      <td>6956</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>34</td>
    </tr>
    <tr>
      <th>11422</th>
      <td>109200390</td>
      <td>2014-10-20</td>
      <td>250,000.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>1480</td>
      <td>3900</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>7</td>
      <td>1480</td>
      <td>0</td>
      <td>1980</td>
      <td>0.0</td>
      <td>98023</td>
      <td>47.2977</td>
      <td>-122.367</td>
      <td>1830</td>
      <td>6956</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>34</td>
    </tr>
    <tr>
      <th>7786</th>
      <td>251300110</td>
      <td>2015-01-14</td>
      <td>358,000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2510</td>
      <td>12013</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2510</td>
      <td>0</td>
      <td>1988</td>
      <td>0.0</td>
      <td>98003</td>
      <td>47.3473</td>
      <td>-122.314</td>
      <td>1870</td>
      <td>8017</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>




```python
#check duplicates
df_scrub.loc[df_scrub['id'] == 4139480200]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>basement</th>
      <th>renovated</th>
      <th>yr_sold</th>
      <th>home_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>313</th>
      <td>4139480200</td>
      <td>2014-06-18</td>
      <td>1,380,000.0</td>
      <td>4</td>
      <td>3.25</td>
      <td>4290</td>
      <td>12103</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3</td>
      <td>11</td>
      <td>2690</td>
      <td>1600</td>
      <td>1997</td>
      <td>0.0</td>
      <td>98006</td>
      <td>47.5503</td>
      <td>-122.102</td>
      <td>3860</td>
      <td>11244</td>
      <td>1</td>
      <td>0</td>
      <td>2014</td>
      <td>17</td>
    </tr>
    <tr>
      <th>314</th>
      <td>4139480200</td>
      <td>2014-12-09</td>
      <td>1,400,000.0</td>
      <td>4</td>
      <td>3.25</td>
      <td>4290</td>
      <td>12103</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3</td>
      <td>11</td>
      <td>2690</td>
      <td>1600</td>
      <td>1997</td>
      <td>0.0</td>
      <td>98006</td>
      <td>47.5503</td>
      <td>-122.102</td>
      <td>3860</td>
      <td>11244</td>
      <td>1</td>
      <td>0</td>
      <td>2014</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>



> **OBSERVATOINS**
> - Duplicates in the `id` column seem to represent multiple sales of the same house. 

> **ACTIONS**
> - I will consider these duplicates as separate homes and keep them in the dataset. The column `id` will be removed later.

## Column Drop


```python
#look at columns
df_scrub.columns
```




    Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
           'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
           'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
           'lat', 'long', 'sqft_living15', 'sqft_lot15', 'basement', 'renovated',
           'yr_sold', 'home_age'],
          dtype='object')



### The `sqft_basement` Column

The `sqft_basement` column can be eliminated now that I have a column which represents whether or not a house has a basement.


```python
#drop the sqft_basement column
df_scrub.drop(columns='sqft_basement', inplace=True)
df_scrub.columns
```




    Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
           'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
           'sqft_above', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
           'sqft_living15', 'sqft_lot15', 'basement', 'renovated', 'yr_sold',
           'home_age'],
          dtype='object')



### The `sqft_living15` and `sqft_lot15` Columns

The `sqft_living15` and `sqft_lot15` columns do not seem to be relevant for predicting home listing prices. I will remove these.


```python
#drop columns
df_scrub.drop(columns=['sqft_living15','sqft_lot15'], inplace=True)
df_scrub.columns
```




    Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
           'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
           'sqft_above', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
           'basement', 'renovated', 'yr_sold', 'home_age'],
          dtype='object')



### `yr_renovated` Column


```python
#drop the yr_renovated column
df_scrub.drop(columns='yr_renovated', inplace=True)
```

### `id` Column


```python
#drop the id column
df_scrub.drop(columns='id', inplace=True)
```

## State of Dataframe


```python
#state of the dataframe
df_scrub
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>basement</th>
      <th>renovated</th>
      <th>yr_sold</th>
      <th>home_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-10-13</td>
      <td>221,900.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>1955</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-12-09</td>
      <td>538,000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>1951</td>
      <td>98125</td>
      <td>47.721000000000004</td>
      <td>-122.319</td>
      <td>1</td>
      <td>1</td>
      <td>2014</td>
      <td>63</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-12-09</td>
      <td>604,000.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>1965</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.39299999999999</td>
      <td>1</td>
      <td>0</td>
      <td>2014</td>
      <td>49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-02-18</td>
      <td>510,000.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>1987</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>28</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2014-05-12</td>
      <td>1,230,000.0</td>
      <td>4</td>
      <td>4.5</td>
      <td>5420</td>
      <td>101930</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>11</td>
      <td>3890</td>
      <td>2001</td>
      <td>98053</td>
      <td>47.6561</td>
      <td>-122.005</td>
      <td>1</td>
      <td>0</td>
      <td>2014</td>
      <td>13</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>2014-05-21</td>
      <td>360,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1530</td>
      <td>2009</td>
      <td>98103</td>
      <td>47.6993</td>
      <td>-122.346</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>5</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>2015-02-23</td>
      <td>400,000.0</td>
      <td>4</td>
      <td>2.5</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2310</td>
      <td>2014</td>
      <td>98146</td>
      <td>47.5107</td>
      <td>-122.36200000000001</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>2014-06-23</td>
      <td>402,101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>2009</td>
      <td>98144</td>
      <td>47.5944</td>
      <td>-122.29899999999999</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>5</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>2015-01-16</td>
      <td>400,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1600</td>
      <td>2004</td>
      <td>98027</td>
      <td>47.5345</td>
      <td>-122.069</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>11</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>2014-10-15</td>
      <td>325,000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>2008</td>
      <td>98144</td>
      <td>47.5941</td>
      <td>-122.29899999999999</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>17704 rows × 20 columns</p>
</div>



# Explore

I will now explore the dataset after initial scrubbing. I will investigate linearity and multicollinearity and correct any issues before modeling.


```python
#create a copy of the scrub dataframe
df_explore = df_scrub.copy()
df_explore.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>basement</th>
      <th>renovated</th>
      <th>yr_sold</th>
      <th>home_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-10-13</td>
      <td>221,900.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>1955</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-12-09</td>
      <td>538,000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>1951</td>
      <td>98125</td>
      <td>47.721000000000004</td>
      <td>-122.319</td>
      <td>1</td>
      <td>1</td>
      <td>2014</td>
      <td>63</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-12-09</td>
      <td>604,000.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>1965</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.39299999999999</td>
      <td>1</td>
      <td>0</td>
      <td>2014</td>
      <td>49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-02-18</td>
      <td>510,000.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>1987</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>28</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2014-05-12</td>
      <td>1,230,000.0</td>
      <td>4</td>
      <td>4.5</td>
      <td>5420</td>
      <td>101930</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>11</td>
      <td>3890</td>
      <td>2001</td>
      <td>98053</td>
      <td>47.6561</td>
      <td>-122.005</td>
      <td>1</td>
      <td>0</td>
      <td>2014</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>



## Linearity

### `bedrooms`


```python
#check linearity between bedrooms and price
sns.lmplot(data=df_explore, x='bedrooms', y='price', height=8);
```


    
![png](output_121_0.png)
    



```python
#view price vs bedrooms another way
sns.catplot(data=df_explore, x='bedrooms', y='price', height=8);
```


    
![png](output_122_0.png)
    


> **OBSERVATIONS**
> - There seems to be a positive linear relationship between the number of bedrooms and the price of the home for homes with a 1-5 bedrooms. Homes with 6+ bedrooms seem to be valued at a lower price. 
> - I notice some outliers that I will need to remove.

### `bathrooms`


```python
#check linearity between bathrooms and price
sns.lmplot(data=df_explore, x='bathrooms', y='price', height=8);
```


    
![png](output_125_0.png)
    


> **OBSERVATIONS**
> - There seems to be a positive linear relationship between `bathrooms` and `price`.

### `sqft_living`


```python
#check linearity between sqft_living and price
sns.lmplot(data=df_explore, x='sqft_living', y='price', height=8);
```


    
![png](output_128_0.png)
    


> **OBSERVATIONS**
> - There seems to be a strong positive linear relationship between `sqft_living` and `price`.

### `sqft_lot`


```python
#check linearity between sqft_lot and price
sns.lmplot(data=df_explore, x='sqft_lot', y='price', height=8)
```




    <seaborn.axisgrid.FacetGrid at 0x7f92e0a6c3d0>




    
![png](output_131_1.png)
    


> **OBSERVATIONS**
> - There seems to be a linear relationship between `sqft_lot` and `price`. However, there seems to be 2 types of high value homes, 1) very small lot homes with high prices and 2) large lot homes with high prices

### `floors`


```python
#check linearity between floors and price
sns.lmplot(data=df_explore, x='floors', y='price', height=8);
```


    
![png](output_134_0.png)
    


> **OBSERVATIONS**
> - There seems to be a linear relationship between `floors` and `price`.

### `grade`


```python
#check linearity between grade and price
sns.lmplot(data=df_explore, x='grade', y='price', height=8);
```


    
![png](output_137_0.png)
    


> **OBSERVATIONS**
> - There seems to be a linear relationship between `grade` and `price`.

### `condition`


```python
#check linearity between condition and price
sns.lmplot(data=df_explore, x='condition', y='price', height=8);
```


    
![png](output_140_0.png)
    



```python
#check relationship another way
sns.catplot(data=df_explore, x='condition', y='price', height=8);
```


    
![png](output_141_0.png)
    


> **OBSERVATIONS**
> - There seems to be a linear relationship between `condition` and `price`, however, there seems to be a sweet spot around 3 i.e. not any additional value to condition 4 and 5. I will test this later

### `home_age`


```python
#check linearity between condition and price
sns.lmplot(data=df_explore, x='home_age', y='price', height=8);
```


    
![png](output_144_0.png)
    


> **OBSERVATIONS**
> - There seems to be a linear relationship between `home_age` and `price`, however, it seems like it is a neutral relationship.

## Multicollinearity

I want to check to see if the independent variables are truly independent from each other by checking for multicollinearity.

### Two Variable Multicollinearity Check


```python
#remove lat and long columns
df_explore.drop(columns=['lat','long'], inplace=True)
```


```python
#create and plot correlations
corr = df_explore.corr()
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(corr, cmap='Reds', annot=True, ax=ax);
```


    
![png](output_150_0.png)
    


> **ACTIONS**
> - Remove `sqft_above` as it correlates very closesly with `sqft_living`


```python
#remove sqft_above
df_explore.drop(columns='sqft_above', inplace=True)
```


```python
corr = df_explore.corr()
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(corr, cmap='Reds', annot=True, ax=ax);
```


    
![png](output_153_0.png)
    


> **OBSERVATIONS**
> - There are no more variables which correlate above .75, therefore, variables are now considered independent.

## Outlier Removal

### `bedrooms`


```python
#check linearity between bedrooms and price
sns.catplot(data=df_explore, x='bedrooms', y='price', height=8);
```


    
![png](output_157_0.png)
    


> **OBSERVATIONS**
> - I believe a single model will struggle with accurately predicting homes with less than 6 homes with homes with 6 or more bedrooms. 

>**ACTIONS**
> - I will keep only homes with 5 or fewer bedrooms


```python
#remove outliers
df_explore = df_explore.loc[df_explore['bedrooms'] <= 5]
len(df_explore)
```




    17424



# Model


```python
#create a copy of the explore dataframe
df_model_base = df_explore.copy()
df_model_base
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>zipcode</th>
      <th>basement</th>
      <th>renovated</th>
      <th>yr_sold</th>
      <th>home_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-10-13</td>
      <td>221,900.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1955</td>
      <td>98178</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-12-09</td>
      <td>538,000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1951</td>
      <td>98125</td>
      <td>1</td>
      <td>1</td>
      <td>2014</td>
      <td>63</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-12-09</td>
      <td>604,000.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1965</td>
      <td>98136</td>
      <td>1</td>
      <td>0</td>
      <td>2014</td>
      <td>49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-02-18</td>
      <td>510,000.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1987</td>
      <td>98074</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>28</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2014-05-12</td>
      <td>1,230,000.0</td>
      <td>4</td>
      <td>4.5</td>
      <td>5420</td>
      <td>101930</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>11</td>
      <td>2001</td>
      <td>98053</td>
      <td>1</td>
      <td>0</td>
      <td>2014</td>
      <td>13</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>2014-05-21</td>
      <td>360,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2009</td>
      <td>98103</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>5</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>2015-02-23</td>
      <td>400,000.0</td>
      <td>4</td>
      <td>2.5</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2014</td>
      <td>98146</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>2014-06-23</td>
      <td>402,101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2009</td>
      <td>98144</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>5</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>2015-01-16</td>
      <td>400,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2004</td>
      <td>98027</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>11</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>2014-10-15</td>
      <td>325,000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2008</td>
      <td>98144</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>17424 rows × 17 columns</p>
</div>



## Model Preprocessing

### Column Drop

#### `yr_built` Column

I will be removing `yr_built` as it is related to the new column I created named `home_age`


```python
#drop the yr_built column
df_model_base.drop(columns='yr_built', inplace=True)
```

#### `date` Column

The `date` column represents the sale date which I do not think is relevant to the model's output since it is a datetime object.


```python
#drop the date column
df_model_base.drop(columns='date', inplace=True)
```

#### `yr_sold` Column

The `yr_sold` column I created in order to create the `home_age` column. It is no longer needed.


```python
#drop the yr_sold column
df_model_base.drop(columns='yr_sold', inplace=True)
```

## Model 1

- The data is now ready for the first model run. So far, I have taken the following steps:
    1. Removed irrelevant columns
    2. Removed some outliers in the raw data
    3. Removed columns due to 2-variable multicollinearity
    

### Model Creation

I will now create the initial model by copying the df_model_original dataframe.


```python
#create a new model dataframe
df_model_1 = df_model_base.copy()
df_model_1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>zipcode</th>
      <th>basement</th>
      <th>renovated</th>
      <th>home_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221,900.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>98178</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538,000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>98125</td>
      <td>1</td>
      <td>1</td>
      <td>63</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604,000.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>98136</td>
      <td>1</td>
      <td>0</td>
      <td>49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510,000.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>98074</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1,230,000.0</td>
      <td>4</td>
      <td>4.5</td>
      <td>5420</td>
      <td>101930</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>11</td>
      <td>98053</td>
      <td>1</td>
      <td>0</td>
      <td>13</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>360,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>98103</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>400,000.0</td>
      <td>4</td>
      <td>2.5</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>98146</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>402,101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>98144</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>400,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>98027</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>325,000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>98144</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>17424 rows × 14 columns</p>
</div>




```python
#define indpendent and dependent variables
x_cols = df_model_1.drop(columns='price').columns
y_col = 'price'
#run funciton to create model and check assumptions
model_1 = fit_new_model(df_model_1, x_cols=x_cols, y_col=y_col, norm=True, diagnose=False)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>zipcode</th>
      <th>basement</th>
      <th>renovated</th>
      <th>home_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221,900.0</td>
      <td>-0.39834574796776623</td>
      <td>-1.4761162000797934</td>
      <td>-0.9902889880360612</td>
      <td>-0.2272499267847439</td>
      <td>-0.9172259379812189</td>
      <td>-0.1052746428490549</td>
      <td>-0.30183497242494894</td>
      <td>-0.6284335139334112</td>
      <td>-0.5616717425173167</td>
      <td>1.8754582283911905</td>
      <td>-0.7960646341307491</td>
      <td>-0.2070054668415114</td>
      <td>0.5433732623906105</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538,000.0</td>
      <td>-0.39834574796776623</td>
      <td>0.19611041462901635</td>
      <td>0.5685541395261385</td>
      <td>-0.18943877133467454</td>
      <td>0.9409004128366808</td>
      <td>-0.1052746428490549</td>
      <td>-0.30183497242494894</td>
      <td>-0.6284335139334112</td>
      <td>-0.5616717425173167</td>
      <td>0.8843946473268516</td>
      <td>1.2561073121770143</td>
      <td>4.830513044675938</td>
      <td>0.6797630490616083</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604,000.0</td>
      <td>0.8097963429966133</td>
      <td>1.199446383454302</td>
      <td>-0.1155424847997189</td>
      <td>-0.24268789854513656</td>
      <td>-0.9172259379812189</td>
      <td>-0.1052746428490549</td>
      <td>-0.30183497242494894</td>
      <td>2.4391741473008666</td>
      <td>-0.5616717425173167</td>
      <td>1.0900870886798275</td>
      <td>1.2561073121770143</td>
      <td>-0.2070054668415114</td>
      <td>0.20239879571311556</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510,000.0</td>
      <td>-0.39834574796776623</td>
      <td>-0.13833490831274559</td>
      <td>-0.4295540500640469</td>
      <td>-0.16953566312666063</td>
      <td>-0.9172259379812189</td>
      <td>-0.1052746428490549</td>
      <td>-0.30183497242494894</td>
      <td>-0.6284335139334112</td>
      <td>0.2967991734144139</td>
      <td>-0.06927030803694632</td>
      <td>-0.7960646341307491</td>
      <td>-0.2070054668415114</td>
      <td>-0.5136475843096237</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1,230,000.0</td>
      <td>0.8097963429966133</td>
      <td>3.206118321104874</td>
      <td>3.76474328596662</td>
      <td>2.0594699518161854</td>
      <td>-0.9172259379812189</td>
      <td>-0.1052746428490549</td>
      <td>-0.30183497242494894</td>
      <td>-0.6284335139334112</td>
      <td>2.8722119212096056</td>
      <td>-0.4619558778926278</td>
      <td>1.2561073121770143</td>
      <td>-0.2070054668415114</td>
      <td>-1.025109284325866</td>
    </tr>
  </tbody>
</table>
</div>


    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.645
    Model:                            OLS   Adj. R-squared:                  0.645
    Method:                 Least Squares   F-statistic:                     2431.
    Date:                Sat, 17 Apr 2021   Prob (F-statistic):               0.00
    Time:                        23:40:35   Log-Likelihood:            -2.3834e+05
    No. Observations:               17424   AIC:                         4.767e+05
    Df Residuals:                   17410   BIC:                         4.768e+05
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    Intercept    5.353e+05   1599.894    334.574      0.000    5.32e+05    5.38e+05
    bedrooms    -3.689e+04   2065.413    -17.863      0.000   -4.09e+04   -3.28e+04
    bathrooms    3.389e+04   2939.233     11.529      0.000    2.81e+04    3.96e+04
    sqft_living  1.511e+05   3296.690     45.839      0.000    1.45e+05    1.58e+05
    sqft_lot    -1.045e+04   1649.271     -6.335      0.000   -1.37e+04   -7216.189
    floors       1.394e+04   2174.766      6.408      0.000    9673.075    1.82e+04
    waterfront   4.106e+04   1817.782     22.589      0.000    3.75e+04    4.46e+04
    view         3.364e+04   1927.367     17.455      0.000    2.99e+04    3.74e+04
    condition    1.192e+04   1783.540      6.684      0.000    8425.023    1.54e+04
    grade        1.419e+05   2759.328     51.436      0.000    1.37e+05    1.47e+05
    zipcode     -2002.1504   1810.953     -1.106      0.269   -5551.799    1547.498
    basement     2824.2373   1869.360      1.511      0.131    -839.896    6488.371
    renovated    4977.5044   1709.075      2.912      0.004    1627.546    8327.462
    home_age     1.026e+05   2405.692     42.630      0.000    9.78e+04    1.07e+05
    ==============================================================================
    Omnibus:                    11594.178   Durbin-Watson:                   1.975
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           530871.059
    Skew:                           2.616   Prob(JB):                         0.00
    Kurtosis:                      29.530   Cond. No.                         4.89
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    
    



```python
#create dataframe of feature coefficients
coefficients_m1_df = pd.DataFrame(model_1.params, columns=['Coefficient'])
coefficients_m1_df.drop('Intercept', inplace=True)
coefficients_m1_df = coefficients_m1_df.sort_values(by='Coefficient')
```


```python
#bar plot showing coefficients
fig, ax = plt.subplots(figsize=(15,20))
sns.barplot(data = coefficients_m1_df, y=coefficients_m1_df.index, x='Coefficient', ax=ax, orient='h');
```


    
![png](output_180_0.png)
    


### Model Interpretation

>**OBSERVATOINS**
> - Adjusted R-Squared of 0.645
> - All features with significant p-values except for `zipcode` and `basement`
> - The most positively correlated features to price are `sqft_living`, `grade` and `home_age`
> - The most negatively correlated features to price are `bedrooms`, `zipcode` and `sqft_lot`
> - No multicollinearity found
> - Residuals not normal on the high end of the distribution
> - I am seeing heteroscedasticity along the bottom edge plus as the price gets higher

> **ACTIONS**
> - I will look at one hot encoding `zipcode`

### Model Tuning

#### OHE Columns

I will evaluate `zipcode` and `grade` for OHE in order to better model this feature.


```python
#check column names
df_model_base.columns
```




    Index(['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
           'waterfront', 'view', 'condition', 'grade', 'zipcode', 'basement',
           'renovated', 'home_age'],
          dtype='object')




```python
#investigate zipcode
get_plots(df_model_base,'zipcode')
```

    count             17,424.0
    mean    98,077.70443067035
    std      53.47790092647879
    min               98,001.0
    25%               98,033.0
    50%               98,065.0
    75%               98,117.0
    max               98,199.0
    Name: zipcode, dtype: float64



    
![png](output_187_1.png)
    



```python
#investigate zipcode
get_plots(df_model_base,'grade')
```

    count            17,424.0
    mean     7.65426997245179
    std     1.164861827514171
    min                   3.0
    25%                   7.0
    50%                   7.0
    75%                   8.0
    max                  13.0
    Name: grade, dtype: float64



    
![png](output_188_1.png)
    


> **OBSERVATIONS**
> - `grade` seems to be categorical and needs to be hot-one encoded to improve the model since it has a high coefficient.


```python
#fit the data
cat_zipcode = ['zipcode']
encoder = OneHotEncoder(drop='first', sparse=False)
encoder.fit(df_model_base[cat_zipcode])
```




    OneHotEncoder(drop='first', sparse=False)




```python
#transform the data
ohe_vars = encoder.transform(df_model_base[cat_zipcode])
ohe_vars
```




    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])




```python
#get the features
encoder.get_feature_names(cat_zipcode)
```




    array(['zipcode_98002', 'zipcode_98003', 'zipcode_98004', 'zipcode_98005',
           'zipcode_98006', 'zipcode_98007', 'zipcode_98008', 'zipcode_98010',
           'zipcode_98011', 'zipcode_98014', 'zipcode_98019', 'zipcode_98022',
           'zipcode_98023', 'zipcode_98024', 'zipcode_98027', 'zipcode_98028',
           'zipcode_98029', 'zipcode_98030', 'zipcode_98031', 'zipcode_98032',
           'zipcode_98033', 'zipcode_98034', 'zipcode_98038', 'zipcode_98039',
           'zipcode_98040', 'zipcode_98042', 'zipcode_98045', 'zipcode_98052',
           'zipcode_98053', 'zipcode_98055', 'zipcode_98056', 'zipcode_98058',
           'zipcode_98059', 'zipcode_98065', 'zipcode_98070', 'zipcode_98072',
           'zipcode_98074', 'zipcode_98075', 'zipcode_98077', 'zipcode_98092',
           'zipcode_98102', 'zipcode_98103', 'zipcode_98105', 'zipcode_98106',
           'zipcode_98107', 'zipcode_98108', 'zipcode_98109', 'zipcode_98112',
           'zipcode_98115', 'zipcode_98116', 'zipcode_98117', 'zipcode_98118',
           'zipcode_98119', 'zipcode_98122', 'zipcode_98125', 'zipcode_98126',
           'zipcode_98133', 'zipcode_98136', 'zipcode_98144', 'zipcode_98146',
           'zipcode_98148', 'zipcode_98155', 'zipcode_98166', 'zipcode_98168',
           'zipcode_98177', 'zipcode_98178', 'zipcode_98188', 'zipcode_98198',
           'zipcode_98199'], dtype=object)




```python
#convert to dataframe
df_cat_zipcode = pd.DataFrame(ohe_vars, columns=encoder.get_feature_names(cat_zipcode), index=df_model_base.index)
df_cat_zipcode
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>zipcode_98019</th>
      <th>zipcode_98022</th>
      <th>zipcode_98023</th>
      <th>zipcode_98024</th>
      <th>zipcode_98027</th>
      <th>...</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>17424 rows × 69 columns</p>
</div>




```python
#concat original dataframe to zipcode dataframe and prepare for model number 2
df_model_base = pd.concat([df_model_base.drop(['zipcode'], axis=1), df_cat_zipcode], axis=1)
df_model_base
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>basement</th>
      <th>renovated</th>
      <th>home_age</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>...</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221,900.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538,000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>63</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604,000.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>49</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510,000.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1,230,000.0</td>
      <td>4</td>
      <td>4.5</td>
      <td>5420</td>
      <td>101930</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>13</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>360,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>400,000.0</td>
      <td>4</td>
      <td>2.5</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>402,101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>400,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>325,000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>17424 rows × 82 columns</p>
</div>



## Model 2

Going to refit the model with the new OHE `zipcode` columns


```python
#copy the base model with the OHE column
df_model_2 = df_model_base.copy()
df_model_2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>basement</th>
      <th>renovated</th>
      <th>home_age</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>...</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221,900.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538,000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>63</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604,000.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>49</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510,000.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1,230,000.0</td>
      <td>4</td>
      <td>4.5</td>
      <td>5420</td>
      <td>101930</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>13</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>360,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>400,000.0</td>
      <td>4</td>
      <td>2.5</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>402,101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>400,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>325,000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>17424 rows × 82 columns</p>
</div>



### Model Creation


```python
#define indpendent and dependent variables
x_cols = df_model_2.drop(columns='price').columns
y_col = 'price'
#run funciton to create model and check assumptions
model_2 = fit_new_model(df_model_2, x_cols=x_cols, y_col=y_col, norm=True, diagnose=False)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>basement</th>
      <th>renovated</th>
      <th>home_age</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>...</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221,900.0</td>
      <td>-0.39834574796776623</td>
      <td>-1.4761162000797934</td>
      <td>-0.9902889880360612</td>
      <td>-0.2272499267847439</td>
      <td>-0.9172259379812189</td>
      <td>-0.1052746428490549</td>
      <td>-0.30183497242494894</td>
      <td>-0.6284335139334112</td>
      <td>-0.5616717425173167</td>
      <td>-0.7960646341307491</td>
      <td>-0.2070054668415114</td>
      <td>0.5433732623906105</td>
      <td>-0.09596277715461221</td>
      <td>-0.11717389485325336</td>
      <td>...</td>
      <td>-0.13699474409500317</td>
      <td>-0.12802573696800537</td>
      <td>-0.15367218648320422</td>
      <td>-0.11229613905271121</td>
      <td>-0.1294079886377916</td>
      <td>-0.11590914317994902</td>
      <td>-0.05200547477121165</td>
      <td>-0.1423334846112359</td>
      <td>-0.10910898221206813</td>
      <td>-0.11359901159350409</td>
      <td>-0.10748168344892405</td>
      <td>9.075444548655437</td>
      <td>-0.07897245267878145</td>
      <td>-0.1166695512602654</td>
      <td>-0.1213808022523802</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538,000.0</td>
      <td>-0.39834574796776623</td>
      <td>0.19611041462901635</td>
      <td>0.5685541395261385</td>
      <td>-0.18943877133467454</td>
      <td>0.9409004128366808</td>
      <td>-0.1052746428490549</td>
      <td>-0.30183497242494894</td>
      <td>-0.6284335139334112</td>
      <td>-0.5616717425173167</td>
      <td>1.2561073121770143</td>
      <td>4.830513044675938</td>
      <td>0.6797630490616083</td>
      <td>-0.09596277715461221</td>
      <td>-0.11717389485325336</td>
      <td>...</td>
      <td>7.299131178370216</td>
      <td>-0.12802573696800537</td>
      <td>-0.15367218648320422</td>
      <td>-0.11229613905271121</td>
      <td>-0.1294079886377916</td>
      <td>-0.11590914317994902</td>
      <td>-0.05200547477121165</td>
      <td>-0.1423334846112359</td>
      <td>-0.10910898221206813</td>
      <td>-0.11359901159350409</td>
      <td>-0.10748168344892405</td>
      <td>-0.11018111592616825</td>
      <td>-0.07897245267878145</td>
      <td>-0.1166695512602654</td>
      <td>-0.1213808022523802</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604,000.0</td>
      <td>0.8097963429966133</td>
      <td>1.199446383454302</td>
      <td>-0.1155424847997189</td>
      <td>-0.24268789854513656</td>
      <td>-0.9172259379812189</td>
      <td>-0.1052746428490549</td>
      <td>-0.30183497242494894</td>
      <td>2.4391741473008666</td>
      <td>-0.5616717425173167</td>
      <td>1.2561073121770143</td>
      <td>-0.2070054668415114</td>
      <td>0.20239879571311556</td>
      <td>-0.09596277715461221</td>
      <td>-0.11717389485325336</td>
      <td>...</td>
      <td>-0.13699474409500317</td>
      <td>-0.12802573696800537</td>
      <td>-0.15367218648320422</td>
      <td>8.90451458377881</td>
      <td>-0.1294079886377916</td>
      <td>-0.11590914317994902</td>
      <td>-0.05200547477121165</td>
      <td>-0.1423334846112359</td>
      <td>-0.10910898221206813</td>
      <td>-0.11359901159350409</td>
      <td>-0.10748168344892405</td>
      <td>-0.11018111592616825</td>
      <td>-0.07897245267878145</td>
      <td>-0.1166695512602654</td>
      <td>-0.1213808022523802</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510,000.0</td>
      <td>-0.39834574796776623</td>
      <td>-0.13833490831274559</td>
      <td>-0.4295540500640469</td>
      <td>-0.16953566312666063</td>
      <td>-0.9172259379812189</td>
      <td>-0.1052746428490549</td>
      <td>-0.30183497242494894</td>
      <td>-0.6284335139334112</td>
      <td>0.2967991734144139</td>
      <td>-0.7960646341307491</td>
      <td>-0.2070054668415114</td>
      <td>-0.5136475843096237</td>
      <td>-0.09596277715461221</td>
      <td>-0.11717389485325336</td>
      <td>...</td>
      <td>-0.13699474409500317</td>
      <td>-0.12802573696800537</td>
      <td>-0.15367218648320422</td>
      <td>-0.11229613905271121</td>
      <td>-0.1294079886377916</td>
      <td>-0.11590914317994902</td>
      <td>-0.05200547477121165</td>
      <td>-0.1423334846112359</td>
      <td>-0.10910898221206813</td>
      <td>-0.11359901159350409</td>
      <td>-0.10748168344892405</td>
      <td>-0.11018111592616825</td>
      <td>-0.07897245267878145</td>
      <td>-0.1166695512602654</td>
      <td>-0.1213808022523802</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1,230,000.0</td>
      <td>0.8097963429966133</td>
      <td>3.206118321104874</td>
      <td>3.76474328596662</td>
      <td>2.0594699518161854</td>
      <td>-0.9172259379812189</td>
      <td>-0.1052746428490549</td>
      <td>-0.30183497242494894</td>
      <td>-0.6284335139334112</td>
      <td>2.8722119212096056</td>
      <td>1.2561073121770143</td>
      <td>-0.2070054668415114</td>
      <td>-1.025109284325866</td>
      <td>-0.09596277715461221</td>
      <td>-0.11717389485325336</td>
      <td>...</td>
      <td>-0.13699474409500317</td>
      <td>-0.12802573696800537</td>
      <td>-0.15367218648320422</td>
      <td>-0.11229613905271121</td>
      <td>-0.1294079886377916</td>
      <td>-0.11590914317994902</td>
      <td>-0.05200547477121165</td>
      <td>-0.1423334846112359</td>
      <td>-0.10910898221206813</td>
      <td>-0.11359901159350409</td>
      <td>-0.10748168344892405</td>
      <td>-0.11018111592616825</td>
      <td>-0.07897245267878145</td>
      <td>-0.1166695512602654</td>
      <td>-0.1213808022523802</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 82 columns</p>
</div>


    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.804
    Model:                            OLS   Adj. R-squared:                  0.803
    Method:                 Least Squares   F-statistic:                     877.0
    Date:                Sat, 17 Apr 2021   Prob (F-statistic):               0.00
    Time:                        23:40:37   Log-Likelihood:            -2.3317e+05
    No. Observations:               17424   AIC:                         4.665e+05
    Df Residuals:                   17342   BIC:                         4.671e+05
    Df Model:                          81                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    Intercept      5.353e+05   1191.443    449.273      0.000    5.33e+05    5.38e+05
    bedrooms      -2.383e+04   1564.307    -15.237      0.000   -2.69e+04   -2.08e+04
    bathrooms      1.983e+04   2208.535      8.980      0.000    1.55e+04    2.42e+04
    sqft_living    1.718e+05   2577.845     66.659      0.000    1.67e+05    1.77e+05
    sqft_lot       7960.0706   1312.149      6.066      0.000    5388.127    1.05e+04
    floors        -1.945e+04   1802.915    -10.790      0.000    -2.3e+04   -1.59e+04
    waterfront     4.657e+04   1373.600     33.904      0.000    4.39e+04    4.93e+04
    view           4.053e+04   1474.862     27.482      0.000    3.76e+04    4.34e+04
    condition       1.41e+04   1362.419     10.349      0.000    1.14e+04    1.68e+04
    grade          7.142e+04   2232.315     31.995      0.000     6.7e+04    7.58e+04
    basement      -2.747e+04   1504.135    -18.266      0.000   -3.04e+04   -2.45e+04
    renovated      6304.1397   1284.039      4.910      0.000    3787.294    8820.985
    home_age       2.168e+04   2079.948     10.424      0.000    1.76e+04    2.58e+04
    zipcode_98002  2847.4567   1486.308      1.916      0.055     -65.858    5760.771
    zipcode_98003 -1836.3316   1606.122     -1.143      0.253   -4984.493    1311.830
    zipcode_98004  9.203e+04   1655.281     55.597      0.000    8.88e+04    9.53e+04
    zipcode_98005  2.664e+04   1453.363     18.328      0.000    2.38e+04    2.95e+04
    zipcode_98006  3.998e+04   1856.570     21.535      0.000    3.63e+04    4.36e+04
    zipcode_98007  1.884e+04   1403.818     13.419      0.000    1.61e+04    2.16e+04
    zipcode_98008  2.927e+04   1607.379     18.211      0.000    2.61e+04    3.24e+04
    zipcode_98010  4869.4235   1355.127      3.593      0.000    2213.237    7525.610
    zipcode_98011  1.171e+04   1478.967      7.915      0.000    8806.611    1.46e+04
    zipcode_98014  7851.6245   1406.108      5.584      0.000    5095.511    1.06e+04
    zipcode_98019  7635.2857   1463.016      5.219      0.000    4767.627    1.05e+04
    zipcode_98022 -3412.0977   1548.000     -2.204      0.028   -6446.334    -377.861
    zipcode_98023 -4233.5881   1849.144     -2.289      0.022   -7858.097    -609.079
    zipcode_98024  9254.4859   1323.053      6.995      0.000    6661.169    1.18e+04
    zipcode_98027  2.348e+04   1757.710     13.356      0.000       2e+04    2.69e+04
    zipcode_98028  1.346e+04   1599.981      8.413      0.000    1.03e+04    1.66e+04
    zipcode_98029  2.577e+04   1669.723     15.432      0.000    2.25e+04     2.9e+04
    zipcode_98030   524.2864   1549.340      0.338      0.735   -2512.576    3561.149
    zipcode_98031  1887.8637   1581.064      1.194      0.232   -1211.180    4986.908
    zipcode_98032   311.8326   1395.817      0.223      0.823   -2424.110    3047.776
    zipcode_98033  5.161e+04   1781.445     28.971      0.000    4.81e+04    5.51e+04
    zipcode_98034  3.271e+04   1880.092     17.398      0.000     2.9e+04    3.64e+04
    zipcode_98038  5506.2634   1951.699      2.821      0.005    1680.736    9331.790
    zipcode_98039  5.924e+04   1280.335     46.269      0.000    5.67e+04    6.18e+04
    zipcode_98040  5.861e+04   1628.824     35.984      0.000    5.54e+04    6.18e+04
    zipcode_98042   934.6431   1905.133      0.491      0.624   -2799.609    4668.895
    zipcode_98045  9364.5603   1537.995      6.089      0.000    6349.935    1.24e+04
    zipcode_98052  3.837e+04   1918.412     20.002      0.000    3.46e+04    4.21e+04
    zipcode_98053  2.702e+04   1754.526     15.400      0.000    2.36e+04    3.05e+04
    zipcode_98055  5351.4293   1579.503      3.388      0.001    2255.445    8447.414
    zipcode_98056  1.343e+04   1745.033      7.694      0.000       1e+04    1.68e+04
    zipcode_98058  4621.1727   1784.024      2.590      0.010    1124.306    8118.039
    zipcode_98059  1.309e+04   1801.962      7.265      0.000    9559.501    1.66e+04
    zipcode_98065  1.077e+04   1631.794      6.600      0.000    7570.936     1.4e+04
    zipcode_98070  1586.2173   1403.857      1.130      0.259   -1165.484    4337.918
    zipcode_98072  1.775e+04   1596.931     11.113      0.000    1.46e+04    2.09e+04
    zipcode_98074  2.518e+04   1792.038     14.049      0.000    2.17e+04    2.87e+04
    zipcode_98075  2.337e+04   1711.877     13.651      0.000       2e+04    2.67e+04
    zipcode_98077  1.265e+04   1520.040      8.325      0.000    9674.231    1.56e+04
    zipcode_98092 -4603.3229   1691.148     -2.722      0.006   -7918.143   -1288.503
    zipcode_98102  3.109e+04   1362.006     22.829      0.000    2.84e+04    3.38e+04
    zipcode_98103  5.381e+04   1989.762     27.043      0.000    4.99e+04    5.77e+04
    zipcode_98105  4.618e+04   1548.344     29.823      0.000    4.31e+04    4.92e+04
    zipcode_98106  1.847e+04   1647.751     11.212      0.000    1.52e+04    2.17e+04
    zipcode_98107   3.96e+04   1628.434     24.317      0.000    3.64e+04    4.28e+04
    zipcode_98108  1.131e+04   1472.762      7.677      0.000    8418.988    1.42e+04
    zipcode_98109  3.204e+04   1385.802     23.119      0.000    2.93e+04    3.48e+04
    zipcode_98112   6.82e+04   1639.720     41.591      0.000     6.5e+04    7.14e+04
    zipcode_98115  5.385e+04   1947.131     27.657      0.000       5e+04    5.77e+04
    zipcode_98116  3.604e+04   1702.718     21.169      0.000    3.27e+04    3.94e+04
    zipcode_98117  4.974e+04   1935.598     25.697      0.000    4.59e+04    5.35e+04
    zipcode_98118  2.703e+04   1871.971     14.438      0.000    2.34e+04    3.07e+04
    zipcode_98119  4.218e+04   1499.555     28.127      0.000    3.92e+04    4.51e+04
    zipcode_98122  3.742e+04   1638.874     22.834      0.000    3.42e+04    4.06e+04
    zipcode_98125  2.773e+04   1739.234     15.942      0.000    2.43e+04    3.11e+04
    zipcode_98126  2.518e+04   1695.782     14.850      0.000    2.19e+04    2.85e+04
    zipcode_98133  2.454e+04   1847.002     13.285      0.000    2.09e+04    2.82e+04
    zipcode_98136   2.82e+04   1600.486     17.620      0.000    2.51e+04    3.13e+04
    zipcode_98144  3.598e+04   1718.334     20.938      0.000    3.26e+04    3.93e+04
    zipcode_98146  1.266e+04   1606.515      7.883      0.000    9514.835    1.58e+04
    zipcode_98148  3154.6956   1287.167      2.451      0.014     631.718    5677.673
    zipcode_98155  2.127e+04   1767.994     12.028      0.000    1.78e+04    2.47e+04
    zipcode_98166  7764.0640   1567.793      4.952      0.000    4691.032    1.08e+04
    zipcode_98168  8721.1891   1594.137      5.471      0.000    5596.521    1.18e+04
    zipcode_98177   2.35e+04   1564.720     15.017      0.000    2.04e+04    2.66e+04
    zipcode_98178  5651.5335   1574.308      3.590      0.000    2565.732    8737.335
    zipcode_98188  2967.7182   1398.834      2.122      0.034     225.862    5709.574
    zipcode_98198   825.7395   1606.333      0.514      0.607   -2322.835    3974.314
    zipcode_98199  4.603e+04   1666.930     27.616      0.000    4.28e+04    4.93e+04
    ==============================================================================
    Omnibus:                    15159.032   Durbin-Watson:                   1.999
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):          1886413.841
    Skew:                           3.630   Prob(JB):                         0.00
    Kurtosis:                      53.454   Cond. No.                         15.4
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    
    



```python
#create dataframe of feature coefficients
coefficients_m2_df = pd.DataFrame(model_2.params, columns=['Coefficient'])
coefficients_m2_df.drop('Intercept', inplace=True)
coefficients_m2_df = coefficients_m2_df.sort_values(by='Coefficient')
```


```python
#bar plot showing coefficients
fig, ax = plt.subplots(figsize=(15,20))
sns.barplot(data = coefficients_m2_df, y=coefficients_m2_df.index, x='Coefficient', ax=ax, orient='h');
```


    
![png](output_201_0.png)
    


### Model Interpretation

>**OBSERVATOINS**
> - Adjusted R-Squared of 0.803
> - All features with significant p-values except for some zipcodes 
> - The most positively correlated features to price are `sqft_living`, `waterfront`, `grade`  and `view`
> - The most negatively correlated features to price are `bedrooms`, `basement` and `floors`
> - QQ plot shows non-normality amongst the residuals
> - Homoscedasticity plot shows a larger spread of residuals in the upper range of `price`

>**ACTIONS**
> - I will proceed with removing outliers on `price` due to it not being modeled accurately on the high end

### Model Tuning

#### `price` Outlier Removal

I will investigate `price` for outliers.


```python
get_plots(df_model_base,'price',outlier='iqr')
```

    The number of rows removed is 897
    count            17,424.0
    mean    535,283.093721304
    std     354,215.472622472
    min              80,000.0
    25%             320,000.0
    50%             450,000.0
    75%             639,912.5
    max           7,060,000.0
    Name: price, dtype: float64



    
![png](output_207_1.png)
    


> **OBSERVATIONS**
> - I will use iqr to remove outliers because there are a lot of outliers on the high side of `price`.


```python
#create a copy of model_2 to set up model_3
df_model_3 = df_model_base.copy()
df_model_3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>basement</th>
      <th>renovated</th>
      <th>home_age</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>...</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221,900.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538,000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>63</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604,000.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>49</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510,000.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1,230,000.0</td>
      <td>4</td>
      <td>4.5</td>
      <td>5420</td>
      <td>101930</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>13</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>360,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>400,000.0</td>
      <td>4</td>
      <td>2.5</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>402,101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>400,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>325,000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>17424 rows × 82 columns</p>
</div>




```python
#remove outliers based off iqr
df_model_base = outliers(df_model_base, 'price', 'iqr')
df_model_3 = df_model_base.copy()
df_model_3
```

    There were 897 outliers removed.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>basement</th>
      <th>renovated</th>
      <th>home_age</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>...</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221,900.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538,000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>63</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604,000.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>49</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510,000.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>257,500.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>1715</td>
      <td>6819</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>360,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>400,000.0</td>
      <td>4</td>
      <td>2.5</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>402,101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>400,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>325,000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>16527 rows × 82 columns</p>
</div>




```python
#recheck the price column
get_plots(df_model_3,'price',outlier='none')
```

    count              16,527.0
    mean    475,336.27548859443
    std      206,876.3808524632
    min                80,000.0
    25%               315,000.0
    50%               435,000.0
    75%               600,000.0
    max             1,110,000.0
    Name: price, dtype: float64



    
![png](output_211_1.png)
    


## Model 3


```python
#view model_3 dataframe
df_model_3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>basement</th>
      <th>renovated</th>
      <th>home_age</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>...</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221,900.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538,000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>63</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604,000.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>49</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510,000.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>257,500.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>1715</td>
      <td>6819</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>360,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>400,000.0</td>
      <td>4</td>
      <td>2.5</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>402,101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>400,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>325,000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>16527 rows × 82 columns</p>
</div>



### Model Creation


```python
#define indpendent and dependent variables
x_cols = df_model_3.drop(columns='price').columns
y_col = 'price'
#run funciton to create model and check assumptions
model_3 = fit_new_model(df_model_3, x_cols=x_cols, y_col=y_col, norm=True, diagnose=False)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>basement</th>
      <th>renovated</th>
      <th>home_age</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>...</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221,900.0</td>
      <td>-0.35980469755201844</td>
      <td>-1.5005865056176506</td>
      <td>-1.026945274052607</td>
      <td>-0.22189347333406925</td>
      <td>-0.8896268959407717</td>
      <td>-0.07440626258298019</td>
      <td>-0.26577047974486373</td>
      <td>-0.6276960691150464</td>
      <td>-0.5150696497094087</td>
      <td>-0.7749107147926876</td>
      <td>-0.19492418999821054</td>
      <td>0.5395735569505244</td>
      <td>-0.09855703159210395</td>
      <td>-0.12035634493417341</td>
      <td>...</td>
      <td>-0.13961288419990514</td>
      <td>-0.131512428098114</td>
      <td>-0.15788832795153093</td>
      <td>-0.11371591232606608</td>
      <td>-0.12838294746101678</td>
      <td>-0.11800651559944278</td>
      <td>-0.05340195862506299</td>
      <td>-0.14535954732139936</td>
      <td>-0.11011327033901408</td>
      <td>-0.11668175328972062</td>
      <td>-0.10463598372304608</td>
      <td>8.857312435719964</td>
      <td>-0.08110084645133986</td>
      <td>-0.11931736963067766</td>
      <td>-0.11561160784814437</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538,000.0</td>
      <td>-0.35980469755201844</td>
      <td>0.2961643839368538</td>
      <td>0.7922061878543151</td>
      <td>-0.18306603547159264</td>
      <td>0.9775219654743189</td>
      <td>-0.07440626258298019</td>
      <td>-0.26577047974486373</td>
      <td>-0.6276960691150464</td>
      <td>-0.5150696497094087</td>
      <td>1.2903931689968218</td>
      <td>5.129889178762823</td>
      <td>0.6768043620307208</td>
      <td>-0.09855703159210395</td>
      <td>-0.12035634493417341</td>
      <td>...</td>
      <td>7.16222932204007</td>
      <td>-0.131512428098114</td>
      <td>-0.15788832795153093</td>
      <td>-0.11371591232606608</td>
      <td>-0.12838294746101678</td>
      <td>-0.11800651559944278</td>
      <td>-0.05340195862506299</td>
      <td>-0.14535954732139936</td>
      <td>-0.11011327033901408</td>
      <td>-0.11668175328972062</td>
      <td>-0.10463598372304608</td>
      <td>-0.11289423289599564</td>
      <td>-0.08110084645133986</td>
      <td>-0.11931736963067766</td>
      <td>-0.11561160784814437</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604,000.0</td>
      <td>0.8635016666615617</td>
      <td>1.3742149176695566</td>
      <td>-0.006126468090449303</td>
      <td>-0.2377463845216382</td>
      <td>-0.8896268959407717</td>
      <td>-0.07440626258298019</td>
      <td>-0.26577047974486373</td>
      <td>2.4561127698458245</td>
      <td>-0.5150696497094087</td>
      <td>1.2903931689968218</td>
      <td>-0.19492418999821054</td>
      <td>0.19649654425003357</td>
      <td>-0.09855703159210395</td>
      <td>-0.12035634493417341</td>
      <td>...</td>
      <td>-0.13961288419990514</td>
      <td>-0.131512428098114</td>
      <td>-0.15788832795153093</td>
      <td>8.793311969251631</td>
      <td>-0.12838294746101678</td>
      <td>-0.11800651559944278</td>
      <td>-0.05340195862506299</td>
      <td>-0.14535954732139936</td>
      <td>-0.11011327033901408</td>
      <td>-0.11668175328972062</td>
      <td>-0.10463598372304608</td>
      <td>-0.11289423289599564</td>
      <td>-0.08110084645133986</td>
      <td>-0.11931736963067766</td>
      <td>-0.11561160784814437</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510,000.0</td>
      <td>-0.35980469755201844</td>
      <td>-0.0631857939740471</td>
      <td>-0.3725742445896854</td>
      <td>-0.16262797458669603</td>
      <td>-0.8896268959407717</td>
      <td>-0.07440626258298019</td>
      <td>-0.26577047974486373</td>
      <td>-0.6276960691150464</td>
      <td>0.4504011567720925</td>
      <td>-0.7749107147926876</td>
      <td>-0.19492418999821054</td>
      <td>-0.5239651824209972</td>
      <td>-0.09855703159210395</td>
      <td>-0.12035634493417341</td>
      <td>...</td>
      <td>-0.13961288419990514</td>
      <td>-0.131512428098114</td>
      <td>-0.15788832795153093</td>
      <td>-0.11371591232606608</td>
      <td>-0.12838294746101678</td>
      <td>-0.11800651559944278</td>
      <td>-0.05340195862506299</td>
      <td>-0.14535954732139936</td>
      <td>-0.11011327033901408</td>
      <td>-0.11668175328972062</td>
      <td>-0.10463598372304608</td>
      <td>-0.11289423289599564</td>
      <td>-0.08110084645133986</td>
      <td>-0.11931736963067766</td>
      <td>-0.11561160784814437</td>
    </tr>
    <tr>
      <th>6</th>
      <td>257,500.0</td>
      <td>-0.35980469755201844</td>
      <td>0.2961643839368538</td>
      <td>-0.3267682725272809</td>
      <td>-0.19338262229057981</td>
      <td>0.9775219654743189</td>
      <td>-0.07440626258298019</td>
      <td>-0.26577047974486373</td>
      <td>-0.6276960691150464</td>
      <td>-0.5150696497094087</td>
      <td>-0.7749107147926876</td>
      <td>-0.19492418999821054</td>
      <td>-0.832734493851439</td>
      <td>-0.09855703159210395</td>
      <td>8.308157692044997</td>
      <td>...</td>
      <td>-0.13961288419990514</td>
      <td>-0.131512428098114</td>
      <td>-0.15788832795153093</td>
      <td>-0.11371591232606608</td>
      <td>-0.12838294746101678</td>
      <td>-0.11800651559944278</td>
      <td>-0.05340195862506299</td>
      <td>-0.14535954732139936</td>
      <td>-0.11011327033901408</td>
      <td>-0.11668175328972062</td>
      <td>-0.10463598372304608</td>
      <td>-0.11289423289599564</td>
      <td>-0.08110084645133986</td>
      <td>-0.11931736963067766</td>
      <td>-0.11561160784814437</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 82 columns</p>
</div>


    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.828
    Model:                            OLS   Adj. R-squared:                  0.827
    Method:                 Least Squares   F-statistic:                     977.1
    Date:                Sat, 17 Apr 2021   Prob (F-statistic):               0.00
    Time:                        23:40:39   Log-Likelihood:            -2.1119e+05
    No. Observations:               16527   AIC:                         4.226e+05
    Df Residuals:                   16445   BIC:                         4.232e+05
    Df Model:                          81                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    Intercept      4.753e+05    669.087    710.426      0.000    4.74e+05    4.77e+05
    bedrooms      -3738.0637    882.895     -4.234      0.000   -5468.633   -2007.494
    bathrooms      1.031e+04   1185.810      8.692      0.000    7982.441    1.26e+04
    sqft_living    8.805e+04   1339.342     65.745      0.000    8.54e+04    9.07e+04
    sqft_lot       1.084e+04    735.303     14.741      0.000    9397.481    1.23e+04
    floors        -5570.7182   1025.588     -5.432      0.000   -7580.981   -3560.455
    waterfront     4413.4761    741.636      5.951      0.000    2959.789    5867.163
    view           2.132e+04    769.366     27.705      0.000    1.98e+04    2.28e+04
    condition      1.285e+04    763.833     16.829      0.000    1.14e+04    1.44e+04
    grade          5.242e+04   1148.530     45.639      0.000    5.02e+04    5.47e+04
    basement      -1.268e+04    848.013    -14.958      0.000   -1.43e+04    -1.1e+04
    renovated      5463.9265    714.646      7.646      0.000    4063.143    6864.710
    home_age       1.649e+04   1183.305     13.938      0.000    1.42e+04    1.88e+04
    zipcode_98002   690.7288    834.744      0.827      0.408    -945.460    2326.918
    zipcode_98003  -605.6287    901.805     -0.672      0.502   -2373.264    1162.006
    zipcode_98004  4.562e+04    807.908     56.467      0.000     4.4e+04    4.72e+04
    zipcode_98005  2.823e+04    803.956     35.110      0.000    2.67e+04    2.98e+04
    zipcode_98006  3.809e+04    986.989     38.592      0.000    3.62e+04       4e+04
    zipcode_98007  1.944e+04    783.234     24.822      0.000    1.79e+04     2.1e+04
    zipcode_98008  2.729e+04    890.585     30.646      0.000    2.55e+04     2.9e+04
    zipcode_98010  6484.5758    761.260      8.518      0.000    4992.424    7976.728
    zipcode_98011  1.439e+04    830.624     17.320      0.000    1.28e+04     1.6e+04
    zipcode_98014  8866.8036    785.714     11.285      0.000    7326.720    1.04e+04
    zipcode_98019  9474.0582    821.762     11.529      0.000    7863.316    1.11e+04
    zipcode_98022  -532.3810    870.675     -0.611      0.541   -2238.998    1174.236
    zipcode_98023 -2784.1068   1036.566     -2.686      0.007   -4815.889    -752.325
    zipcode_98024  8986.9320    736.719     12.199      0.000    7542.884    1.04e+04
    zipcode_98027  2.723e+04    976.128     27.892      0.000    2.53e+04    2.91e+04
    zipcode_98028  1.594e+04    896.656     17.778      0.000    1.42e+04    1.77e+04
    zipcode_98029  2.914e+04    933.954     31.202      0.000    2.73e+04     3.1e+04
    zipcode_98030   981.6896    869.852      1.129      0.259    -723.314    2686.693
    zipcode_98031  1671.2277    887.657      1.883      0.060     -68.676    3411.132
    zipcode_98032  -674.2099    783.882     -0.860      0.390   -2210.703     862.283
    zipcode_98033  4.387e+04    959.024     45.744      0.000     4.2e+04    4.57e+04
    zipcode_98034  3.009e+04   1045.968     28.767      0.000     2.8e+04    3.21e+04
    zipcode_98038  7819.8287   1093.407      7.152      0.000    5676.632    9963.025
    zipcode_98039  7973.8217    672.811     11.851      0.000    6655.039    9292.605
    zipcode_98040  3.918e+04    821.418     47.700      0.000    3.76e+04    4.08e+04
    zipcode_98042  2228.5210   1069.250      2.084      0.037     132.675    4324.367
    zipcode_98045   1.15e+04    862.742     13.332      0.000    9810.810    1.32e+04
    zipcode_98052  4.335e+04   1069.763     40.520      0.000    4.13e+04    4.54e+04
    zipcode_98053  3.382e+04    970.708     34.844      0.000    3.19e+04    3.57e+04
    zipcode_98055  4815.7805    886.855      5.430      0.000    3077.448    6554.113
    zipcode_98056  1.443e+04    977.320     14.765      0.000    1.25e+04    1.63e+04
    zipcode_98058  5715.4576   1000.673      5.712      0.000    3754.031    7676.885
    zipcode_98059  1.643e+04   1002.332     16.392      0.000    1.45e+04    1.84e+04
    zipcode_98065  1.696e+04    913.756     18.559      0.000    1.52e+04    1.87e+04
    zipcode_98070  9444.5175    793.138     11.908      0.000    7889.881     1.1e+04
    zipcode_98072  2.116e+04    889.769     23.780      0.000    1.94e+04    2.29e+04
    zipcode_98074  3.224e+04    995.027     32.401      0.000    3.03e+04    3.42e+04
    zipcode_98075  3.227e+04    947.100     34.075      0.000    3.04e+04    3.41e+04
    zipcode_98077  1.854e+04    846.730     21.893      0.000    1.69e+04    2.02e+04
    zipcode_98092 -1408.3070    949.737     -1.483      0.138   -3269.895     453.281
    zipcode_98102  2.411e+04    751.211     32.092      0.000    2.26e+04    2.56e+04
    zipcode_98103  4.981e+04   1115.002     44.672      0.000    4.76e+04     5.2e+04
    zipcode_98105  3.305e+04    830.852     39.781      0.000    3.14e+04    3.47e+04
    zipcode_98106  1.427e+04    926.022     15.406      0.000    1.25e+04    1.61e+04
    zipcode_98107  3.536e+04    913.607     38.703      0.000    3.36e+04    3.71e+04
    zipcode_98108   1.01e+04    827.480     12.203      0.000    8475.491    1.17e+04
    zipcode_98109  2.597e+04    760.328     34.162      0.000    2.45e+04    2.75e+04
    zipcode_98112  3.791e+04    835.360     45.385      0.000    3.63e+04    3.96e+04
    zipcode_98115  5.014e+04   1080.673     46.393      0.000     4.8e+04    5.23e+04
    zipcode_98116  3.548e+04    947.005     37.464      0.000    3.36e+04    3.73e+04
    zipcode_98117  4.791e+04   1084.578     44.170      0.000    4.58e+04       5e+04
    zipcode_98118  2.467e+04   1049.473     23.502      0.000    2.26e+04    2.67e+04
    zipcode_98119  3.306e+04    816.276     40.502      0.000    3.15e+04    3.47e+04
    zipcode_98122  3.297e+04    910.289     36.214      0.000    3.12e+04    3.47e+04
    zipcode_98125   2.61e+04    972.910     26.824      0.000    2.42e+04     2.8e+04
    zipcode_98126   2.38e+04    953.613     24.963      0.000    2.19e+04    2.57e+04
    zipcode_98133  2.193e+04   1037.495     21.139      0.000    1.99e+04     2.4e+04
    zipcode_98136  2.804e+04    894.795     31.331      0.000    2.63e+04    2.98e+04
    zipcode_98144  2.961e+04    948.821     31.205      0.000    2.77e+04    3.15e+04
    zipcode_98146  1.204e+04    898.807     13.390      0.000    1.03e+04    1.38e+04
    zipcode_98148  2492.2259    722.879      3.448      0.001    1075.304    3909.147
    zipcode_98155  1.911e+04    989.753     19.309      0.000    1.72e+04    2.11e+04
    zipcode_98166  1.135e+04    874.371     12.986      0.000    9640.420    1.31e+04
    zipcode_98168  5306.7810    895.606      5.925      0.000    3551.296    7062.266
    zipcode_98177  2.182e+04    858.887     25.409      0.000    2.01e+04    2.35e+04
    zipcode_98178  5993.3084    883.791      6.781      0.000    4260.982    7725.635
    zipcode_98188  2323.7967    785.505      2.958      0.003     784.122    3863.471
    zipcode_98198  2460.5070    900.742      2.732      0.006     694.956    4226.058
    zipcode_98199  3.997e+04    903.833     44.228      0.000    3.82e+04    4.17e+04
    ==============================================================================
    Omnibus:                     1493.017   Durbin-Watson:                   1.987
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             5641.662
    Skew:                           0.406   Prob(JB):                         0.00
    Kurtosis:                       5.744   Cond. No.                         14.9
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    
    



```python
#create dataframe of feature coefficients
coefficients_m3_df = pd.DataFrame(model_3.params, columns=['Coefficient'])
coefficients_m3_df.drop('Intercept', inplace=True)
coefficients_m3_df = coefficients_m3_df.sort_values(by='Coefficient')
```


```python
#bar plot showing coefficients
fig, ax = plt.subplots(figsize=(15,20))
sns.barplot(data = coefficients_m3_df, y=coefficients_m3_df.index, x='Coefficient', ax=ax, orient='h');
```


    
![png](output_217_0.png)
    


### Model Interpretation

>**OBSERVATIONS**
> - Adjusted R-Squared is now 0.827
> - All features with a significant p-value except for some zipcodes
> - Majority of zipcodes with significant p-values so I will keep them in
> - Coefficients of features are smaller in absolute than they were in model 2. I believe this is because of removing outliers in pricing
> - The distribution of the residuals is more normal
> - The variance in the residuals is more even throughout the prediction of price

> **ACTIONS**
> - Going to look through outliers of all columns and remove extreme values

### Model Tuning


```python
fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model_3,'sqft_lot', fig=fig)
plt.show()
```


    
![png](output_221_0.png)
    


>**OBSERVATIONS**
> - The residuals of `sqft_lot` show heteroscedasticity toward the lower side. This seems mostly skewed by the high priced homes which have very small lot sizes that may represent homes closer to the city.

> **ACTIONS**
> - Will remove the outliers of sqft_lot first and maybe and see if there is any improvement in the overall model.

#### `sqft_lot` Outlier Removal

I will investigate `sqft_lot` for outliers.


```python
get_plots(df_model_base,'sqft_lot',outlier='iqr')
```

    The number of rows removed is 1814
    count              16,527.0
    mean    14,748.061293640709
    std      41,001.93285064867
    min                   520.0
    25%                 5,000.0
    50%                 7,500.0
    75%                10,283.5
    max             1,651,359.0
    Name: sqft_lot, dtype: float64



    
![png](output_225_1.png)
    


> **OBSERVATIONS**
> - I will use iqr to remove outliers of `sqft_lot`.


```python
#create a copy of model_2 to set up model_3
df_model_4 = df_model_base.copy()
df_model_4
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>basement</th>
      <th>renovated</th>
      <th>home_age</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>...</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221,900.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538,000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>63</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604,000.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>49</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510,000.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>257,500.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>1715</td>
      <td>6819</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>360,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>400,000.0</td>
      <td>4</td>
      <td>2.5</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>402,101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>400,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>325,000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>16527 rows × 82 columns</p>
</div>




```python
#remove outliers based off iqr
df_model_base = outliers(df_model_base, 'sqft_lot', 'iqr')
df_model_4 = df_model_base.copy()
df_model_4
```

    There were 1814 outliers removed.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>basement</th>
      <th>renovated</th>
      <th>home_age</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>...</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221,900.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538,000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>63</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604,000.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>49</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510,000.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>257,500.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>1715</td>
      <td>6819</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>360,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>400,000.0</td>
      <td>4</td>
      <td>2.5</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>402,101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>400,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>325,000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>14713 rows × 82 columns</p>
</div>




```python
#recheck the sqft_lot column
get_plots(df_model_4,'sqft_lot',outlier='none')
```

    count             14,713.0
    mean    7,188.183307279277
    std     3,443.210270301906
    min                  520.0
    25%                4,800.0
    50%                7,161.0
    75%                9,163.0
    max               18,200.0
    Name: sqft_lot, dtype: float64



    
![png](output_229_1.png)
    


## Model 4


```python
#view model_3 dataframe
df_model_4
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>basement</th>
      <th>renovated</th>
      <th>home_age</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>...</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221,900.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538,000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>63</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604,000.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>49</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510,000.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>257,500.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>1715</td>
      <td>6819</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>360,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>400,000.0</td>
      <td>4</td>
      <td>2.5</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>402,101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>400,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>325,000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>14713 rows × 82 columns</p>
</div>



### Model Creation


```python
#define indpendent and dependent variables
x_cols = df_model_4.drop(columns='price').columns
y_col = 'price'
#run funciton to create model and check assumptions
model_4 = fit_new_model(df_model_4, x_cols=x_cols, y_col=y_col, norm=True, diagnose=True)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>basement</th>
      <th>renovated</th>
      <th>home_age</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>...</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221,900.0</td>
      <td>-0.3393806244251373</td>
      <td>-1.4785611049109177</td>
      <td>-1.0026384463856017</td>
      <td>-0.4467294142754769</td>
      <td>-0.8800286019803993</td>
      <td>-0.06125326270434481</td>
      <td>-0.2531408140419408</td>
      <td>-0.6287318497048804</td>
      <td>-0.48307903473139885</td>
      <td>-0.7829744567260256</td>
      <td>-0.19235536441354206</td>
      <td>0.5051469756323201</td>
      <td>-0.10418559569785305</td>
      <td>-0.1246155621713666</td>
      <td>...</td>
      <td>-0.1469447356246033</td>
      <td>-0.13953242447855949</td>
      <td>-0.16673666109306365</td>
      <td>-0.11974518010645996</td>
      <td>-0.1362052552918972</td>
      <td>-0.12234634998876734</td>
      <td>-0.056000707069491826</td>
      <td>-0.1500522249433953</td>
      <td>-0.10681999793357178</td>
      <td>-0.11619162829398433</td>
      <td>-0.10583944313193025</td>
      <td>8.370933743331179</td>
      <td>-0.08313637724624941</td>
      <td>-0.12061810966115992</td>
      <td>-0.12234634998876728</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538,000.0</td>
      <td>-0.3393806244251373</td>
      <td>0.33264793499935746</td>
      <td>0.9343861946228085</td>
      <td>0.015629801405071992</td>
      <td>0.9655790982459019</td>
      <td>-0.06125326270434481</td>
      <td>-0.2531408140419408</td>
      <td>-0.6287318497048804</td>
      <td>-0.48307903473139885</td>
      <td>1.2770940664874963</td>
      <td>5.198357924379685</td>
      <td>0.638862819314041</td>
      <td>-0.10418559569785305</td>
      <td>-0.1246155621713666</td>
      <td>...</td>
      <td>6.804816985419732</td>
      <td>-0.13953242447855949</td>
      <td>-0.16673666109306365</td>
      <td>-0.11974518010645996</td>
      <td>-0.1362052552918972</td>
      <td>-0.12234634998876734</td>
      <td>-0.056000707069491826</td>
      <td>-0.1500522249433953</td>
      <td>-0.10681999793357178</td>
      <td>-0.11619162829398433</td>
      <td>-0.10583944313193025</td>
      <td>-0.11945286673580272</td>
      <td>-0.08313637724624941</td>
      <td>-0.12061810966115992</td>
      <td>-0.12234634998876728</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604,000.0</td>
      <td>0.88267202539548</td>
      <td>1.4193733589455224</td>
      <td>0.08432502123062846</td>
      <td>-0.6355067322355</td>
      <td>-0.8800286019803993</td>
      <td>-0.06125326270434481</td>
      <td>-0.2531408140419408</td>
      <td>2.4455558554157273</td>
      <td>-0.48307903473139885</td>
      <td>1.2770940664874963</td>
      <td>-0.19235536441354206</td>
      <td>0.1708573664280181</td>
      <td>-0.10418559569785305</td>
      <td>-0.1246155621713666</td>
      <td>...</td>
      <td>-0.1469447356246033</td>
      <td>-0.13953242447855949</td>
      <td>-0.16673666109306365</td>
      <td>8.350499218481739</td>
      <td>-0.1362052552918972</td>
      <td>-0.12234634998876734</td>
      <td>-0.056000707069491826</td>
      <td>-0.1500522249433953</td>
      <td>-0.10681999793357178</td>
      <td>-0.11619162829398433</td>
      <td>-0.10583944313193025</td>
      <td>-0.11945286673580272</td>
      <td>-0.08313637724624941</td>
      <td>-0.12061810966115992</td>
      <td>-0.12234634998876728</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510,000.0</td>
      <td>-0.3393806244251373</td>
      <td>-0.029593872982697576</td>
      <td>-0.30586699278545415</td>
      <td>0.2590073282519941</td>
      <td>-0.8800286019803993</td>
      <td>-0.06125326270434481</td>
      <td>-0.2531408140419408</td>
      <td>-0.6287318497048804</td>
      <td>0.536947061422888</td>
      <td>-0.7829744567260256</td>
      <td>-0.19235536441354206</td>
      <td>-0.5311508129010162</td>
      <td>-0.10418559569785305</td>
      <td>-0.1246155621713666</td>
      <td>...</td>
      <td>-0.1469447356246033</td>
      <td>-0.13953242447855949</td>
      <td>-0.16673666109306365</td>
      <td>-0.11974518010645996</td>
      <td>-0.1362052552918972</td>
      <td>-0.12234634998876734</td>
      <td>-0.056000707069491826</td>
      <td>-0.1500522249433953</td>
      <td>-0.10681999793357178</td>
      <td>-0.11619162829398433</td>
      <td>-0.10583944313193025</td>
      <td>-0.11945286673580272</td>
      <td>-0.08313637724624941</td>
      <td>-0.12061810966115992</td>
      <td>-0.12234634998876728</td>
    </tr>
    <tr>
      <th>6</th>
      <td>257,500.0</td>
      <td>-0.3393806244251373</td>
      <td>0.33264793499935746</td>
      <td>-0.25709299103344385</td>
      <td>-0.10722066859045074</td>
      <td>0.9655790982459019</td>
      <td>-0.06125326270434481</td>
      <td>-0.2531408140419408</td>
      <td>-0.6287318497048804</td>
      <td>-0.48307903473139885</td>
      <td>-0.7829744567260256</td>
      <td>-0.19235536441354206</td>
      <td>-0.832011461184888</td>
      <td>-0.10418559569785305</td>
      <td>8.024134509950041</td>
      <td>...</td>
      <td>-0.1469447356246033</td>
      <td>-0.13953242447855949</td>
      <td>-0.16673666109306365</td>
      <td>-0.11974518010645996</td>
      <td>-0.1362052552918972</td>
      <td>-0.12234634998876734</td>
      <td>-0.056000707069491826</td>
      <td>-0.1500522249433953</td>
      <td>-0.10681999793357178</td>
      <td>-0.11619162829398433</td>
      <td>-0.10583944313193025</td>
      <td>-0.11945286673580272</td>
      <td>-0.08313637724624941</td>
      <td>-0.12061810966115992</td>
      <td>-0.12234634998876728</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 82 columns</p>
</div>


    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.837
    Model:                            OLS   Adj. R-squared:                  0.836
    Method:                 Least Squares   F-statistic:                     928.9
    Date:                Sat, 17 Apr 2021   Prob (F-statistic):               0.00
    Time:                        23:40:42   Log-Likelihood:            -1.8736e+05
    No. Observations:               14713   AIC:                         3.749e+05
    Df Residuals:                   14631   BIC:                         3.755e+05
    Df Model:                          81                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    Intercept      4.655e+05    678.507    686.053      0.000    4.64e+05    4.67e+05
    bedrooms      -3462.2915    911.665     -3.798      0.000   -5249.269   -1675.314
    bathrooms      1.058e+04   1205.433      8.774      0.000    8213.190    1.29e+04
    sqft_living    8.499e+04   1359.254     62.528      0.000    8.23e+04    8.77e+04
    sqft_lot       5391.0531    981.067      5.495      0.000    3468.038    7314.069
    floors        -4425.0934   1110.844     -3.984      0.000   -6602.487   -2247.699
    waterfront     4949.2598    738.321      6.703      0.000    3502.057    6396.463
    view           2.091e+04    769.983     27.152      0.000    1.94e+04    2.24e+04
    condition      1.307e+04    778.881     16.775      0.000    1.15e+04    1.46e+04
    grade          4.826e+04   1159.868     41.605      0.000     4.6e+04    5.05e+04
    basement      -1.177e+04    871.659    -13.507      0.000   -1.35e+04   -1.01e+04
    renovated      5284.2503    725.461      7.284      0.000    3862.256    6706.244
    home_age        1.68e+04   1246.391     13.482      0.000    1.44e+04    1.92e+04
    zipcode_98002  1087.8128    874.680      1.244      0.214    -626.669    2802.295
    zipcode_98003  -542.2482    940.987     -0.576      0.564   -2386.702    1302.206
    zipcode_98004  4.626e+04    829.253     55.791      0.000    4.46e+04    4.79e+04
    zipcode_98005  2.629e+04    799.913     32.872      0.000    2.47e+04    2.79e+04
    zipcode_98006  3.831e+04   1021.295     37.513      0.000    3.63e+04    4.03e+04
    zipcode_98007  2.012e+04    810.831     24.820      0.000    1.85e+04    2.17e+04
    zipcode_98008   2.81e+04    931.968     30.148      0.000    2.63e+04    2.99e+04
    zipcode_98010  4985.2323    729.898      6.830      0.000    3554.540    6415.925
    zipcode_98011  1.425e+04    854.139     16.688      0.000    1.26e+04    1.59e+04
    zipcode_98014  5697.5583    733.193      7.771      0.000    4260.408    7134.708
    zipcode_98019  7917.4331    812.397      9.746      0.000    6325.032    9509.834
    zipcode_98022   -12.0443    827.327     -0.015      0.988   -1633.709    1609.620
    zipcode_98023 -2526.7942   1085.697     -2.327      0.020   -4654.897    -398.691
    zipcode_98024  4845.0438    709.578      6.828      0.000    3454.181    6235.907
    zipcode_98027  2.802e+04    927.609     30.208      0.000    2.62e+04    2.98e+04
    zipcode_98028  1.572e+04    928.267     16.939      0.000    1.39e+04    1.75e+04
    zipcode_98029  3.057e+04    983.389     31.082      0.000    2.86e+04    3.25e+04
    zipcode_98030   743.1074    903.763      0.822      0.411   -1028.382    2514.597
    zipcode_98031  1537.9587    917.478      1.676      0.094    -260.413    3336.331
    zipcode_98032  -295.0154    806.958     -0.366      0.715   -1876.754    1286.723
    zipcode_98033  4.432e+04   1003.964     44.148      0.000    4.24e+04    4.63e+04
    zipcode_98034   3.14e+04   1106.189     28.390      0.000    2.92e+04    3.36e+04
    zipcode_98038  7207.1707   1109.938      6.493      0.000    5031.553    9382.789
    zipcode_98039  8416.9041    683.032     12.323      0.000    7078.074    9755.734
    zipcode_98040  3.993e+04    847.465     47.117      0.000    3.83e+04    4.16e+04
    zipcode_98042  2037.9571   1074.556      1.897      0.058     -68.308    4144.222
    zipcode_98045  9037.1926    834.596     10.828      0.000    7401.280    1.07e+04
    zipcode_98052   4.37e+04   1113.604     39.240      0.000    4.15e+04    4.59e+04
    zipcode_98053  3.044e+04    927.713     32.814      0.000    2.86e+04    3.23e+04
    zipcode_98055  5141.5935    926.141      5.552      0.000    3326.239    6956.947
    zipcode_98056  1.456e+04   1023.003     14.230      0.000    1.26e+04    1.66e+04
    zipcode_98058  5262.7359   1023.052      5.144      0.000    3257.424    7268.048
    zipcode_98059  1.558e+04   1024.500     15.211      0.000    1.36e+04    1.76e+04
    zipcode_98065  1.762e+04    952.583     18.497      0.000    1.58e+04    1.95e+04
    zipcode_98070  2771.4419    725.416      3.820      0.000    1349.535    4193.349
    zipcode_98072  1.387e+04    826.418     16.783      0.000    1.22e+04    1.55e+04
    zipcode_98074  3.153e+04   1019.614     30.919      0.000    2.95e+04    3.35e+04
    zipcode_98075  2.975e+04    947.370     31.404      0.000    2.79e+04    3.16e+04
    zipcode_98077  9811.2057    732.329     13.397      0.000    8375.749    1.12e+04
    zipcode_98092 -2017.1068    930.761     -2.167      0.030   -3841.517    -192.697
    zipcode_98102  2.605e+04    782.619     33.282      0.000    2.45e+04    2.76e+04
    zipcode_98103  5.405e+04   1222.330     44.220      0.000    5.17e+04    5.64e+04
    zipcode_98105  3.564e+04    882.277     40.391      0.000    3.39e+04    3.74e+04
    zipcode_98106  1.576e+04    985.344     15.999      0.000    1.38e+04    1.77e+04
    zipcode_98107  3.846e+04    981.760     39.175      0.000    3.65e+04    4.04e+04
    zipcode_98108  1.118e+04    872.810     12.810      0.000    9469.713    1.29e+04
    zipcode_98109  2.808e+04    796.107     35.266      0.000    2.65e+04    2.96e+04
    zipcode_98112  4.084e+04    888.936     45.942      0.000    3.91e+04    4.26e+04
    zipcode_98115  5.384e+04   1174.536     45.843      0.000    5.15e+04    5.61e+04
    zipcode_98116  3.816e+04   1018.795     37.459      0.000    3.62e+04    4.02e+04
    zipcode_98117   5.18e+04   1184.587     43.732      0.000    4.95e+04    5.41e+04
    zipcode_98118  2.693e+04   1135.281     23.717      0.000    2.47e+04    2.92e+04
    zipcode_98119  3.569e+04    865.510     41.235      0.000     3.4e+04    3.74e+04
    zipcode_98122  3.588e+04    979.136     36.646      0.000     3.4e+04    3.78e+04
    zipcode_98125   2.75e+04   1032.765     26.631      0.000    2.55e+04    2.95e+04
    zipcode_98126    2.6e+04   1024.962     25.365      0.000     2.4e+04     2.8e+04
    zipcode_98133  2.369e+04   1108.626     21.365      0.000    2.15e+04    2.59e+04
    zipcode_98136  3.024e+04    950.359     31.820      0.000    2.84e+04    3.21e+04
    zipcode_98144  3.223e+04   1026.274     31.402      0.000    3.02e+04    3.42e+04
    zipcode_98146  1.271e+04    939.827     13.519      0.000    1.09e+04    1.45e+04
    zipcode_98148  2724.9929    741.438      3.675      0.000    1271.681    4178.305
    zipcode_98155  1.992e+04   1037.138     19.210      0.000    1.79e+04     2.2e+04
    zipcode_98166  9923.5322    885.575     11.206      0.000    8187.694    1.17e+04
    zipcode_98168  5758.3648    917.839      6.274      0.000    3959.285    7557.444
    zipcode_98177  2.156e+04    886.032     24.335      0.000    1.98e+04    2.33e+04
    zipcode_98178  6421.5638    933.172      6.881      0.000    4592.430    8250.698
    zipcode_98188  2224.2459    808.527      2.751      0.006     639.430    3809.061
    zipcode_98198  2832.4799    930.130      3.045      0.002    1009.307    4655.653
    zipcode_98199  4.284e+04    967.289     44.286      0.000    4.09e+04    4.47e+04
    ==============================================================================
    Omnibus:                     1111.490   Durbin-Watson:                   1.991
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3643.258
    Skew:                           0.364   Prob(JB):                         0.00
    Kurtosis:                       5.326   Cond. No.                         15.4
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    
    
    VIF Multicollinearity Test Results
    ======================================================================================================



    [('bedrooms', 1.8052258460138977),
     ('bathrooms', 3.1560787120331084),
     ('sqft_living', 4.012941576892276),
     ('sqft_lot', 2.090542150108155),
     ('floors', 2.6802014238503222),
     ('waterfront', 1.1840010763445867),
     ('view', 1.2877257008684329),
     ('condition', 1.3176616917253916),
     ('grade', 2.921989992063223),
     ('basement', 1.6502669654756037),
     ('renovated', 1.143112521172393),
     ('home_age', 3.3741918215413835),
     ('zipcode_98002', 1.6617258029584627),
     ('zipcode_98003', 1.9232201966915534),
     ('zipcode_98004', 1.4936048169063694),
     ('zipcode_98005', 1.3897811759551244),
     ('zipcode_98006', 2.2654970339341958),
     ('zipcode_98007', 1.4279809229577263),
     ('zipcode_98008', 1.8865269808841547),
     ('zipcode_98010', 1.1571404780548797),
     ('zipcode_98011', 1.5845967782458514),
     ('zipcode_98014', 1.1676093769575406),
     ('zipcode_98019', 1.433501415885143),
     ('zipcode_98022', 1.486673275547539),
     ('zipcode_98023', 2.5602279321654233),
     ('zipcode_98024', 1.0936091358571864),
     ('zipcode_98027', 1.8689220994735234),
     ('zipcode_98028', 1.8715755169655057),
     ('zipcode_98029', 2.1004495967023833),
     ('zipcode_98030', 1.774069262301884),
     ('zipcode_98031', 1.828320959299267),
     ('zipcode_98032', 1.4143690903184738),
     ('zipcode_98033', 2.1892604924059134),
     ('zipcode_98034', 2.657786186068738),
     ('zipcode_98038', 2.675831188551039),
     ('zipcode_98039', 1.0133141908844567),
     ('zipcode_98040', 1.5599276738774221),
     ('zipcode_98042', 2.5079527305462688),
     ('zipcode_98045', 1.5129120598504768),
     ('zipcode_98052', 2.6935348582190257),
     ('zipcode_98053', 1.869343274227229),
     ('zipcode_98055', 1.8630136567100917),
     ('zipcode_98056', 2.2730830459620006),
     ('zipcode_98058', 2.2733024689353742),
     ('zipcode_98059', 2.2797409357861045),
     ('zipcode_98065', 1.9709104371783528),
     ('zipcode_98070', 1.142972093759657),
     ('zipcode_98072', 1.4834098805528684),
     ('zipcode_98074', 2.2580453146287804),
     ('zipcode_98075', 1.949398271127825),
     ('zipcode_98077', 1.1648601472289926),
     ('zipcode_98092', 1.8816472473492962),
     ('zipcode_98102', 1.3303395753245912),
     ('zipcode_98103', 3.2451753616688555),
     ('zipcode_98105', 1.690717992420347),
     ('zipcode_98106', 2.1088086170398457),
     ('zipcode_98107', 2.0934963327977436),
     ('zipcode_98108', 1.6546304843976902),
     ('zipcode_98109', 1.376587366905492),
     ('zipcode_98112', 1.716337658147101),
     ('zipcode_98115', 2.9963631679233114),
     ('zipcode_98116', 2.2544209259137618),
     ('zipcode_98117', 3.0478617449740577),
     ('zipcode_98118', 2.7994216335510944),
     ('zipcode_98119', 1.6270675377665895),
     ('zipcode_98122', 2.0823196206571866),
     ('zipcode_98125', 2.316672245674781),
     ('zipcode_98126', 2.2817982723431034),
     ('zipcode_98133', 2.669507821832827),
     ('zipcode_98136', 1.9617170169617866),
     ('zipcode_98144', 2.2876414123383713),
     ('zipcode_98146', 1.9184778942422824),
     ('zipcode_98148', 1.1940191590201628),
     ('zipcode_98155', 2.3363328091193654),
     ('zipcode_98166', 1.7033817215030915),
     ('zipcode_98168', 1.8297601939446475),
     ('zipcode_98177', 1.7051424981098284),
     ('zipcode_98178', 1.8914042810811187),
     ('zipcode_98188', 1.419877315729031),
     ('zipcode_98198', 1.8790960306169342),
     ('zipcode_98199', 2.0322353245607885)]


    
    
    Normality Test Results
    ======================================================================================================



    
![png](output_233_4.png)
    


    
    
    Homoscedasticity Test Results
    ======================================================================================================



    
![png](output_233_6.png)
    



```python
#create dataframe of feature coefficients
coefficients_m4_df = pd.DataFrame(model_4.params, columns=['Coefficient'])
coefficients_m4_df.drop('Intercept', inplace=True)
coefficients_m4_df = coefficients_m4_df.sort_values(by='Coefficient')
```


```python
#create csv to export for data viz
# coefficients_df.to_csv(r'Model Coefficients (scaled) for Tableau.csv', header=True)
```


```python
#bar plot showing coefficients
fig, ax = plt.subplots(figsize=(15,20))
sns.barplot(data = coefficients_m4_df, y=coefficients_m4_df.index, x='Coefficient', ax=ax, orient='h');
```


    
![png](output_236_0.png)
    


### Model Interpretation

>**OBSERVATIONS**
> - $R^2$ is 0.836
> - The normality and homoscedasticity of the residuals are acceptable and therefore there will not be another iteration of the model.
> - All features except for some zipcodes are statistically significant.
> - Most negatively correlated features with price are `basement`, `floors` and `bedrooms`
> - Most positively correlated features with price are `sqft_living`, `grade` and `view`
 

# Interpret

The final model was created after 4 total iterations. Each iteration highlighted issues within the model that affected the accuracy of the model or its significance. Before the first model, I had evaluated linearity of features and multicollinearity between features. These were dealt with and remedied prior to running the first model. I also initially had zipcode not being OHE but this was difficult for the model to deal with because each zipcode has a lot of variation in how it affects home price. OHE zipcode jumped the $R^2$ significantly, however, the residuals of the model still showed room for improvement. This was primarily because of outliers in price and sqft_lot which were remedied in model iteration 3 and 4. Model 4 (final model) showed a $R^2$ of .836 with significant features and almost normal and homoscedastic residuals. The final model highlighted a few insights:
 - Square Footage is the feature which best predicts home price (highest normalized coefficient).
 - Zipcodes vary widely in their influence on home price
 - The grade of construction is a highly influential feature for home price but it depends on whether or not you have high or low construction quality.
 - The view is of the home is an extremely important feature when predicting price but is mostly an uncontrollable feature for a home owner.
 - Bathrooms are also very important to the overall home price
 - It is important to stay away from adding bedrooms or floors

# Recommendations and Conclusions

Based on what the model showed were significantly impactful features, I recommend the following actions for any renovator looking to make smart decisions that will add value to their home:
 - Add a full size bathroom (~60 square feet) with above average construction quality to improve home value
 - If the house has an unfinished basement of more than 350 square feet, finish the basement to get the extra square feet. This will offset the fact that the model views having a basement negatively affects the home value when considered by itself. However, if a home has an unfinished basement around the median size of the area, 700 square feet, then it will end up being a large value increase to the value of the home. Again, utilizing above average construction quality will add additional value.

# Appendix

## Dataset for Tableau


```python
#view the dataframe
df_scrub
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>basement</th>
      <th>renovated</th>
      <th>yr_sold</th>
      <th>home_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-10-13</td>
      <td>221,900.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>1955</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-12-09</td>
      <td>538,000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>1951</td>
      <td>98125</td>
      <td>47.721000000000004</td>
      <td>-122.319</td>
      <td>1</td>
      <td>1</td>
      <td>2014</td>
      <td>63</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-12-09</td>
      <td>604,000.0</td>
      <td>4</td>
      <td>3.0</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>1965</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.39299999999999</td>
      <td>1</td>
      <td>0</td>
      <td>2014</td>
      <td>49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-02-18</td>
      <td>510,000.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>1987</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>28</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2014-05-12</td>
      <td>1,230,000.0</td>
      <td>4</td>
      <td>4.5</td>
      <td>5420</td>
      <td>101930</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>11</td>
      <td>3890</td>
      <td>2001</td>
      <td>98053</td>
      <td>47.6561</td>
      <td>-122.005</td>
      <td>1</td>
      <td>0</td>
      <td>2014</td>
      <td>13</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>2014-05-21</td>
      <td>360,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1530</td>
      <td>2009</td>
      <td>98103</td>
      <td>47.6993</td>
      <td>-122.346</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>5</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>2015-02-23</td>
      <td>400,000.0</td>
      <td>4</td>
      <td>2.5</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2310</td>
      <td>2014</td>
      <td>98146</td>
      <td>47.5107</td>
      <td>-122.36200000000001</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>2014-06-23</td>
      <td>402,101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>2009</td>
      <td>98144</td>
      <td>47.5944</td>
      <td>-122.29899999999999</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>5</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>2015-01-16</td>
      <td>400,000.0</td>
      <td>3</td>
      <td>2.5</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1600</td>
      <td>2004</td>
      <td>98027</td>
      <td>47.5345</td>
      <td>-122.069</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>11</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>2014-10-15</td>
      <td>325,000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>2008</td>
      <td>98144</td>
      <td>47.5941</td>
      <td>-122.29899999999999</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>17704 rows × 20 columns</p>
</div>




```python
#remove outliers
df_scrub = df_scrub.loc[df_scrub['bedrooms'] <= 5]
```


```python
#remove outliers based off iqr
df_scrub = outliers(df_scrub, 'price', 'iqr')
```

    There were 897 outliers removed.



```python
#remove outliers based off iqr
df_scrub = outliers(df_scrub, 'sqft_lot', 'iqr')
```

    There were 1814 outliers removed.



```python
#write csv file for tableau
# df_scrub.to_csv(r'Scrubbed Housing Data for Tableau.csv',index=False, header=True)
```
