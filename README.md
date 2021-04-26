<img src='Images/Rennovation README Title.png'>

# Evaluating Home Renovation Opportunities in King County, WA

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

<img src='Images/Model Feature Influence.png' >

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