# Customer Segmentation and MailOut Response Prediction 
##                    - Arvato Bertelsmann Capstone


### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Acknowledgements](#Acknowledgements)

## Installation <a name="installation"></a>
- This project was built using Anaconda version 1.9.12 and Python  3.6.    

- Libraries used:

    `pandas`

    `numpy`

    `matplotlib`

    `seaborn`

    `sklearn`

    `xgboost`

    
## Project Motivation<a name="motivation"></a>

This is part of the Udacity Data Scientist Nanodegree Capstone Project provided in collaboration with Bertelsmann Arvato Analytics, aimed to help the company create a more efficient way of targeting people who are more likely to become customers. The purpose is to analyze the demographics of the general population in Germany against demographics data of customers of a German mail-order company, explain the difference and predict customer response to a marketing campaign.

Using principal component analysis(PCA) and unsupervised techniques KMeans, original data set was consolidated to 55 components and classified into 10 clusters. Center value comparison shows that, there are significant difference between cluster 2(customers), and cluster 8(general population), in turn made it possible to identify parts of the population that best described the core customer base of the German company. After that, supervised learning tool XGBClassifier was used to identify the important features for customer, and predicted whether a person would respond to a marketing campaign.


## File Descriptions <a name="files"></a>

- There are two iPython notebook files and two supporting files  
**Arvato Project Workbook - Part1 - Data Wrangling.ipynb** - Data exploratin and clean  
**Arvato Project Workbook - Part2 - Segmentation & Prediction.ipynb** - PCA, unsupervised learning model for customer segementation and supervised learning model for prediction  
**utils.py** - Collective functions generated in Part1 to be used in the project  
**attr_summary_1.xlsx** - Attribute Data Summary


## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/p/e19ddc729295/edit), or [知乎](https://zhuanlan.zhihu.com/p/360846170).

## Licensing, Authors, Acknowledgements<a name="Acknowledgements"></a>
Thanks to Udacity for providing this excellent project and and Bertelsmann Arvato Analytics for providing opportunity to apply my data science skills on a real-life problem.
