#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[84]:


data = pd.read_csv('C:\\Users\\SIDDHARTH\\Desktop\\python project\playstore-analysis.csv')


# In[85]:


data.head()


# Drop records where rating is missing since rating is our target/study variable

# In[86]:


pd.isnull(data["Rating"]).sum()


# In[87]:


missing=data[data.isnull().any(axis=1)]


# In[88]:


missing


# In[89]:


data1=data.drop(list(missing.index),inplace=True)


# Check the null values for the Android Ver column.

# In[90]:


data.isna().sum()


# Current ver – replace with most common value

# In[91]:


data.mode()


# In[92]:


data['Current Ver'].value_counts()


# In[93]:


data['Current Ver'].replace(to_replace="Varies with device",
            value = 1.0)


# In[94]:


data.info()


# In[95]:


data["Price"].unique()


# In[96]:


data.loc[data["Price"]=="Everyone"]


# Price variable – remove $ sign and convert to float

# In[97]:


data["Price"]=data["Price"].apply(lambda x : float(x.replace("$","")))


# Installs – remove ‘,’ and ‘+’ sign, convert to integer

# In[98]:



data['Installs']=data['Installs'].str.replace(',', '').str.replace('+', '').astype(int)


# Which all variables need to be brought to numeric types?

# In[99]:


data.dtypes


# In[100]:


data["Reviews"]=data["Reviews"].apply(lambda x : float(x))


# In[101]:



data.dtypes


# Avg. rating should be between 1 and 5, as only these values are allowed on the play store

# In[102]:


(data.loc[data.Rating > 5])


# Reviews should not be more than installs as only those who installed can review the app.

# In[103]:


data.loc[data["Reviews"] > data["Installs"]]


# In[104]:




data.drop([4663,10697],inplace=True)


# i. Make suitable plot to identify outliers in price

# In[105]:


data.boxplot(["Price"])


# In[106]:



#Do you expect apps on the play store to cost $200? Check out these cases
#ans:yes


# In[107]:


data.loc[data['Price'] > 200]
data.drop(data.loc[data['Price'] > 200])


# In[108]:


data.describe()


# In[109]:


(data.loc[data.Price > 30])


# In[110]:


data.drop([5373,5369,5366,5364,5362,5360,5359,5358,5357,5356,5354,5355,5351,4367,4362,4197,2414,2402,2365,2301,2253],inplace = True)


# In[111]:


(data.loc[data.Price > 30])


# In[112]:


data.max()


# After dropping the useless records, make the suitable plot again to identify outliers

# In[113]:


sns.boxplot(data["Price"])


# i. Make suitable plot
# ii. Limit data to apps with < 1 Million reviews

# In[114]:


lower_bound=0.1
upper_bound=0.95
data.quantile([lower_bound,upper_bound])


# In[115]:


data.boxplot(["Reviews"])
plt.xlim(-100000,1000000)
gt_1m = data[data['Reviews'] > 1000000 ].index
data.drop(labels = gt_1m, inplace=True)
print(gt_1m.value_counts().sum(),'cols dropped')


# What is the 95th percentile of the installs?

# In[116]:


data["Installs"].quantile(.95)


# In[117]:


(data.loc[data.Installs > data["Installs"].quantile(.95)])


# Drop records having a value more than the 95th percentile

# In[118]:


data.drop([3234,3235,3255,3265,3326,3450,3454,3473,3476,3522,3523,3533,3562,3565,3569,3574,3665,3687,3703,3711,3736,3739,1661,1662,1700,1702,1705,1722,1729,1750,1751,1759,1842,1869,1872,1885,1886,1908,1917,1920,2544,2545,2546,2550,2554,2603,2604,2610,2611,2808,2884,3117,3127,3223,3232],inplace = True)


# What is the distribution of ratings like? (use Seaborn) More skewed towards higher/lower
# values?

# In[119]:


sns.distplot(data['Rating'])
plt.show()
print('The skewness of this distribution is',data['Rating'].skew())
print('The Median of this distribution {} is greater than mean {} of this distribution'.format(data.Rating.median(),data.Rating.mean()))


# What is the implication of this on your analysis?
# 

# In[120]:


data['Rating'].mode()


# Since mode>= median > mean, the distribution of Rating is Negatively Skewed.Therefore distribution of Rating is more Skewed towards lower values.

# In[121]:


#What are the top Content Rating values?
#a. Are there any values with very few records?


# In[122]:


data['Content Rating'].value_counts()


# In[123]:


cr = []
for k in data['Content Rating']:
    cr.append(k.replace('Adults only 18+','NaN').replace('Unrated','NaN'))

data['Content Rating']=cr


# In[124]:


# Droping the NaN values.
temp2 = data[data["Content Rating"] == 'NaN'].index
data.drop(labels=temp2, inplace=True)
print('droped cols',temp2)


# Make a joinplot to understand the effect of size on rating

# In[125]:


sns.jointplot(data["Size"],data["Rating"],kind="hex")


# b. Do you see any patterns?
# Yes, patterns can be observed between Size and Rating ie. their is correlation between Size and Rating.

# c. How do you explain the pattern?
# Generally on increasing Rating, Size of App also increases. But this is not always true ie. for higher Rating, their is constant Size. Thus we can conclude that their is positive correlation between Size and Rating.

# 8. Effect of price on rating
# a. Make a jointplot (with regression line)

# In[126]:


sns.jointplot(x='Price', y='Rating', data=data,kind='reg')
plt.show()


# In[ ]:


(data.loc[data.Price > 0])


# In[127]:


data1=data.loc[data.Price>0]
sns.jointplot(x='Price', y='Rating', data=data1, kind='reg')
plt.show()


# e. Does the pattern change?
# Yes, On limiting the record with Price > 0, the overall pattern changed a slight ie their is very weakly Negative Correlation between Price and Rating.

# In[128]:


data1.corr()


# f. What is your overall inference on the effect of price on the rating
# Generally increasing the Prices, doesn't have signifcant effect on Higher Rating. For Higher Price, Rating is High and almost constant ie greater than 4

# In[129]:


sns.pairplot(data)


# Make a bar plot displaying the rating for each content rating

# In[130]:


data.groupby(['Content Rating'])['Rating'].count().plot.bar(color="darkgreen")
plt.show()


# b. Which metric would you use? Mean? Median? Some other quantile?
# We must use Median in this case as we are having Outliers in Rating. Because in case of Outliers , median is the best measure of central tendency.

# In[131]:


plt.boxplot(data['Rating'])
plt.show()


# c. Choose the right metric and plot

# In[132]:


data.groupby(['Content Rating'])['Rating'].median().plot.barh(color="darkgreen")
plt.show()


# a. Create 5 buckets (20% records in each) based on Size

# In[133]:


bins=[0, 20000, 40000, 60000, 80000, 100000]
data['Bucket Size'] = pd.cut(data['Size'], bins, labels=['0-20k','20k-40k','40k-60k','60k-80k','80k-100k'])
pd.pivot_table(data, values='Rating', index='Bucket Size', columns='Content Rating')


# b. By Content Rating vs. Size buckets, get the rating (20th percentile) for each combination

# In[134]:


temp3=pd.pivot_table(data, values='Rating', index='Bucket Size', columns='Content Rating', aggfunc=lambda x:np.quantile(x,0.2))
temp3


# c. Make a heatmap of this
# i. Annotated

# In[135]:


f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(temp3, annot=True, linewidths=.5, fmt='.1f',ax=ax)
plt.show()


# ii.Greens color map

# In[136]:


f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(temp3, annot=True, linewidths=.5, cmap='Greens',fmt='.1f',ax=ax)
plt.show()


# d. What’s your inference? Are lighter apps preferred in all categories? Heavier? Some?
# Based on analysis, its not true that lighter apps are preferred in all categories. Because apps with size 40k-60k and 80k-100k have got the highest rating in all cateegories. So, in general we can conclude that heavier apps are preferred in all categories.

# In[ ]:




