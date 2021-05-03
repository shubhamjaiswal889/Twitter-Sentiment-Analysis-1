#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[4]:


#reading/loading dataset
df= pd.read_csv(r'C:\Users\shubham.kj\Desktop\IndiaWantsOxygen.csv')


# In[5]:


df.head()


# In[10]:


df.dtypes


# In[7]:


df.shape


# In[8]:


df.isnull().sum()


# In[9]:


df= df[pd.notnull(df["user_name"])]


# In[11]:


df.isnull().sum()


# In[12]:


len(df)


# In[13]:


len(df.user_name.unique())


# In[14]:


#wordcloud visualisation of texts within the tweets
from wordcloud import WordCloud, STOPWORDS
plt.figure(figsize= (20,20))
words= "".join(df["text"])
final = WordCloud(width = 2000, height = 800, background_color ="black",min_font_size = 10).generate(words)
plt.imshow(final)
plt.axis("off") 
plt.show()   


# In[1]:


pip install wordcloud


# In[2]:


#wordcloud visualisation of texts within the tweets
from wordcloud import WordCloud, STOPWORDS
plt.figure(figsize= (20,20))
words= "".join(df["text"])
final = WordCloud(width = 2000, height = 800, background_color ="black",min_font_size = 10).generate(words)
plt.imshow(final)
plt.axis("off") 
plt.show()   


# In[3]:


#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[5]:


#reading/loading dataset
df= pd.read_csv(r'C:\Users\shubham.kj\Desktop\IndiaWantsOxygen.csv')


# In[6]:


#wordcloud visualisation of texts within the tweets
from wordcloud import WordCloud, STOPWORDS
plt.figure(figsize= (20,20))
words= "".join(df["text"])
final = WordCloud(width = 2000, height = 800, background_color ="black",min_font_size = 10).generate(words)
plt.imshow(final)
plt.axis("off") 
plt.show()   


# In[7]:


#wordcloud visualisation of hashtags used during tweets
from wordcloud import WordCloud, STOPWORDS
plt.figure(figsize= (20,20))
words= "".join(df["hashtags"])
final = WordCloud(width = 2000, height = 800, background_color ="black",min_font_size = 10).generate(words)
plt.imshow(final)
plt.axis("off") 
plt.show()   


# In[8]:


#top locations from where people tweeted
df.user_location.value_counts().head(15)


# In[9]:


#Pakistan is the top location for the tweets on #indianeedsoxygen varying from cities like lahore and karachi
#England ,France,Dubai were other location countries that followed

x= df.user_location.value_counts().head(15)
plt.figure(figsize= (10,7))
sns.set_style("whitegrid")
ax= sns.barplot(x.values,x.index)
ax.set_xlabel("No of tweets")
ax.set_ylabel("Locations")
plt.show()


# In[10]:


#setting date datatype to date column and removing time values from date for plotting
df["date"]= pd.to_datetime(df.date)
df.date= df.date.apply(lambda x: str(x).split(" ")[0])
df.date


# In[11]:


#Number of tweets over time
x= df.groupby("date").date.count()
plt.figure(figsize= (15,7))
sns.set_style("whitegrid")
ax= sns.lineplot(x.index,x.values)
ax.set_xlabel("date")
ax.set_ylabel("No of tweets")
plt.show()


# In[12]:


#Top sources used for tweets
df.source.value_counts()


# In[13]:


#Top sources coutplot
plt.figure(figsize= (15,7))
ax= sns.countplot(x= "source",data= df)
plt.xticks(rotation=90)
plt.show()


# In[14]:


#Out of overall users that tweeted, 98% of the users are verified users
x= df.user_verified.value_counts()
plt.figure(figsize= (15,7))
labels=("Verified","Non verified")
plt.pie(x,labels= labels,autopct= "%1.1f%%")
plt.show()


# In[15]:


#Not a single tweet in the dataset is a retweet
x= df.is_retweet.value_counts()
x


# In[16]:


#users with most(multiple) tweets on the subject
df.user_name.value_counts().head(20)


# In[17]:


x= df.user_name.value_counts().head(20)
plt.figure(figsize= (7,10))
ax= sns.barplot(x.values,x.index)
ax.set_xlabel("No of tweets")
ax.set_ylabel("Usernames")
plt.show()


# In[18]:


df.user_location.unique()


# In[19]:


#wordcloud visualisation of locations for tweets
x = df[pd.notnull(df["user_location"])]
from wordcloud import WordCloud, STOPWORDS
plt.figure(figsize= (20,20))
words= "".join(str(x["user_location"]))
final = WordCloud(width = 2000, height = 800, background_color ="black",min_font_size = 10).generate(words)
plt.imshow(final)
plt.axis("off") 
plt.show()   


# In[20]:


#wordcloud visualisation of usernames 
from wordcloud import WordCloud, STOPWORDS
plt.figure(figsize= (20,20))
words= "".join(df["user_name"])
final = WordCloud(width = 2000, height = 800, background_color ="black",min_font_size = 10).generate(words)
plt.imshow(final)
plt.axis("off") 
plt.show()   
     


# In[ ]:




