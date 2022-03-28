#import geopandas
#from geopandas import GeoDataFrame
import re
import pandas as pd
#import geoplotlib
#from geoplotlib.utils import BoundingBox
import datetime
import matplotlib.pyplot as plt
#from shapely.geometry import Point, Polygon
import os

os.chdir('D:/美国/UC Berkeley/Assignments/mids-w200-assignments-upstream-spring2021/SUBMISSIONS/project_2')

graf_df = pd.read_csv("Parks_Graffiti_Report_-_Dataset.csv")
#rename the columns
graf_df = graf_df.rename(columns = {'Location Name':"location_name","Work Performed Date":"work_date","Location Address":"location_address","Issue":"issue",
                                    "Description":"descr","Labor Cost":"labor","Material Cost":"material","Total Cost":"total_cost","Location":"location"})
#add column of the month work is perfomed
graf_df['work_date'] = pd.to_datetime(graf_df['work_date'])
graf_df["work_month"] = graf_df.work_date.dt.strftime('%Y-%m')
graf_df["work month without year"] = graf_df['work_date'].dt.month

graf_df['work_year'] = graf_df.work_date.dt.year
null_data = graf_df[graf_df.isnull().any(axis=1)]
null_data.shape
#remove the entries with missing values from the dataset
graf_df = graf_df.dropna().reset_index()



###################################################  plot

################ Incident by time

#prepare df

#by_time = graf_df.groupby("work_month").describe()
incident_by_year_month = graf_df.groupby("work_month").agg({'count'})
incident_by_year_month = incident_by_year_month.iloc[:,0].reset_index()
incident_by_year_month.columns = ['year_month', 'count']

incident_by_month = graf_df.groupby("work month without year").agg({'count'})
incident_by_month = incident_by_month.iloc[:,0].reset_index()
incident_by_month.columns = ['month', 'count']

incident_by_year = graf_df.groupby("work_year").agg({'count'})
incident_by_year = incident_by_year.iloc[:,0].reset_index()
incident_by_year.columns = ['year', 'count']
incident_by_year['year'] = incident_by_year['year'].astype(int).astype(str)
##incident_by_year_month['index']

#plotting
fig, ax = plt.subplots(1,3,figsize = (20,5))
ax[0].plot(incident_by_year_month['year_month'], incident_by_year_month['count'])
ax[1].bar(incident_by_month['month'], incident_by_month['count'], color = '#9f90ac')
ax[2].bar(incident_by_year['year'], incident_by_year['count'], color = '#9fc8ac')
  
  
ax[0].set_title('trend of incident by year and month')
ax[0].set_xticklabels(labels = incident_by_year_month['year_month'], rotation = 90, fontsize = 7)
ax[0].set_xlabel('year month')
ax[0].set_ylabel('count')

ax[1].set_title('trend of incident by month')
ax[1].set_xticklabels(labels = incident_by_month['month'], fontsize = 10)
ax[1].set_xlabel('month')
ax[1].set_ylabel('count')

ax[2].set_title('trend of incident by year')
ax[2].set_xticklabels(labels = incident_by_year['year'],  fontsize = 12)
ax[2].set_xlabel('year')
ax[2].set_ylabel('count')


################ cost by time

cost_by_year_month = graf_df['total_cost'].groupby(graf_df["work_month"] ).sum()
cost_by_month = graf_df['total_cost'].groupby(graf_df["work month without year"] ).sum()
cost_by_year = graf_df['total_cost'].groupby(graf_df["work_year"] ).sum().reset_index()
cost_by_year['work_year'] = cost_by_year['work_year'].astype(int).astype(str)


fig, ax = plt.subplots(1,3,figsize = (20,5))
ax[0].plot(cost_by_year_month.index, cost_by_year_month)
ax[1].plot(cost_by_month.index, cost_by_month)
ax[2].plot(cost_by_year['work_year'], cost_by_year['total_cost'])


ax[0].set_title('total cost by year and month')
ax[0].set_xticklabels(cost_by_year_month.index, rotation = 90, fontsize = 7)
ax[0].set_xlabel('year month')
ax[0].set_ylabel('total cost')

ax[1].set_title('total cost by month')
ax[1].set_xticklabels(labels = cost_by_month.index, fontsize = 10)
ax[1].set_xlabel('month')
ax[1].set_ylabel('total cost')

ax[2].set_title('total cost by year')
ax[2].set_xticklabels(labels = cost_by_year['work_year'],  fontsize = 12)
ax[2].set_xlabel('year')
ax[2].set_ylabel('total cost')

################ labor vs material cost

LMcost_by_year_month = graf_df[['labor', 'material']].groupby(graf_df["work_month"] ).sum()
LMcost_by_month = graf_df[['labor', 'material']].groupby(graf_df["work month without year"] ).sum()
LMcost_by_year = graf_df[['labor', 'material']].groupby(graf_df["work_year"] ).sum().reset_index()
LMcost_by_year['work_year'] = LMcost_by_year['work_year'].astype(int).astype(str)


fig, ax = plt.subplots(1,3,figsize = (20,5))
ax[0].bar(LMcost_by_year_month.index, LMcost_by_year_month['labor'], color = '#9f90ac')                                                  
ax[0].bar(LMcost_by_year_month.index, LMcost_by_year_month['material'], bottom = LMcost_by_year_month['labor'], color = '#9fc8ac')

ax[1].bar(LMcost_by_month.index, LMcost_by_month['labor'], color = '#9f90ac')                                                  
ax[1].bar(LMcost_by_month.index, LMcost_by_month['material'], bottom = LMcost_by_month['labor'], color = '#9fc8ac')

ax[2].bar(LMcost_by_year['work_year'], LMcost_by_year['labor'], color = '#9f90ac', width = .6)                                                  
ax[2].bar(LMcost_by_year['work_year'], LMcost_by_year['material'], bottom = LMcost_by_year['labor'], color = '#9fc8ac', width = .6)

ax[0].set_title('labor/material cost by year and month')
ax[0].set_xticklabels(LMcost_by_year_month.index, rotation = 90, fontsize = 7)
ax[0].set_xlabel('year month')
ax[0].set_ylabel('labor/material cost')

ax[1].set_title('labor/material cost by month')
ax[1].set_xticklabels(labels = LMcost_by_month.index, fontsize = 10)
ax[1].set_xlabel('month')
ax[1].set_ylabel('labor/material cost')

ax[2].set_title('labor/material cost by year')
ax[2].set_xticklabels(labels = LMcost_by_year['work_year'],  fontsize = 12)
ax[2].set_xlabel('year')
ax[2].set_ylabel('labor/material cost')


####################### correlation between incident and total cost
incident_by_year_month.columns
cost_by_year_month2 = cost_by_year_month.reset_index()
cost_count = pd.merge(left = incident_by_year_month, right = cost_by_year_month2, how = 'inner', left_on = 'year_month', right_on = 'work_month')


import seaborn as sns
sns.regplot(x=cost_count['count'],
            y=cost_count['total_cost'], 
            scatter_kws={'color': '#ff6600', 's':cost_count['count'] * 8, 'alpha': 0.7},
            line_kws={'color': '#cc66ff'},
            data=cost_count)
plt.title('correlation between incident and total cost')
plt.ylabel('total cost')


################ top 10 bubble of description column


description_list = graf_df['descr'].str.lower().str.replace(',', '').str.split(' ')

word_list = []
for x in description_list:
    for i in x:
        if i.endswith('s') and i.endswith('es') == False:
           i = i[:-1]
        word_list.append(i)
        

from collections import Counter
word_count = Counter(word_list)

sorted_word_count = sorted(word_count.items(), key=lambda item: item[1], reverse = True)

word_l = []
occurence_l = []
for x in sorted_word_count:
    word_l.append(x[0])
    occurence_l.append(x[1])


word_occurence = pd.DataFrame({'word': word_l, 'occurence': occurence_l})



import nltk
nltk.download('popular')
from nltk.corpus import stopwords
stopwords = list(stopwords.words('english'))

  
filter_word = word_occurence[(~word_occurence['word'].isin(stopwords)) &
                             (word_occurence['occurence'] > 10) &
                             (word_occurence['word'].str.lower() != 'graffiti') &
                             (word_occurence['word'] != '')]  
filter_word = filter_word.sort_values(by = ['occurence'], ascending = False)

#plt.figure(figsize=(15,10))
#plt.bar(filter_word['word'], filter_word['occurence'], data = filter_word)


ax = filter_word.plot.bar(figsize=(20,10), color = '#9fc8ac')
ax.set_xticklabels(filter_word['word'], rotation = 45)
ax.set_title('Frequent Graffiti Places')

###### wordcloud
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

word_list2 = [x + ' ' for x in word_list if x not in stopwords and x.lower() != 'graffiti']
string = ''.join(word_list2)
wordcloud = WordCloud(background_color="white").generate(string)

# Display the generated image:
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


######################## duplicate_location (park/not park)
duplicate_location = graf_df['location_name'].value_counts().reset_index()
duplicate_location.columns = ['location name', 'count']

colors = []
for x in duplicate_location['location name']:
    if bool(re.match(r".*park", x.lower())):
        colors.append('#9f90ac')
    else:
        colors.append('#9fc8ac')
 
duplicate_location2 = duplicate_location[duplicate_location['count'] > 1]
duplicate_location2 = duplicate_location2.sort_values(by = ['count'], ascending = False)
plt.figure(figsize=(30,13))
plt.bar(duplicate_location2['location name'], duplicate_location2['count'], color = colors)
plt.xticks(rotation = 90)

import matplotlib.patches as mpatches
park = mpatches.Patch(color='#9f90ac', label='a park')
nopark = mpatches.Patch(color='#9fc8ac', label='not a park')                      
plt.legend(handles=[park, nopark])
plt.title('Duplicate Graffiti Location Name')
plt.xlable('location name')
plt.ylabel('count')



# =============================================================================
# ax = duplicate_location2.plot.bar(figsize=(20,10), color = colors)
# ax.set_xticklabels(duplicate_location2['location name'], rotation = 90)
# ax.set_title('Frequent Graffiti Places')
# =============================================================================

