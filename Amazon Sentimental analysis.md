# Amazon Coments Sentiment Analyzer 


```python
!pip install wordcloud
!pip install cufflinks
!pip install vaderSentiment
```

    Requirement already satisfied: wordcloud in c:\users\don_q\anaconda3\lib\site-packages (1.9.2)
    Requirement already satisfied: numpy>=1.6.1 in c:\users\don_q\anaconda3\lib\site-packages (from wordcloud) (1.21.5)
    Requirement already satisfied: pillow in c:\users\don_q\anaconda3\lib\site-packages (from wordcloud) (9.2.0)
    Requirement already satisfied: matplotlib in c:\users\don_q\anaconda3\lib\site-packages (from wordcloud) (3.5.2)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\don_q\anaconda3\lib\site-packages (from matplotlib->wordcloud) (4.25.0)
    Requirement already satisfied: cycler>=0.10 in c:\users\don_q\anaconda3\lib\site-packages (from matplotlib->wordcloud) (0.11.0)
    Requirement already satisfied: packaging>=20.0 in c:\users\don_q\anaconda3\lib\site-packages (from matplotlib->wordcloud) (21.3)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\don_q\anaconda3\lib\site-packages (from matplotlib->wordcloud) (1.4.2)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\don_q\anaconda3\lib\site-packages (from matplotlib->wordcloud) (2.8.2)
    Requirement already satisfied: pyparsing>=2.2.1 in c:\users\don_q\anaconda3\lib\site-packages (from matplotlib->wordcloud) (3.0.9)
    Requirement already satisfied: six>=1.5 in c:\users\don_q\anaconda3\lib\site-packages (from python-dateutil>=2.7->matplotlib->wordcloud) (1.16.0)
    Requirement already satisfied: cufflinks in c:\users\don_q\anaconda3\lib\site-packages (0.17.3)
    Requirement already satisfied: six>=1.9.0 in c:\users\don_q\anaconda3\lib\site-packages (from cufflinks) (1.16.0)
    Requirement already satisfied: ipython>=5.3.0 in c:\users\don_q\anaconda3\lib\site-packages (from cufflinks) (7.31.1)
    Requirement already satisfied: pandas>=0.19.2 in c:\users\don_q\anaconda3\lib\site-packages (from cufflinks) (1.4.4)
    Requirement already satisfied: ipywidgets>=7.0.0 in c:\users\don_q\anaconda3\lib\site-packages (from cufflinks) (7.6.5)
    Requirement already satisfied: plotly>=4.1.1 in c:\users\don_q\anaconda3\lib\site-packages (from cufflinks) (5.9.0)
    Requirement already satisfied: numpy>=1.9.2 in c:\users\don_q\anaconda3\lib\site-packages (from cufflinks) (1.21.5)
    Requirement already satisfied: setuptools>=34.4.1 in c:\users\don_q\anaconda3\lib\site-packages (from cufflinks) (63.4.1)
    Requirement already satisfied: colorlover>=0.2.1 in c:\users\don_q\anaconda3\lib\site-packages (from cufflinks) (0.3.0)
    Requirement already satisfied: jedi>=0.16 in c:\users\don_q\anaconda3\lib\site-packages (from ipython>=5.3.0->cufflinks) (0.18.1)
    Requirement already satisfied: traitlets>=4.2 in c:\users\don_q\anaconda3\lib\site-packages (from ipython>=5.3.0->cufflinks) (5.1.1)
    Requirement already satisfied: backcall in c:\users\don_q\anaconda3\lib\site-packages (from ipython>=5.3.0->cufflinks) (0.2.0)
    Requirement already satisfied: pickleshare in c:\users\don_q\anaconda3\lib\site-packages (from ipython>=5.3.0->cufflinks) (0.7.5)
    Requirement already satisfied: pygments in c:\users\don_q\anaconda3\lib\site-packages (from ipython>=5.3.0->cufflinks) (2.11.2)
    Requirement already satisfied: colorama in c:\users\don_q\anaconda3\lib\site-packages (from ipython>=5.3.0->cufflinks) (0.4.5)
    Requirement already satisfied: matplotlib-inline in c:\users\don_q\anaconda3\lib\site-packages (from ipython>=5.3.0->cufflinks) (0.1.6)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in c:\users\don_q\anaconda3\lib\site-packages (from ipython>=5.3.0->cufflinks) (3.0.20)
    Requirement already satisfied: decorator in c:\users\don_q\anaconda3\lib\site-packages (from ipython>=5.3.0->cufflinks) (5.1.1)
    Requirement already satisfied: nbformat>=4.2.0 in c:\users\don_q\anaconda3\lib\site-packages (from ipywidgets>=7.0.0->cufflinks) (5.5.0)
    Requirement already satisfied: ipython-genutils~=0.2.0 in c:\users\don_q\anaconda3\lib\site-packages (from ipywidgets>=7.0.0->cufflinks) (0.2.0)
    Requirement already satisfied: ipykernel>=4.5.1 in c:\users\don_q\anaconda3\lib\site-packages (from ipywidgets>=7.0.0->cufflinks) (6.15.2)
    Requirement already satisfied: jupyterlab-widgets>=1.0.0 in c:\users\don_q\anaconda3\lib\site-packages (from ipywidgets>=7.0.0->cufflinks) (1.0.0)
    Requirement already satisfied: widgetsnbextension~=3.5.0 in c:\users\don_q\anaconda3\lib\site-packages (from ipywidgets>=7.0.0->cufflinks) (3.5.2)
    Requirement already satisfied: python-dateutil>=2.8.1 in c:\users\don_q\anaconda3\lib\site-packages (from pandas>=0.19.2->cufflinks) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in c:\users\don_q\anaconda3\lib\site-packages (from pandas>=0.19.2->cufflinks) (2022.1)
    Requirement already satisfied: tenacity>=6.2.0 in c:\users\don_q\anaconda3\lib\site-packages (from plotly>=4.1.1->cufflinks) (8.0.1)
    Requirement already satisfied: debugpy>=1.0 in c:\users\don_q\anaconda3\lib\site-packages (from ipykernel>=4.5.1->ipywidgets>=7.0.0->cufflinks) (1.5.1)
    Requirement already satisfied: jupyter-client>=6.1.12 in c:\users\don_q\anaconda3\lib\site-packages (from ipykernel>=4.5.1->ipywidgets>=7.0.0->cufflinks) (7.3.4)
    Requirement already satisfied: packaging in c:\users\don_q\anaconda3\lib\site-packages (from ipykernel>=4.5.1->ipywidgets>=7.0.0->cufflinks) (21.3)
    Requirement already satisfied: nest-asyncio in c:\users\don_q\anaconda3\lib\site-packages (from ipykernel>=4.5.1->ipywidgets>=7.0.0->cufflinks) (1.5.5)
    Requirement already satisfied: pyzmq>=17 in c:\users\don_q\anaconda3\lib\site-packages (from ipykernel>=4.5.1->ipywidgets>=7.0.0->cufflinks) (23.2.0)
    Requirement already satisfied: psutil in c:\users\don_q\anaconda3\lib\site-packages (from ipykernel>=4.5.1->ipywidgets>=7.0.0->cufflinks) (5.9.0)
    Requirement already satisfied: tornado>=6.1 in c:\users\don_q\anaconda3\lib\site-packages (from ipykernel>=4.5.1->ipywidgets>=7.0.0->cufflinks) (6.1)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in c:\users\don_q\anaconda3\lib\site-packages (from jedi>=0.16->ipython>=5.3.0->cufflinks) (0.8.3)
    Requirement already satisfied: jsonschema>=2.6 in c:\users\don_q\anaconda3\lib\site-packages (from nbformat>=4.2.0->ipywidgets>=7.0.0->cufflinks) (4.16.0)
    Requirement already satisfied: jupyter_core in c:\users\don_q\anaconda3\lib\site-packages (from nbformat>=4.2.0->ipywidgets>=7.0.0->cufflinks) (4.11.1)
    Requirement already satisfied: fastjsonschema in c:\users\don_q\anaconda3\lib\site-packages (from nbformat>=4.2.0->ipywidgets>=7.0.0->cufflinks) (2.16.2)
    Requirement already satisfied: wcwidth in c:\users\don_q\anaconda3\lib\site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=5.3.0->cufflinks) (0.2.5)
    Requirement already satisfied: notebook>=4.4.1 in c:\users\don_q\anaconda3\lib\site-packages (from widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (6.4.12)
    Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in c:\users\don_q\anaconda3\lib\site-packages (from jsonschema>=2.6->nbformat>=4.2.0->ipywidgets>=7.0.0->cufflinks) (0.18.0)
    Requirement already satisfied: attrs>=17.4.0 in c:\users\don_q\anaconda3\lib\site-packages (from jsonschema>=2.6->nbformat>=4.2.0->ipywidgets>=7.0.0->cufflinks) (21.4.0)
    Requirement already satisfied: entrypoints in c:\users\don_q\anaconda3\lib\site-packages (from jupyter-client>=6.1.12->ipykernel>=4.5.1->ipywidgets>=7.0.0->cufflinks) (0.4)
    Requirement already satisfied: pywin32>=1.0 in c:\users\don_q\anaconda3\lib\site-packages (from jupyter_core->nbformat>=4.2.0->ipywidgets>=7.0.0->cufflinks) (302)
    Requirement already satisfied: argon2-cffi in c:\users\don_q\anaconda3\lib\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (21.3.0)
    Requirement already satisfied: terminado>=0.8.3 in c:\users\don_q\anaconda3\lib\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (0.13.1)
    Requirement already satisfied: nbconvert>=5 in c:\users\don_q\anaconda3\lib\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (6.4.4)
    Requirement already satisfied: Send2Trash>=1.8.0 in c:\users\don_q\anaconda3\lib\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (1.8.0)
    Requirement already satisfied: jinja2 in c:\users\don_q\anaconda3\lib\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (2.11.3)
    Requirement already satisfied: prometheus-client in c:\users\don_q\anaconda3\lib\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (0.14.1)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in c:\users\don_q\anaconda3\lib\site-packages (from packaging->ipykernel>=4.5.1->ipywidgets>=7.0.0->cufflinks) (3.0.9)
    Requirement already satisfied: pandocfilters>=1.4.1 in c:\users\don_q\anaconda3\lib\site-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (1.5.0)
    Requirement already satisfied: defusedxml in c:\users\don_q\anaconda3\lib\site-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (0.7.1)
    Requirement already satisfied: bleach in c:\users\don_q\anaconda3\lib\site-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (4.1.0)
    Requirement already satisfied: nbclient<0.6.0,>=0.5.0 in c:\users\don_q\anaconda3\lib\site-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (0.5.13)
    Requirement already satisfied: jupyterlab-pygments in c:\users\don_q\anaconda3\lib\site-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (0.1.2)
    Requirement already satisfied: mistune<2,>=0.8.1 in c:\users\don_q\anaconda3\lib\site-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (0.8.4)
    Requirement already satisfied: testpath in c:\users\don_q\anaconda3\lib\site-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (0.6.0)
    Requirement already satisfied: beautifulsoup4 in c:\users\don_q\anaconda3\lib\site-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (4.11.1)
    Requirement already satisfied: MarkupSafe>=0.23 in c:\users\don_q\anaconda3\lib\site-packages (from jinja2->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (2.0.1)
    Requirement already satisfied: pywinpty>=1.1.0 in c:\users\don_q\anaconda3\lib\site-packages (from terminado>=0.8.3->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (2.0.2)
    Requirement already satisfied: argon2-cffi-bindings in c:\users\don_q\anaconda3\lib\site-packages (from argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (21.2.0)
    Requirement already satisfied: cffi>=1.0.1 in c:\users\don_q\anaconda3\lib\site-packages (from argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (1.15.1)
    Requirement already satisfied: soupsieve>1.2 in c:\users\don_q\anaconda3\lib\site-packages (from beautifulsoup4->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (2.3.1)
    Requirement already satisfied: webencodings in c:\users\don_q\anaconda3\lib\site-packages (from bleach->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (0.5.1)
    Requirement already satisfied: pycparser in c:\users\don_q\anaconda3\lib\site-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->cufflinks) (2.21)
    Requirement already satisfied: vaderSentiment in c:\users\don_q\anaconda3\lib\site-packages (3.3.2)
    Requirement already satisfied: requests in c:\users\don_q\anaconda3\lib\site-packages (from vaderSentiment) (2.28.1)
    Requirement already satisfied: charset-normalizer<3,>=2 in c:\users\don_q\anaconda3\lib\site-packages (from requests->vaderSentiment) (2.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\don_q\anaconda3\lib\site-packages (from requests->vaderSentiment) (2022.9.14)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\don_q\anaconda3\lib\site-packages (from requests->vaderSentiment) (1.26.11)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\don_q\anaconda3\lib\site-packages (from requests->vaderSentiment) (3.3)
    


```python
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks as cf
%matplotlib inline
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)
cf.go_offline();
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import warnings 
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

pd.set_option('display.max_columns', None)
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.12.1.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.12.1.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




```python
df = pd.read_csv("C:/Users/don_q/Downloads/amazon.csv")
```


```python
df.head()
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
      <th>Unnamed: 0</th>
      <th>reviewerName</th>
      <th>overall</th>
      <th>reviewText</th>
      <th>reviewTime</th>
      <th>day_diff</th>
      <th>helpful_yes</th>
      <th>helpful_no</th>
      <th>total_vote</th>
      <th>score_pos_neg_diff</th>
      <th>score_average_rating</th>
      <th>wilson_lower_bound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>No issues.</td>
      <td>23-07-2014</td>
      <td>138</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0mie</td>
      <td>5</td>
      <td>Purchased this for my device, it worked as adv...</td>
      <td>25-10-2013</td>
      <td>409</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1K3</td>
      <td>4</td>
      <td>it works as expected. I should have sprung for...</td>
      <td>23-12-2012</td>
      <td>715</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1m2</td>
      <td>5</td>
      <td>This think has worked out great.Had a diff. br...</td>
      <td>21-11-2013</td>
      <td>382</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2&amp;amp;1/2Men</td>
      <td>5</td>
      <td>Bought it with Retail Packaging, arrived legit...</td>
      <td>13-07-2013</td>
      <td>513</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.sort_values("wilson_lower_bound", ascending = False)
df.drop('Unnamed: 0', inplace = True, axis=1)
df.head()
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
      <th>reviewerName</th>
      <th>overall</th>
      <th>reviewText</th>
      <th>reviewTime</th>
      <th>day_diff</th>
      <th>helpful_yes</th>
      <th>helpful_no</th>
      <th>total_vote</th>
      <th>score_pos_neg_diff</th>
      <th>score_average_rating</th>
      <th>wilson_lower_bound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2031</th>
      <td>Hyoun Kim "Faluzure"</td>
      <td>5</td>
      <td>[[ UPDATE - 6/19/2014 ]]So my lovely wife boug...</td>
      <td>05-01-2013</td>
      <td>702</td>
      <td>1952</td>
      <td>68</td>
      <td>2020</td>
      <td>1884</td>
      <td>0.966337</td>
      <td>0.957544</td>
    </tr>
    <tr>
      <th>3449</th>
      <td>NLee the Engineer</td>
      <td>5</td>
      <td>I have tested dozens of SDHC and micro-SDHC ca...</td>
      <td>26-09-2012</td>
      <td>803</td>
      <td>1428</td>
      <td>77</td>
      <td>1505</td>
      <td>1351</td>
      <td>0.948837</td>
      <td>0.936519</td>
    </tr>
    <tr>
      <th>4212</th>
      <td>SkincareCEO</td>
      <td>1</td>
      <td>NOTE:  please read the last update (scroll to ...</td>
      <td>08-05-2013</td>
      <td>579</td>
      <td>1568</td>
      <td>126</td>
      <td>1694</td>
      <td>1442</td>
      <td>0.925620</td>
      <td>0.912139</td>
    </tr>
    <tr>
      <th>317</th>
      <td>Amazon Customer "Kelly"</td>
      <td>1</td>
      <td>If your card gets hot enough to be painful, it...</td>
      <td>09-02-2012</td>
      <td>1033</td>
      <td>422</td>
      <td>73</td>
      <td>495</td>
      <td>349</td>
      <td>0.852525</td>
      <td>0.818577</td>
    </tr>
    <tr>
      <th>4672</th>
      <td>Twister</td>
      <td>5</td>
      <td>Sandisk announcement of the first 128GB micro ...</td>
      <td>03-07-2014</td>
      <td>158</td>
      <td>45</td>
      <td>4</td>
      <td>49</td>
      <td>41</td>
      <td>0.918367</td>
      <td>0.808109</td>
    </tr>
  </tbody>
</table>
</div>




```python
def missing_values_analysis(df):
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=True)
    ratio_ = (df[na_columns_].isnull().sum() / df.shape[0]* 100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis =1, keys=['Missing Values','Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df

def check_dataframe(df,head=5,tail=5):
    
    print("SHAPE".center(82,'~'))
    print('Rows: {}'.format(df.shape[0]))
    print('Columns: {}'.format(df.shape[1]))
    print('TYPES'.center(82,'~'))
    print (df.dtypes)
    print("".center(82, '~'))
    print(missing_values_analysis(df))
    print('DUPLICATED VALUES'.center(83,'~'))
    print(df.duplicated().sum())
    print("QUANTILES".center(82,'~'))
    print(df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    
check_dataframe(df)
```

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SHAPE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Rows: 4915
    Columns: 11
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TYPES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    reviewerName             object
    overall                   int64
    reviewText               object
    reviewTime               object
    day_diff                  int64
    helpful_yes               int64
    helpful_no                int64
    total_vote                int64
    score_pos_neg_diff        int64
    score_average_rating    float64
    wilson_lower_bound      float64
    dtype: object
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                  Missing Values  Ratio
    reviewerName               1   0.02
    reviewText                 1   0.02
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DUPLICATED VALUES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    0
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~QUANTILES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                           0.00  0.05   0.50        0.95       0.99         1.00
    overall                 1.0   2.0    5.0    5.000000    5.00000     5.000000
    day_diff                1.0  98.0  431.0  748.000000  943.00000  1064.000000
    helpful_yes             0.0   0.0    0.0    1.000000    3.00000  1952.000000
    helpful_no              0.0   0.0    0.0    0.000000    2.00000   183.000000
    total_vote              0.0   0.0    0.0    1.000000    4.00000  2020.000000
    score_pos_neg_diff   -130.0   0.0    0.0    1.000000    2.00000  1884.000000
    score_average_rating    0.0   0.0    0.0    1.000000    1.00000     1.000000
    wilson_lower_bound      0.0   0.0    0.0    0.206549    0.34238     0.957544
    


```python
def check_class(dataframe):
    nunique_df = pd.DataFrame({'Variable':dataframe.columns,
                              'Classes':[dataframe[i].nunique()\
                                        for i in dataframe.columns]})
    nunique_df = nunique_df.sort_values('Classes', ascending = False)
    nunique_df = nunique_df.reset_index(drop = True)
    return nunique_df
check_class(df)
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
      <th>Variable</th>
      <th>Classes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>reviewText</td>
      <td>4912</td>
    </tr>
    <tr>
      <th>1</th>
      <td>reviewerName</td>
      <td>4594</td>
    </tr>
    <tr>
      <th>2</th>
      <td>reviewTime</td>
      <td>690</td>
    </tr>
    <tr>
      <th>3</th>
      <td>day_diff</td>
      <td>690</td>
    </tr>
    <tr>
      <th>4</th>
      <td>wilson_lower_bound</td>
      <td>40</td>
    </tr>
    <tr>
      <th>5</th>
      <td>score_average_rating</td>
      <td>28</td>
    </tr>
    <tr>
      <th>6</th>
      <td>score_pos_neg_diff</td>
      <td>27</td>
    </tr>
    <tr>
      <th>7</th>
      <td>total_vote</td>
      <td>26</td>
    </tr>
    <tr>
      <th>8</th>
      <td>helpful_yes</td>
      <td>23</td>
    </tr>
    <tr>
      <th>9</th>
      <td>helpful_no</td>
      <td>17</td>
    </tr>
    <tr>
      <th>10</th>
      <td>overall</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
constraints = ['#00FF7F','#BF3EFF','#00FFFF','#FF1493','#FFFF00']
def categorical_variable_summary(df, column_name):
    fig = make_subplots(rows = 1, cols = 2,
                       subplot_titles=('Countplot', 'Percentage'),
                        specs=[[{"type": "xy"},{'type':'domain'}]])
    
    fig.add_trace(go.Bar( y = df[column_name].value_counts().values.tolist(),
                        x = [str(i) for i in df[column_name].value_counts().index],
                        text = df[column_name].value_counts().values.tolist(),
                        textfont = dict(size=14),
                        name = column_name,
                        textposition = 'auto',
                        showlegend = False,
                        marker= dict(color = constraints,
                                    line= dict(color='#DBE6EC',
                                              width =1))),
                 row = 1, col = 1)
    fig.add_trace(go.Pie(labels = df[column_name].value_counts().keys(),
                        values = df[column_name].value_counts().values,
                        textfont = dict(size=18),
                        textposition = 'auto',
                        showlegend = False,
                        name = column_name,
                        marker=dict(colors=constraints)),
                 row = 1, col = 2)
    
    fig.update_layout(title={'text': column_name,
                            'y':0.9,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'},
                     template='plotly_white')
    iplot(fig)
```


```python
categorical_variable_summary(df, 'overall')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_21636\2128563884.py in <module>
    ----> 1 categorical_variable_summary(df, 'overall')
    

    NameError: name 'categorical_variable_summary' is not defined



```python
df.reviewText.head()
```




    2031    [[ UPDATE - 6/19/2014 ]]So my lovely wife boug...
    3449    I have tested dozens of SDHC and micro-SDHC ca...
    4212    NOTE:  please read the last update (scroll to ...
    317     If your card gets hot enough to be painful, it...
    4672    Sandisk announcement of the first 128GB micro ...
    Name: reviewText, dtype: object




```python
review_example = df.reviewText[2031]
review_example
```




    '[[ UPDATE - 6/19/2014 ]]So my lovely wife bought me a Samsung Galaxy Tab 4 for Father\'s Day and I\'ve been loving it ever since.  Just as other with Samsung products, the Galaxy Tab 4 has the ability to add a microSD card to expand the memory on the device.  Since it\'s been over a year, I decided to do some more research to see if SanDisk offered anything new.  As of 6/19/2014, their product lineup for microSD cards from worst to best (performance-wise) are the as follows:SanDiskSanDisk UltraSanDisk Ultra PLUSSanDisk ExtremeSanDisk Extreme PLUSSanDisk Extreme PRONow, the difference between all of these cards are simply the speed in which you can read/write data to the card.  Yes, the published rating of most all these cards (except the SanDisk regular) are Class 10/UHS-I but that\'s just a rating... Actual real world performance does get better with each model, but with faster cards come more expensive prices.  Since Amazon doesn\'t carry the Ultra PLUS model of microSD card, I had to do direct comparisons between the SanDisk Ultra ($34.27), Extreme ($57.95), and Extreme PLUS ($67.95).As mentioned in my earlier review, I purchased the SanDisk Ultra for my Galaxy S4.  My question was, did I want to pay over $20 more for a card that is faster than the one I already owned?  Or I could pay almost double to get SanDisk\'s 2nd-most fastest microSD card.The Ultra works perfectly fine for my style of usage (storing/capturing pictures & HD video and movie playback) on my phone.  So in the end, I ended up just buying another SanDisk Ultra 64GB card.  I use my cell phone *more* than I do my tablet and if the card is good enough for my phone, it\'s good enough for my tablet.  I don\'t own a 4K HD camera or anything like that, so I honestly didn\'t see a need to get one of the faster cards at this time.I am now a proud owner of 2 SanDisk Ultra cards and have absolutely 0 issues with it in my Samsung devices.[[ ORIGINAL REVIEW - 5/1/2013 ]]I haven\'t had to buy a microSD card in a long time. The last time I bought one was for my cell phone over 2 years ago. But since my cellular contract was up, I knew I would have to get a newer card in addition to my new phone, the Samsung Galaxy S4. Reason for this is because I knew my small 16GB microSD card wasn\'t going to cut it.Doing research on the Galaxy S4, I wanted to get the best card possible that had decent capacity (32 GB or greater). This led me to find that the Galaxy S4 supports the microSDXC Class 10 UHS-I card, which is the fastest possible given that class. Searching for that specifically on Amazon gave me results of only 3 vendors (as of April) that makes these microSDXC Class 10 UHS-1 cards. They are Sandisk (the majority), Samsung and Lexar. Nobody else makes these that are sold on Amazon.Seeing how SanDisk is a pretty good name out of the 3 (I\'ve used them the most), I decided upon the SanDisk because Lexar was overpriced and the Samsung one was overpriced (as well as not eligible for Amazon Prime).But the scary thing is that when you filter by the SanDisk, you literally get DOZENS of options. All of them have different model numbers, different sizes, etc. Then there\'s that confusion of what\'s the difference between SDHC & SDXC?SDHC vs SDXC:SDHC stand for "Secure Digital High Capacity" and SDXC stands for "Secure Digital eXtended Capacity". Essentially these two cards are the same with the exception that SDHC only supports capcities up to 32GB and is formated with the FAT32 file system. The SDXC cards are formatted with the exFAT file system. If you use an SDXC card in a device, it must support that file system, otherwise it may not be recognizable and/or you have to reformat the card to FAT32.FAT32 vs exFAT:The differences between the two file systems means that FAT32 has a maximum file size of 4GB, limited by that file system. exFAT on the otherhand, supports file sizes up to 2TB (terabytes). The only thing you need to know here really is that it\'s possible your device doesn\'t support exFAT. If that\'s the case, just reformat it to FAT32. REMEMBER FORMATTING ERASES ALL DATA!To clarify the model numbers, I I hopped over to the SanDisk official webpage. What I found there is that they offer two "highspeed" options for SanDisk cards. These are SanDisk Extreme Pro and SanDisk Ultra. SanDisk Extreme Pro is a line that supports read speeds up to 95MB/sec, however they are SDHC only. To make things worse, they are currently only available in 16GB & 8GB capacities. Since one of my requirements was to have a lot of storage, I ruled these out.The remaining devices listed on Amazon\'s search were the SanDisk Ultra line. But here, confusion sets in because SanDisk separates these cards to two different devices. Cameras & mobile devices. Is there a real difference between the two or is this just a marketing stunt? Unfortunately I\'m not sure but I do know the price difference between the two range from a couple cents to a few dollars. Since I wasn\'t sure, I opted for the one specifically targeted for mobile devices (just in case there is some kind of compatibility issue). To find the exact model number, I would go to Sandisk\'s webpage (sandisk.com) and compare their existing product lineup. From there, you get exact model numbers and you can then search Amazon for these model numbers. That is how I got mine (SDSDQUA-064G).As for speed tests, I haven\'t run any specific testing, but copying 8 GB worth of data from my PC to the card literally took just a few minutes.One last note is that Amazon attaches additional characters to the end (for example SDSDQUA-064G-AFFP-A vs SDSDQUA-064G-U46A). The difference between the two is that the "AFFP-A" means "Amazon Frustration Free Packaging". Other than that, these are exactly the same.  If you\'re wondering what I got (and want to use it in your Galaxy S4), I got the SDSDQUA-064G-u46A and it works like charm.'




```python
review_example = re.sub("[^a-zA-Z]",'',review_example)
review_example
```




    'UPDATESomylovelywifeboughtmeaSamsungGalaxyTabforFathersDayandIvebeenlovingiteversinceJustasotherwithSamsungproductstheGalaxyTabhastheabilitytoaddamicroSDcardtoexpandthememoryonthedeviceSinceitsbeenoverayearIdecidedtodosomemoreresearchtoseeifSanDiskofferedanythingnewAsoftheirproductlineupformicroSDcardsfromworsttobestperformancewisearetheasfollowsSanDiskSanDiskUltraSanDiskUltraPLUSSanDiskExtremeSanDiskExtremePLUSSanDiskExtremePRONowthedifferencebetweenallofthesecardsaresimplythespeedinwhichyoucanreadwritedatatothecardYesthepublishedratingofmostallthesecardsexcepttheSanDiskregularareClassUHSIbutthatsjustaratingActualrealworldperformancedoesgetbetterwitheachmodelbutwithfastercardscomemoreexpensivepricesSinceAmazondoesntcarrytheUltraPLUSmodelofmicroSDcardIhadtododirectcomparisonsbetweentheSanDiskUltraExtremeandExtremePLUSAsmentionedinmyearlierreviewIpurchasedtheSanDiskUltraformyGalaxySMyquestionwasdidIwanttopayovermoreforacardthatisfasterthantheoneIalreadyownedOrIcouldpayalmostdoubletogetSanDisksndmostfastestmicroSDcardTheUltraworksperfectlyfineformystyleofusagestoringcapturingpicturesHDvideoandmovieplaybackonmyphoneSointheendIendedupjustbuyinganotherSanDiskUltraGBcardIusemycellphonemorethanIdomytabletandifthecardisgoodenoughformyphoneitsgoodenoughformytabletIdontownaKHDcameraoranythinglikethatsoIhonestlydidntseeaneedtogetoneofthefastercardsatthistimeIamnowaproudownerofSanDiskUltracardsandhaveabsolutelyissueswithitinmySamsungdevicesORIGINALREVIEWIhaventhadtobuyamicroSDcardinalongtimeThelasttimeIboughtonewasformycellphoneoveryearsagoButsincemycellularcontractwasupIknewIwouldhavetogetanewercardinadditiontomynewphonetheSamsungGalaxySReasonforthisisbecauseIknewmysmallGBmicroSDcardwasntgoingtocutitDoingresearchontheGalaxySIwantedtogetthebestcardpossiblethathaddecentcapacityGBorgreaterThisledmetofindthattheGalaxySsupportsthemicroSDXCClassUHSIcardwhichisthefastestpossiblegiventhatclassSearchingforthatspecificallyonAmazongavemeresultsofonlyvendorsasofAprilthatmakesthesemicroSDXCClassUHScardsTheyareSandiskthemajoritySamsungandLexarNobodyelsemakesthesethataresoldonAmazonSeeinghowSanDiskisaprettygoodnameoutoftheIveusedthemthemostIdecidedupontheSanDiskbecauseLexarwasoverpricedandtheSamsungonewasoverpricedaswellasnoteligibleforAmazonPrimeButthescarythingisthatwhenyoufilterbytheSanDiskyouliterallygetDOZENSofoptionsAllofthemhavedifferentmodelnumbersdifferentsizesetcThentheresthatconfusionofwhatsthedifferencebetweenSDHCSDXCSDHCvsSDXCSDHCstandforSecureDigitalHighCapacityandSDXCstandsforSecureDigitaleXtendedCapacityEssentiallythesetwocardsarethesamewiththeexceptionthatSDHConlysupportscapcitiesuptoGBandisformatedwiththeFATfilesystemTheSDXCcardsareformattedwiththeexFATfilesystemIfyouuseanSDXCcardinadeviceitmustsupportthatfilesystemotherwiseitmaynotberecognizableandoryouhavetoreformatthecardtoFATFATvsexFATThedifferencesbetweenthetwofilesystemsmeansthatFAThasamaximumfilesizeofGBlimitedbythatfilesystemexFATontheotherhandsupportsfilesizesuptoTBterabytesTheonlythingyouneedtoknowherereallyisthatitspossibleyourdevicedoesntsupportexFATIfthatsthecasejustreformatittoFATREMEMBERFORMATTINGERASESALLDATAToclarifythemodelnumbersIIhoppedovertotheSanDiskofficialwebpageWhatIfoundthereisthattheyoffertwohighspeedoptionsforSanDiskcardsTheseareSanDiskExtremeProandSanDiskUltraSanDiskExtremeProisalinethatsupportsreadspeedsuptoMBsechowevertheyareSDHConlyTomakethingsworsetheyarecurrentlyonlyavailableinGBGBcapacitiesSinceoneofmyrequirementswastohavealotofstorageIruledtheseoutTheremainingdeviceslistedonAmazonssearchweretheSanDiskUltralineButhereconfusionsetsinbecauseSanDiskseparatesthesecardstotwodifferentdevicesCamerasmobiledevicesIstherearealdifferencebetweenthetwooristhisjustamarketingstuntUnfortunatelyImnotsurebutIdoknowthepricedifferencebetweenthetworangefromacouplecentstoafewdollarsSinceIwasntsureIoptedfortheonespecificallytargetedformobiledevicesjustincasethereissomekindofcompatibilityissueTofindtheexactmodelnumberIwouldgotoSandiskswebpagesandiskcomandcomparetheirexistingproductlineupFromthereyougetexactmodelnumbersandyoucanthensearchAmazonforthesemodelnumbersThatishowIgotmineSDSDQUAGAsforspeedtestsIhaventrunanyspecifictestingbutcopyingGBworthofdatafrommyPCtothecardliterallytookjustafewminutesOnelastnoteisthatAmazonattachesadditionalcharacterstotheendforexampleSDSDQUAGAFFPAvsSDSDQUAGUAThedifferencebetweenthetwoisthattheAFFPAmeansAmazonFrustrationFreePackagingOtherthanthattheseareexactlythesameIfyourewonderingwhatIgotandwanttouseitinyourGalaxySIgottheSDSDQUAGuAanditworkslikecharm'




```python
review_example = review_example.lower().split()
review_example
```




    ['updatesomylovelywifeboughtmeasamsunggalaxytabforfathersdayandivebeenlovingiteversincejustasotherwithsamsungproductsthegalaxytabhastheabilitytoaddamicrosdcardtoexpandthememoryonthedevicesinceitsbeenoverayearidecidedtodosomemoreresearchtoseeifsandiskofferedanythingnewasoftheirproductlineupformicrosdcardsfromworsttobestperformancewisearetheasfollowssandisksandiskultrasandiskultraplussandiskextremesandiskextremeplussandiskextremepronowthedifferencebetweenallofthesecardsaresimplythespeedinwhichyoucanreadwritedatatothecardyesthepublishedratingofmostallthesecardsexceptthesandiskregularareclassuhsibutthatsjustaratingactualrealworldperformancedoesgetbetterwitheachmodelbutwithfastercardscomemoreexpensivepricessinceamazondoesntcarrytheultraplusmodelofmicrosdcardihadtododirectcomparisonsbetweenthesandiskultraextremeandextremeplusasmentionedinmyearlierreviewipurchasedthesandiskultraformygalaxysmyquestionwasdidiwanttopayovermoreforacardthatisfasterthantheoneialreadyownedoricouldpayalmostdoubletogetsandisksndmostfastestmicrosdcardtheultraworksperfectlyfineformystyleofusagestoringcapturingpictureshdvideoandmovieplaybackonmyphonesointheendiendedupjustbuyinganothersandiskultragbcardiusemycellphonemorethanidomytabletandifthecardisgoodenoughformyphoneitsgoodenoughformytabletidontownakhdcameraoranythinglikethatsoihonestlydidntseeaneedtogetoneofthefastercardsatthistimeiamnowaproudownerofsandiskultracardsandhaveabsolutelyissueswithitinmysamsungdevicesoriginalreviewihaventhadtobuyamicrosdcardinalongtimethelasttimeiboughtonewasformycellphoneoveryearsagobutsincemycellularcontractwasupiknewiwouldhavetogetanewercardinadditiontomynewphonethesamsunggalaxysreasonforthisisbecauseiknewmysmallgbmicrosdcardwasntgoingtocutitdoingresearchonthegalaxysiwantedtogetthebestcardpossiblethathaddecentcapacitygborgreaterthisledmetofindthatthegalaxyssupportsthemicrosdxcclassuhsicardwhichisthefastestpossiblegiventhatclasssearchingforthatspecificallyonamazongavemeresultsofonlyvendorsasofaprilthatmakesthesemicrosdxcclassuhscardstheyaresandiskthemajoritysamsungandlexarnobodyelsemakesthesethataresoldonamazonseeinghowsandiskisaprettygoodnameoutoftheiveusedthemthemostidecideduponthesandiskbecauselexarwasoverpricedandthesamsungonewasoverpricedaswellasnoteligibleforamazonprimebutthescarythingisthatwhenyoufilterbythesandiskyouliterallygetdozensofoptionsallofthemhavedifferentmodelnumbersdifferentsizesetcthentheresthatconfusionofwhatsthedifferencebetweensdhcsdxcsdhcvssdxcsdhcstandforsecuredigitalhighcapacityandsdxcstandsforsecuredigitalextendedcapacityessentiallythesetwocardsarethesamewiththeexceptionthatsdhconlysupportscapcitiesuptogbandisformatedwiththefatfilesystemthesdxccardsareformattedwiththeexfatfilesystemifyouuseansdxccardinadeviceitmustsupportthatfilesystemotherwiseitmaynotberecognizableandoryouhavetoreformatthecardtofatfatvsexfatthedifferencesbetweenthetwofilesystemsmeansthatfathasamaximumfilesizeofgblimitedbythatfilesystemexfatontheotherhandsupportsfilesizesuptotbterabytestheonlythingyouneedtoknowherereallyisthatitspossibleyourdevicedoesntsupportexfatifthatsthecasejustreformatittofatrememberformattingerasesalldatatoclarifythemodelnumbersiihoppedovertothesandiskofficialwebpagewhatifoundthereisthattheyoffertwohighspeedoptionsforsandiskcardsthesearesandiskextremeproandsandiskultrasandiskextremeproisalinethatsupportsreadspeedsuptombsechowevertheyaresdhconlytomakethingsworsetheyarecurrentlyonlyavailableingbgbcapacitiessinceoneofmyrequirementswastohavealotofstorageiruledtheseouttheremainingdeviceslistedonamazonssearchwerethesandiskultralinebuthereconfusionsetsinbecausesandiskseparatesthesecardstotwodifferentdevicescamerasmobiledevicesistherearealdifferencebetweenthetwooristhisjustamarketingstuntunfortunatelyimnotsurebutidoknowthepricedifferencebetweenthetworangefromacouplecentstoafewdollarssinceiwasntsureioptedfortheonespecificallytargetedformobiledevicesjustincasethereissomekindofcompatibilityissuetofindtheexactmodelnumberiwouldgotosandiskswebpagesandiskcomandcomparetheirexistingproductlineupfromthereyougetexactmodelnumbersandyoucanthensearchamazonforthesemodelnumbersthatishowigotminesdsdquagasforspeedtestsihaventrunanyspecifictestingbutcopyinggbworthofdatafrommypctothecardliterallytookjustafewminutesonelastnoteisthatamazonattachesadditionalcharacterstotheendforexamplesdsdquagaffpavssdsdquaguathedifferencebetweenthetwoisthattheaffpameansamazonfrustrationfreepackagingotherthanthattheseareexactlythesameifyourewonderingwhatigotandwanttouseitinyourgalaxysigotthesdsdquaguaanditworkslikecharm']




```python
rt = lambda x: re.sub("[^a-zA-Z]", ' ',str(x))
df["reviewText"] = df["reviewText"].map(rt)
df["reviewText"] = df["reviewText"].str.lower()
df.head()
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
      <th>reviewerName</th>
      <th>overall</th>
      <th>reviewText</th>
      <th>reviewTime</th>
      <th>day_diff</th>
      <th>helpful_yes</th>
      <th>helpful_no</th>
      <th>total_vote</th>
      <th>score_pos_neg_diff</th>
      <th>score_average_rating</th>
      <th>wilson_lower_bound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2031</th>
      <td>Hyoun Kim "Faluzure"</td>
      <td>5</td>
      <td>updatesomylovelywifeboughtmeasamsunggalaxytabf...</td>
      <td>05-01-2013</td>
      <td>702</td>
      <td>1952</td>
      <td>68</td>
      <td>2020</td>
      <td>1884</td>
      <td>0.966337</td>
      <td>0.957544</td>
    </tr>
    <tr>
      <th>3449</th>
      <td>NLee the Engineer</td>
      <td>5</td>
      <td>ihavetesteddozensofsdhcandmicrosdhccardsonedis...</td>
      <td>26-09-2012</td>
      <td>803</td>
      <td>1428</td>
      <td>77</td>
      <td>1505</td>
      <td>1351</td>
      <td>0.948837</td>
      <td>0.936519</td>
    </tr>
    <tr>
      <th>4212</th>
      <td>SkincareCEO</td>
      <td>1</td>
      <td>notepleasereadthelastupdatescrolltothebottomim...</td>
      <td>08-05-2013</td>
      <td>579</td>
      <td>1568</td>
      <td>126</td>
      <td>1694</td>
      <td>1442</td>
      <td>0.925620</td>
      <td>0.912139</td>
    </tr>
    <tr>
      <th>317</th>
      <td>Amazon Customer "Kelly"</td>
      <td>1</td>
      <td>ifyourcardgetshotenoughtobepainfulitisdefectiv...</td>
      <td>09-02-2012</td>
      <td>1033</td>
      <td>422</td>
      <td>73</td>
      <td>495</td>
      <td>349</td>
      <td>0.852525</td>
      <td>0.818577</td>
    </tr>
    <tr>
      <th>4672</th>
      <td>Twister</td>
      <td>5</td>
      <td>sandiskannouncementofthefirstgbmicrosdtookinte...</td>
      <td>03-07-2014</td>
      <td>158</td>
      <td>45</td>
      <td>4</td>
      <td>49</td>
      <td>41</td>
      <td>0.918367</td>
      <td>0.808109</td>
    </tr>
  </tbody>
</table>
</div>




```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
```


```python

df[['polarity', 'subjectivity']] = df['reviewText'].apply(lambda Text:pd.Series(TextBlob(Text).sentiment))

for index, row in df['reviewText'].iteritems():
    
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    if neg > pos:
        df.loc[index, 'sentiment'] = "Negative"
    elif pos > neg:
        df.loc[index, 'sentiment'] = "Positive"
    else:
        df.loc[index, 'sentiment'] = "Neutral"
```


```python
df[df['sentiment']=='Positive'].sort_values("wilson_lower_bound",
                                           ascending= False).head(15)
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
      <th>reviewerName</th>
      <th>overall</th>
      <th>reviewText</th>
      <th>reviewTime</th>
      <th>day_diff</th>
      <th>helpful_yes</th>
      <th>helpful_no</th>
      <th>total_vote</th>
      <th>score_pos_neg_diff</th>
      <th>score_average_rating</th>
      <th>wilson_lower_bound</th>
      <th>polarity</th>
      <th>subjectivity</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3234</th>
      <td>minh thong cao</td>
      <td>5</td>
      <td>good</td>
      <td>07-07-2014</td>
      <td>154</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.700000</td>
      <td>0.600000</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>3605</th>
      <td>peter Metcalf</td>
      <td>5</td>
      <td>perfect</td>
      <td>07-02-2014</td>
      <td>304</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>4308</th>
      <td>Stephane Gauthier</td>
      <td>5</td>
      <td>super</td>
      <td>15-02-2013</td>
      <td>661</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.333333</td>
      <td>0.666667</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>3741</th>
      <td>RASHAWN</td>
      <td>5</td>
      <td>great</td>
      <td>07-12-2014</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.800000</td>
      <td>0.750000</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>705</th>
      <td>Brandon Warren</td>
      <td>5</td>
      <td>yes</td>
      <td>14-07-2014</td>
      <td>147</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>2496</th>
      <td>Josh H.</td>
      <td>5</td>
      <td>yes</td>
      <td>13-07-2014</td>
      <td>148</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>Positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
categorical_variable_summary(df, 'sentiment')
```


<div>                            <div id="2120df15-cb0a-46b5-954a-58014c1e49cb" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("2120df15-cb0a-46b5-954a-58014c1e49cb")) {                    Plotly.newPlot(                        "2120df15-cb0a-46b5-954a-58014c1e49cb",                        [{"marker":{"color":["#00FF7F","#BF3EFF","#00FFFF","#FF1493","#FFFF00"],"line":{"color":"#DBE6EC","width":1}},"name":"sentiment","showlegend":false,"text":["4909","6"],"textfont":{"size":14},"textposition":"auto","x":["Neutral","Positive"],"xaxis":"x","y":[4909,6],"yaxis":"y","type":"bar"},{"domain":{"x":[0.55,1.0],"y":[0.0,1.0]},"labels":["Neutral","Positive"],"marker":{"colors":["#00FF7F","#BF3EFF","#00FFFF","#FF1493","#FFFF00"]},"name":"sentiment","showlegend":false,"textfont":{"size":18},"textposition":"auto","values":[4909,6],"type":"pie"}],                        {"annotations":[{"font":{"size":16},"showarrow":false,"text":"Countplot","x":0.225,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"},{"font":{"size":16},"showarrow":false,"text":"Percentage","x":0.775,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"}],"template":{"data":{"barpolar":[{"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"#C8D4E3","linecolor":"#C8D4E3","minorgridcolor":"#C8D4E3","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"#C8D4E3","linecolor":"#C8D4E3","minorgridcolor":"#C8D4E3","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"white","showlakes":true,"showland":true,"subunitcolor":"#C8D4E3"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"white","polar":{"angularaxis":{"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":""},"bgcolor":"white","radialaxis":{"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"white","gridcolor":"#DFE8F3","gridwidth":2,"linecolor":"#EBF0F8","showbackground":true,"ticks":"","zerolinecolor":"#EBF0F8"},"yaxis":{"backgroundcolor":"white","gridcolor":"#DFE8F3","gridwidth":2,"linecolor":"#EBF0F8","showbackground":true,"ticks":"","zerolinecolor":"#EBF0F8"},"zaxis":{"backgroundcolor":"white","gridcolor":"#DFE8F3","gridwidth":2,"linecolor":"#EBF0F8","showbackground":true,"ticks":"","zerolinecolor":"#EBF0F8"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"#DFE8F3","linecolor":"#A2B1C6","ticks":""},"baxis":{"gridcolor":"#DFE8F3","linecolor":"#A2B1C6","ticks":""},"bgcolor":"white","caxis":{"gridcolor":"#DFE8F3","linecolor":"#A2B1C6","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":"","title":{"standoff":15},"zerolinecolor":"#EBF0F8","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":"","title":{"standoff":15},"zerolinecolor":"#EBF0F8","zerolinewidth":2}}},"title":{"text":"sentiment","x":0.5,"xanchor":"center","y":0.9,"yanchor":"top"},"xaxis":{"anchor":"y","domain":[0.0,0.45]},"yaxis":{"anchor":"x","domain":[0.0,1.0]}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('2120df15-cb0a-46b5-954a-58014c1e49cb');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python

```
