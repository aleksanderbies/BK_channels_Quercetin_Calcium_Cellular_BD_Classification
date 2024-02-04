```python
import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns 

import plotly.express as px
import plotly.graph_objects as go

from modules.plots import draw_plot
```


```python
df_calcium_clf = pd.read_csv('./prepared_datasets/calcium_classification.csv')
df_calcium_concentration_clf = pd.read_csv('./prepared_datasets/calcium_concentration_classification.csv')
df_quercetin_clf = pd.read_csv('./prepared_datasets/quercetin_classification.csv')
df_quercetin_concentration_clf = pd.read_csv('./prepared_datasets/quercetin_concentration_classification.csv')
```


```python
df_calcium_clf
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
      <th>f_0</th>
      <th>f_1</th>
      <th>f_2</th>
      <th>f_3</th>
      <th>f_4</th>
      <th>f_5</th>
      <th>f_6</th>
      <th>f_7</th>
      <th>f_8</th>
      <th>f_9</th>
      <th>...</th>
      <th>f_4991</th>
      <th>f_4992</th>
      <th>f_4993</th>
      <th>f_4994</th>
      <th>f_4995</th>
      <th>f_4996</th>
      <th>f_4997</th>
      <th>f_4998</th>
      <th>f_4999</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.635397</td>
      <td>13.077894</td>
      <td>13.062591</td>
      <td>12.894789</td>
      <td>12.650586</td>
      <td>12.711683</td>
      <td>12.360680</td>
      <td>12.925277</td>
      <td>13.398274</td>
      <td>13.535571</td>
      <td>...</td>
      <td>13.231437</td>
      <td>13.399334</td>
      <td>13.155131</td>
      <td>13.643429</td>
      <td>13.521326</td>
      <td>13.353523</td>
      <td>13.704420</td>
      <td>14.391117</td>
      <td>13.735014</td>
      <td>No_Ca</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.651611</td>
      <td>13.124609</td>
      <td>13.612906</td>
      <td>13.597603</td>
      <td>13.216200</td>
      <td>13.444997</td>
      <td>13.246694</td>
      <td>12.010691</td>
      <td>12.575289</td>
      <td>12.865186</td>
      <td>...</td>
      <td>12.774651</td>
      <td>13.583349</td>
      <td>13.018846</td>
      <td>12.622043</td>
      <td>13.003540</td>
      <td>13.232437</td>
      <td>12.988234</td>
      <td>13.064531</td>
      <td>12.866229</td>
      <td>No_Ca</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.942526</td>
      <td>13.339223</td>
      <td>12.881420</td>
      <td>12.606817</td>
      <td>12.957714</td>
      <td>12.957711</td>
      <td>12.789909</td>
      <td>12.469406</td>
      <td>13.049303</td>
      <td>13.095100</td>
      <td>...</td>
      <td>13.462266</td>
      <td>13.447063</td>
      <td>12.897760</td>
      <td>12.897757</td>
      <td>13.569054</td>
      <td>12.897751</td>
      <td>12.745149</td>
      <td>12.928246</td>
      <td>13.035043</td>
      <td>No_Ca</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.912940</td>
      <td>13.035037</td>
      <td>12.638334</td>
      <td>12.577231</td>
      <td>12.592529</td>
      <td>13.248626</td>
      <td>13.111323</td>
      <td>13.065520</td>
      <td>13.660617</td>
      <td>13.660614</td>
      <td>...</td>
      <td>13.371680</td>
      <td>13.432777</td>
      <td>13.631074</td>
      <td>13.585371</td>
      <td>13.035969</td>
      <td>12.730866</td>
      <td>13.570063</td>
      <td>14.592360</td>
      <td>14.104157</td>
      <td>No_Ca</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.310654</td>
      <td>13.371651</td>
      <td>13.997249</td>
      <td>14.210946</td>
      <td>13.798943</td>
      <td>13.936240</td>
      <td>13.539537</td>
      <td>13.646334</td>
      <td>13.463231</td>
      <td>13.264829</td>
      <td>...</td>
      <td>13.494794</td>
      <td>13.250591</td>
      <td>12.930189</td>
      <td>12.701286</td>
      <td>12.518183</td>
      <td>13.036980</td>
      <td>13.632077</td>
      <td>13.235374</td>
      <td>13.143771</td>
      <td>No_Ca</td>
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
      <th>10695</th>
      <td>33.633477</td>
      <td>32.305884</td>
      <td>31.268290</td>
      <td>32.336497</td>
      <td>33.099404</td>
      <td>34.304811</td>
      <td>33.831817</td>
      <td>32.885824</td>
      <td>32.550131</td>
      <td>32.702737</td>
      <td>...</td>
      <td>32.247823</td>
      <td>32.766629</td>
      <td>33.438036</td>
      <td>33.132843</td>
      <td>32.812449</td>
      <td>33.346456</td>
      <td>33.224363</td>
      <td>32.751370</td>
      <td>32.049476</td>
      <td>Ca</td>
    </tr>
    <tr>
      <th>10696</th>
      <td>32.141083</td>
      <td>32.446190</td>
      <td>33.193896</td>
      <td>33.163403</td>
      <td>32.034310</td>
      <td>31.439216</td>
      <td>32.171623</td>
      <td>32.568330</td>
      <td>32.781937</td>
      <td>32.415743</td>
      <td>...</td>
      <td>25.536929</td>
      <td>29.992535</td>
      <td>32.372842</td>
      <td>32.800149</td>
      <td>33.242655</td>
      <td>32.952762</td>
      <td>33.685169</td>
      <td>33.563075</td>
      <td>33.273182</td>
      <td>Ca</td>
    </tr>
    <tr>
      <th>10697</th>
      <td>30.114589</td>
      <td>30.572396</td>
      <td>32.144102</td>
      <td>32.601809</td>
      <td>30.953916</td>
      <td>31.640522</td>
      <td>32.861229</td>
      <td>32.708636</td>
      <td>32.601842</td>
      <td>33.135949</td>
      <td>...</td>
      <td>22.259235</td>
      <td>22.732341</td>
      <td>23.266348</td>
      <td>23.052755</td>
      <td>23.296861</td>
      <td>23.022268</td>
      <td>21.801575</td>
      <td>22.213581</td>
      <td>22.915488</td>
      <td>Ca</td>
    </tr>
    <tr>
      <th>10698</th>
      <td>22.625595</td>
      <td>22.320402</td>
      <td>21.938908</td>
      <td>21.923615</td>
      <td>22.686622</td>
      <td>22.961228</td>
      <td>22.671335</td>
      <td>23.724242</td>
      <td>23.052848</td>
      <td>22.534055</td>
      <td>...</td>
      <td>33.034940</td>
      <td>32.943447</td>
      <td>31.860054</td>
      <td>32.928161</td>
      <td>33.904767</td>
      <td>34.942374</td>
      <td>34.881281</td>
      <td>33.767487</td>
      <td>32.638294</td>
      <td>Ca</td>
    </tr>
    <tr>
      <th>10699</th>
      <td>32.027901</td>
      <td>31.326007</td>
      <td>32.302614</td>
      <td>32.684121</td>
      <td>33.782728</td>
      <td>34.667734</td>
      <td>34.835641</td>
      <td>34.347348</td>
      <td>33.111354</td>
      <td>32.592561</td>
      <td>...</td>
      <td>22.082146</td>
      <td>22.570453</td>
      <td>22.448360</td>
      <td>22.341566</td>
      <td>21.944873</td>
      <td>22.936680</td>
      <td>22.845087</td>
      <td>22.494193</td>
      <td>22.952000</td>
      <td>Ca</td>
    </tr>
  </tbody>
</table>
<p>10700 rows × 5001 columns</p>
</div>




```python
draw_plot(df_calcium_clf.loc[0][0:5000],
          'Activity of an ion channel not activated by calcium')
```


    
![png](output_3_0.png)
    



```python
draw_plot(df_calcium_clf.loc[df_calcium_clf.shape[0]-1][0:5000], 
          'Activity of an ion channel activated by calcium')
```


    
![png](output_4_0.png)
    



```python

```
