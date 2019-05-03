import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ages = pd.DataFrame({"age": [18,22,25,27,7,21,23,37,30,61,45,41,9,18,80,100]})
ages['age'].hist()
#等寬劃分
ages['equal_width_age']=pd.cut(ages['age'],4)
ages['equal_width_age'].value_counts().plot.bar()
#等頻劃分(每個區間樣本數量相同)
ages['equal_freq_age']=pd.qcut(ages['age'],4)
ages['equal_freq_age'].value_counts().plot.bar()

#作業
cut_rule=[0,10,20,30,50,100] #不包含0(>0)，包含100(<=100)
ages['customized_age_grp']=pd.cut(ages['age'],bins=cut_rule)
