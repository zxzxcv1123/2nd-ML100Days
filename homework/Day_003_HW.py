import pandas as pd
import numpy as np
#作業3-1:
pop=np.random.randint(10,100,10)
#創造資料
country=['a','a','b','c','a','d','b','c','a','c']
column=['country','pop']
dic={'country':country,
    'pop':pop}
df=pd.DataFrame(dic)
print(df)
#查看哪個國家人口最多
total=df.groupby('country')['pop'].sum()
total.sort_values(ascending=False)

#作業3-2:讀取txt檔並轉換成dataframe 並讀取前5張
import requests
import pandas as pd
response=requests.get('https://raw.githubusercontent.com/vashineyu/slides_and_others/master/tutorial/examples/imagenet_urls_examples.txt')
data=response.text
data.split('\n')
data=data=data.split('\n')
list_=[]
for i in data:
    list_.append(i.split('\t'))
df=pd.DataFrame(list_)

from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
first_link=df[1].iloc[0]
response=requests.get(first_link)
img=Image.open(BytesIO(response.content))
plt.imshow(img)

def img2arr_fromURLs(url_list):
    imglist=[]
    for i in url_list:
        response=requests.get(i)
        try:
            img=Image.open(BytesIO(response.content))
            imglist.append(img)
        except:
            pass
    return imglist

result=img2arr_fromURLs(df[0:5][1])
print('得到幾張照片',len(result))
for i in result:
    plt.imshow(i)
    plt.show()



