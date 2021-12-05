## Kaggle Tricks

### 无法打印日志
在Notebook中添加：
```python
import logging,sys
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, format='%(filename)s:%(lineno)s %(levelname)s:%(message)s')
```

### 无法使用异步代码
在Notebook中添加：
```python
import nest_asyncio
nest_asyncio.apply()
```

### kaggle kernel安装pip包
```shell
!pip install numpy
```
需要在命令前面加上`!`

### 环境变量不起作用
以下命令控制环境变量起作用：
```shell
os.environ['AGG_IP']="159.138.44.120"
```

以下导入环境变量命令不起作用：
```shell
!export AGG_IP="127.0.0.1"
```


### 添加自定义py文件到kaggle notebook
* 将py文件打包成Kaggle Datasets发布，导入到Kaggle Notebook中，即可添加自定义文件。
* 其他添加方式，在Kaggle Notebook关闭后都不会记录到Notebook中。



