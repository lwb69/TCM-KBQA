# TCM-KBQA
## 基于中医药领域知识图谱的智能问答系统<br>
![aa](https://github.com/lwb69/TCM-KBQA/blob/master/问答流程.png) 
### 项目运行
python QA_system/ans_bot.py
### 项目介绍
中医药知识库来自于互联网爬取和人工从中医药书籍中整理<br>
问答系统中所用到的预训练语言模型为rbt3<br>
[rbt3](https://huggingface.co/hfl/rbt3)
### 项目主要文件
#### ans_bot.py
启动问答系统
#### mention_extrator.py
mention识别
#### entitylink.py
实体链接
#### path_extrator.py
路径召回
#### KG.py
与知识库交互<br>
<br>
部分代码参考 [CCKS2019-CKBQA]([https://huggingface.co/hfl/rbt3](https://github.com/ThisIsSoMe/CCKS2019-CKBQA))
