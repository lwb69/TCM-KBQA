# TCM-KBQA
## 基于中医药领域知识图谱的智能问答系统<br>
##  Intelligent Question Answering System Based on knowledge graph of traditional Chinese Medicine<br>
![aa](https://github.com/lwb69/TCM-KBQA/blob/master/问答流程.png) 
### 项目运行 Start project
python QA_system/ans_bot.py
### 项目介绍 Project introduction
中医药知识库来自于互联网爬取和人工从中医药书籍中整理<br>
The knowledge base of traditional Chinese medicine is partly from the Internet and partly from the books of traditional Chinese medicine<br>
问答系统中所用到的预训练语言模型为rbt3<br>
The pre training language model used in the question answering system is rbt3<br>
[rbt3](https://huggingface.co/hfl/rbt3)
### 项目主要文件 Main documents of the project
#### ans_bot.py
启动问答系统 Start the question answering system
#### mention_extrator.py
mention识别 Menton recognition
#### entitylink.py
实体链接 Entity links
#### path_extrator.py
路径召回 Path recall
#### KG.py
与知识库交互 Interact with knowledge base<br>
<br>
部分代码参考 [CCKS2019-CKBQA](https://github.com/ThisIsSoMe/CCKS2019-CKBQA)
