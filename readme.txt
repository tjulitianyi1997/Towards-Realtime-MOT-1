1.数据集的改动主要是data文件夹下的文件列表，可能出现多了/data之类的问题，或者是数据集名字不对，基本是直接改配置文件，就是CUHK-SYSU是重命名了数据集的文件夹名(mv CUHK-SYSU CUHKSYSU)
2. batchsize=8 一般是适合两卡的，1088x608
