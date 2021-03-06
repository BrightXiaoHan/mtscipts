# 多垂域混合模型训练

## 数据准备
将所有数据处理成以下格式，并存放在某个目录下面
- `online`目录中为公司二级垂域分类语料
- opensource目录中主要为开源数据，或者其他来源（如爬虫）搜集的通用数据
- `test`目录中主要为开源的测试集（主要来源为sacrebleu）

Note: 所有二级垂域的需要在`DOMAIN_LIST.txt`中配置映射关系，如新增垂域也需要在此文件中进行配置

目前没找到原因`期刊/论文`数据暂时没有加入
```
thesis QKLW
```
下面这连个垂域`半导体`和`人工智能`数据较少，暂时没有加入训练
```
semiconductor BDT
ai RZZN
```
此为通用领域，没有划分到垂域的开源数据等都划分到通用领域
```
general GEN
```
其中第一个词要与二级垂域的目录名称相同，第二个词与第一个词使用空格隔开，需指定全大写字母，一般为垂域中文名称的拼音首字母大写。



```
# 垂域数据
├── online
│   ├── agriculture
│   │   └── afaf
│   │       ├── en
│   │       └── zh
│   ├── anime-literature
│   │   ├── anime
│   │   │   ├── en
│   │   │   └── zh
│   │   ├── man-novel
│   │   │   ├── en
│   │   │   └── zh
│   │   └── woman-novel
│   │       ├── en
│   │       └── zh
│   ├── biomedicine
│   │   ├── biological-products
│   │   │   ├── en
│   │   │   └── zh
│   │   ├── chemical-drugs
│   │   │   ├── en
│   │   │   └── zh
│   │   ├── chinese-medicine
│   │   │   ├── en
│   │   │   └── zh
│   │   └── medical-health
│   │       ├── en
│   │       └── zh

# 通用领域数据
├── opensource
│   ├── ai-challenger
│   │   ├── en
│   │   └── zh
│   ├── CCMT
│   │   ├── en
│   │   └── zh
│   ├── news-commentary-v15
│   │   ├── en
│   │   └── zh
│   ├── ted
│   │   ├── en
│   │   └── zh
│   ├── translation2019zh
│   │   ├── en
│   │   └── zh
│   ├── UM-Corpus
│   │   ├── en
│   │   └── zh
│   ├── UNv1.0
│   │   ├── en
│   │   └── zh
│   ├── wikititles-v2
│   │   ├── en
│   │   └── zh
│   └── wyjcrawl
│       ├── en
│       └── zh
```

## 训练模型
需要指定的环境变量
```bash
# 指定原文语言，译文语言
export SRCLANG=en
export TGTLANG=zh
# 指定环境变量为上面数据集的存放目录
export DATA_DIR=/path/to/dataset

# 指定eflomal的路径
export EFLOMAL_PATH=/path/to/eflomal/

# 指定模型训练目录
export TRAIN_DIR=/path/to/train/dir

# 指定terashuf，在不占用内存的情况下乱序文件
# 查看此连接安装 https://github.com/alexandres/terashuf
# 指定安装目录为环境变量
export TERASHUF_PATH=/path/to/terashuf/
export MEMORY=20  # 指定乱序时最大的内存使用
```

清洗和准备语料
```bash
# 清洗公司内垂域数据
bash run_filter_online.sh

# 清洗其他开源数据
bash run_filter_opensource.sh
```

使用CodeSwitch技术对语料进行扩充
```bash
bash apply_codeswitch.sh
```

训练分词模型
```bash
bash tokenize.sh train
```
分词（分词过程中使用Subword Regularzation对语料进行五倍扩充）
```bash
bash tokenize.sh tokenize
```

数据合并、乱序、分片等预处理操作

Note: 可以根据自己设备内存的大小，动态调整每个分片数据的最大数量
```bash
# 每个分片的最大数据量为50000000
bash preprocess.sh 50000000
```
- Note: `$TRAIN_DIR/data-bin-$SRCLANG-$TGTLANG`位置记录分片的数量，作为后面训练时的参数
- TODO: 这里后面需要做的更自动化一点

使用fairseq进行模型训练

Note: `train.sh`脚本测试在4xA100 GPU上训练，如果在其他设备配置上训练，适当调整`--max-tokens`等参数
```bash
bash train.sh preprocess $SHARED_NUM # fairseq-preprocess
bash train.sh train $SHARED_NUM # fairseq-train
bash train.sh eval  # fairseq-generate
```



## 评测与部署
发布模型
```
# 第一个参数是模型输出的目录位置，第二个参数是将最后几个checkpoints进行平均
bash release.sh /path/to/output_dir n_everage
```
