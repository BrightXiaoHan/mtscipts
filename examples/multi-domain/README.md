# 多垂域混合模型训练

## 数据准备
将所有数据处理成以下格式，并存放在某个目录下面
- `online`目录中为公司二级垂域分类语料
- opensource目录中主要为开源数据，或者其他来源（如爬虫）搜集的通用数据
- `test`目录中主要为开源的测试集（主要来源为sacrebleu）
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

# 其他测试集
```
清洗和准备语料
```
# 指定环境变量为上面数据集的存放目录
export DATA_DIR=/path/to/dataset

# 清洗公司内垂域数据
bash run_filter_online.sh

# 清洗其他开源数据
bash run_filter_opensource.sh
```

训练分词模型以及训练字词
