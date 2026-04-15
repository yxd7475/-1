# 城乡产业融合智能决策系统

## 项目简介

本系统是基于KMeans聚类算法的城乡产业融合智能决策系统，面向政府部门、农业企业、乡村经营主体、金融机构提供全流程智能决策支持。

### 核心功能

1. **数据管理**: 多源数据采集、清洗、标准化处理
2. **KMeans聚类分析**: 农户分类、产业聚类、区域竞争力分析
3. **融合指数计算**: 五维度融合指数（生产、供应、市场、服务、价值）
4. **智能决策支持**: 产业规划、项目选址、风险评估、投资分析
5. **ABM模拟**: 基于主体的农户行为演化模拟
6. **可视化报告**: 交互式Web界面与报告生成

## 项目结构

```
城乡融合智能决策系统/
├── app.py                    # Streamlit主应用
├── api.py                    # FastAPI后端服务
├── requirements.txt          # 依赖包列表
├── run.bat                   # Windows启动脚本
├── run.sh                    # Linux/Mac启动脚本
├── config/
│   └── config.py            # 配置文件
├── src/
│   ├── models/
│   │   ├── kmeans_model.py  # KMeans聚类模型
│   │   └── abm_model.py     # ABM模拟模型
│   ├── analysis/
│   │   ├── fusion_index.py  # 融合指数计算
│   │   └── decision_support.py # 决策支持引擎
│   └── utils/
│       ├── data_generator.py # 数据生成器
│       └── data_processor.py # 数据处理器
├── data/
│   ├── raw/                 # 原始数据
│   ├── processed/           # 处理后数据
│   └── output/              # 输出结果
├── static/                  # 静态资源
├── templates/               # 模板文件
└── tests/                   # 测试文件
```

## 安装说明

### 环境要求

- Python 3.8+
- pip 包管理器

### 安装步骤

1. **克隆项目**
```bash
git clone [项目地址]
cd 城乡融合智能决策系统
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **启动系统**

**方式一：使用启动脚本**
```bash
# Windows
run.bat

# Linux/Mac
chmod +x run.sh
./run.sh
```

**方式二：命令行启动**
```bash
# 启动Web界面
streamlit run app.py

# 启动API服务
python api.py
```

4. **访问系统**
- Web界面: http://localhost:8501
- API文档: http://localhost:8000/docs

## 功能模块说明

### 1. 数据概览

- 查看农户、村庄、产业、时间序列数据
- 数据统计与分布展示
- 数据下载功能

### 2. KMeans聚类分析

#### 农户聚类
- 基于土地、资本、劳动力、融合指数等特征
- 自动识别农户类型：规模融合型、融合发展型、初级融合型、传统型
- 生成转型建议

#### 产业聚类
- 产业类型聚类分析
- 优势产业识别

#### 区域竞争力分析
- 村庄竞争力评估
- 区域排名

### 3. 融合指数分析

五维度融合指数计算：
- **生产融合**: 技术应用、规模生产
- **供应融合**: 供应链整合、合作社参与
- **市场融合**: 品牌建设、渠道拓展
- **服务融合**: 社会化服务利用
- **价值融合**: 产业链延伸、附加值提升

### 4. 智能决策支持

#### 产业规划
- 产业诊断
- 优先发展产业识别
- 政策建议生成

#### 项目选址
- 多维度评分
- 最优选址推荐

#### 风险评估
- 市场、生产、政策、金融风险
- 风险等级划分
- 缓解措施建议

#### 投资分析
- ROI、NPV、IRR计算
- 可行性评估
- 敏感性分析

### 5. ABM模拟

基于主体的建模模拟：
- 农户行为演化
- 状态转移模拟
- 空间分布可视化

### 6. 综合报告

一键生成完整分析报告

## API接口

### 数据生成
```
POST /api/data/generate?n_farmers=1000&n_villages=50
```

### 农户聚类
```
POST /api/clustering/farmers
Body: { "data": [...], "n_clusters": 4 }
```

### 融合指数计算
```
POST /api/fusion/calculate
Body: { "data": [...] }
```

### 产业规划
```
POST /api/decision/planning
```

### 项目选址
```
POST /api/decision/site-selection
```

### 风险评估
```
POST /api/decision/risk-assessment
```

### 投资分析
```
POST /api/decision/investment
```

完整API文档: http://localhost:8000/docs

## 模型参数

### KMeans参数
- n_clusters: 聚类数量 (默认4)
- max_iter: 最大迭代次数 (默认300)
- n_init: 初始化次数 (默认10)

### 融合指数权重
- production: 0.25
- supply: 0.20
- market: 0.25
- service: 0.15
- value: 0.15

### ABM参数
- town_decay: 0.05 (城镇距离衰减)
- road_decay: 0.08 (道路距离衰减)
- learning_rate: 0.3 (学习速率)

## 扩展开发

### 添加新的聚类特征

编辑 `src/models/kmeans_model.py`:
```python
self.clustering_features = [
    'land_area', 'capital', 'labor',
    'new_feature'  # 添加新特征
]
```

### 修改融合指数权重

编辑 `config/config.py`:
```python
'fusion_weights': {
    'production': 0.25,
    'supply': 0.20,
    # ...
}
```

### 自定义决策规则

编辑 `src/analysis/decision_support.py` 中的相关方法

## 技术栈

- **后端**: Python, FastAPI, Streamlit
- **数据处理**: Pandas, NumPy, Scikit-learn
- **可视化**: Plotly, Matplotlib, Seaborn
- **聚类算法**: KMeans, MiniBatchKMeans
- **建模**: ABM (基于主体的建模)

## 许可证

MIT License

## 联系方式

如有问题或建议，请联系开发团队。

---

**城乡产业融合智能决策系统** - 让数据驱动决策，助力乡村振兴
