# -*- coding: utf-8 -*-
"""
城乡产业融合智能决策系统 - 配置文件
"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

# 模型参数配置
MODEL_CONFIG = {
    # KMeans聚类参数
    'kmeans': {
        'n_clusters': 4,  # 农户分类：不参与、仅合作社、仅企业、复合模式
        'max_iter': 300,
        'n_init': 10,
        'random_state': 42
    },

    # 融合指数权重
    'fusion_weights': {
        'production': 0.25,   # 生产融合
        'supply': 0.20,       # 供应融合
        'market': 0.25,       # 市场融合
        'service': 0.15,      # 服务融合
        'value': 0.15         # 价值融合
    },

    # ABM模型参数
    'abm': {
        'town_decay': 0.05,           # 城镇距离衰减
        'road_decay': 0.08,           # 道路距离衰减
        'learning_rate': 0.3,         # 学习速率
        'risk_aversion_base': 1.5,    # 基础风险厌恶系数
        'land_rent_base': 300,        # 基础土地租金
        'max_ticks': 1000,            # 最大迭代次数
        'convergence_threshold': 0.01 # 收敛阈值
    },

    # 生产函数参数
    'production': {
        'theta_tech': 0.5,      # 技术系数
        'alpha': 0.3,           # 劳动弹性
        'beta': 0.3,            # 资本弹性
        'gamma': 0.4,           # 土地弹性
        'lambda_equip': 0.2,    # 装备系数
        'delta_frag': 0.1       # 碎片化惩罚
    },

    # 价格函数参数
    'price': {
        'base_price': 2.0,      # 基础价格
        'eta_brand': 0.2,       # 品牌溢价
        'phi_value': 0.3,       # 价值系数
        'final_price': 10.0,    # 终端价格
        'proc_cost': 3.0,       # 加工成本
        'floor_price': 1.8      # 保底价格
    },

    # 成本函数参数
    'cost': {
        'c_input0': 500,        # 基础投入成本
        'q0': 500,              # 基准产量
        'rho_bulk': 0.2,        # 规模经济
        'phi_scale': 0.8,       # 规模系数
        'tau0': 0.5,            # 基础交易成本
        'mu': 1.0,              # 服务系数
        'sigma0': 1000,         # 风险成本基数
        'nu': 1.0,              # 风险衰减
        'psi': 0.05,            # 规模风险
        'gamma0': 50,           # 组织成本基数
        'gamma1': 50,           # 组织成本增量
        'c_switch': 200,        # 状态转换成本
        'c_member0': 300,       # 会员费
        'c_contract0': 200,     # 合约成本
        'c_coor0': 100          # 协调成本
    }
}

# 农户状态定义
FARMER_STATES = {
    1: '不参与组织',
    2: '仅合作社',
    3: '仅企业',
    4: '复合模式'
}

# 产业类型定义
INDUSTRY_TYPES = {
    'primary': '第一产业(农业)',
    'secondary': '第二产业(加工业)',
    'tertiary': '第三产业(服务业)',
    'fusion': '产业融合'
}

# 风险预警等级
RISK_LEVELS = {
    'red': {'threshold': 0.7, 'desc': '高风险-需立即干预'},
    'yellow': {'threshold': 0.4, 'desc': '中风险-需关注监控'},
    'blue': {'threshold': 0.0, 'desc': '低风险-正常运行'}
}

# 决策场景
DECISION_SCENARIOS = {
    'government': {
        'name': '政府端',
        'functions': ['产业规划', '园区布局', '招商方向', '政策制定', '资金分配']
    },
    'enterprise': {
        'name': '企业端',
        'functions': ['项目选址', '产品定位', '渠道拓展', '供应链优化']
    },
    'finance': {
        'name': '金融端',
        'functions': ['涉农信贷风控', '保险定价', '投资标的筛选']
    },
    'village': {
        'name': '乡村端',
        'functions': ['特色产业选择', '文旅项目设计', '电商运营策略']
    }
}

# 空间数据配置
SPATIAL_CONFIG = {
    'crs': 'EPSG:4326',  # WGS84坐标系
    'grid_size': 50,      # 网格大小(km)
    'town_center': {'lat': 30.0, 'lng': 120.0}  # 默认城镇中心
}

# API配置
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': True
}
