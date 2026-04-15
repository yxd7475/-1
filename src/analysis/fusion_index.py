# -*- coding: utf-8 -*-
"""
城乡融合指数计算模块
根据文档中的融合指数公式，计算生产融合、供应融合、市场融合、服务融合、价值融合五个维度
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import MODEL_CONFIG


class FusionIndexCalculator:
    """城乡融合指数计算器"""

    def __init__(self):
        self.config = MODEL_CONFIG
        self.weights = self.config['fusion_weights']

    def calculate_base_fusion_index(self, dist_town, dist_road, town_decay=0.05, road_decay=0.08):
        """
        计算基础融合指数（受空间位置影响）

        Parameters:
        -----------
        dist_town : float
            到城镇的距离
        dist_road : float
            到主干道的距离
        town_decay : float
            城镇距离衰减系数
        road_decay : float
            道路距离衰减系数

        Returns:
        --------
        dict
            各维度基础融合指数
        """
        # 基于空间衰减函数
        town_factor = np.exp(-town_decay * dist_town)
        road_factor = np.exp(-road_decay * dist_road)

        # 各维度基础值
        f_pro0 = 0.8 * town_factor  # 生产融合
        f_sup0 = 0.7 * road_factor  # 供应融合
        f_mar0 = 0.9 * town_factor  # 市场融合
        f_ser0 = 0.6 * town_factor  # 服务融合
        f_val0 = 0.7 * town_factor  # 价值融合

        return {
            'f_production': f_pro0,
            'f_supply': f_sup0,
            'f_market': f_mar0,
            'f_service': f_ser0,
            'f_value': f_val0
        }

    def calculate_organization_delta(self, state):
        """
        计算组织参与带来的融合指数增量

        Parameters:
        -----------
        state : int
            农户状态 (1=不参与, 2=仅合作社, 3=仅企业, 4=复合模式)

        Returns:
        --------
        dict
            各维度增量
        """
        # 合作社参与增量
        coop_delta = {
            'production': 0.3,
            'supply': 0.4,
            'market': 0.2,
            'service': 0.5,
            'value': 0.6
        }

        # 企业参与增量
        firm_delta = {
            'production': 0.4,
            'supply': 0.1,
            'market': 0.6,
            'service': 0.0,
            'value': 0.3
        }

        delta = {
            'production': 0,
            'supply': 0,
            'market': 0,
            'service': 0,
            'value': 0
        }

        if state in [2, 4]:  # 合作社参与
            for key in delta:
                delta[key] += coop_delta[key]

        if state in [3, 4]:  # 企业参与
            for key in delta:
                delta[key] += firm_delta[key]

        return delta

    def calculate_actual_fusion_index(self, base_index, delta):
        """
        计算实际融合指数

        Parameters:
        -----------
        base_index : dict
            基础融合指数
        delta : dict
            组织增量

        Returns:
        --------
        dict
            实际融合指数
        """
        actual = {}
        for key in base_index:
            actual[key] = min(1.0, base_index[key] + delta[key])

        return actual

    def calculate_comprehensive_index(self, fusion_indices, weights=None):
        """
        计算综合融合指数

        Parameters:
        -----------
        fusion_indices : dict
            各维度融合指数
        weights : dict
            权重字典

        Returns:
        --------
        float
            综合融合指数
        """
        if weights is None:
            weights = self.weights

        total = 0
        weight_sum = 0

        for key, value in fusion_indices.items():
            weight = weights.get(key.replace('f_', ''), 0.1)
            total += value * weight
            weight_sum += weight

        if weight_sum > 0:
            return total / weight_sum
        return 0

    def calculate_farmer_fusion(self, df):
        """
        批量计算农户融合指数

        Parameters:
        -----------
        df : pd.DataFrame
            农户数据

        Returns:
        --------
        pd.DataFrame
            带融合指数的数据
        """
        result = df.copy()

        # 初始化融合指数列
        result['f_production'] = 0.0
        result['f_supply'] = 0.0
        result['f_market'] = 0.0
        result['f_service'] = 0.0
        result['f_value'] = 0.0

        for idx, row in df.iterrows():
            # 基础融合指数
            base = self.calculate_base_fusion_index(
                row.get('dist_town', 10),
                row.get('dist_road', 5),
                self.config['abm']['town_decay'],
                self.config['abm']['road_decay']
            )

            # 组织增量
            delta = self.calculate_organization_delta(row.get('state', 1))

            # 实际融合指数
            actual = self.calculate_actual_fusion_index(base, delta)

            # 更新数据
            result.loc[idx, 'f_production'] = actual['production']
            result.loc[idx, 'f_supply'] = actual['supply']
            result.loc[idx, 'f_market'] = actual['market']
            result.loc[idx, 'f_service'] = actual['service']
            result.loc[idx, 'f_value'] = actual['value']

        # 计算综合融合指数
        result['fusion_index'] = result.apply(
            lambda row: self.calculate_comprehensive_index({
                'production': row['f_production'],
                'supply': row['f_supply'],
                'market': row['f_market'],
                'service': row['f_service'],
                'value': row['f_value']
            }),
            axis=1
        )

        return result

    def calculate_village_fusion(self, df):
        """
        计算村庄融合指数

        Parameters:
        -----------
        df : pd.DataFrame
            村庄数据

        Returns:
        --------
        pd.DataFrame
            带融合指数的数据
        """
        result = df.copy()

        # 标准化函数
        def normalize(series, positive=True):
            min_val, max_val = series.min(), series.max()
            if max_val - min_val == 0:
                return np.ones(len(series)) * 0.5
            if positive:
                return (series - min_val) / (max_val - min_val)
            else:
                return (max_val - series) / (max_val - min_val)

        # 计算各维度指标

        # 1. 生产融合：基于耕种面积、机械化程度
        if 'crop_area' in result.columns:
            result['f_production'] = normalize(result['crop_area'])
        else:
            result['f_production'] = 0.5

        # 2. 供应融合：基于合作社数量、加工企业
        if 'coop_count' in result.columns and 'processing_count' in result.columns:
            result['f_supply'] = normalize(result['coop_count'] + result['processing_count'] * 2)
        else:
            result['f_supply'] = 0.5

        # 3. 市场融合：基于企业数量、电商活跃度
        if 'firm_count' in result.columns:
            result['f_market'] = normalize(result['firm_count'])
        else:
            result['f_market'] = 0.5

        # 4. 服务融合：基于人口密度、基础设施
        if 'population' in result.columns:
            result['f_service'] = normalize(result['population'] / 100)
        else:
            result['f_service'] = 0.5

        # 5. 价值融合：基于人均收入
        if 'per_capita_income' in result.columns:
            result['f_value'] = normalize(result['per_capita_income'])
        else:
            result['f_value'] = 0.5

        # 综合融合指数
        result['fusion_index'] = (
            result['f_production'] * self.weights['production'] +
            result['f_supply'] * self.weights['supply'] +
            result['f_market'] * self.weights['market'] +
            result['f_service'] * self.weights['service'] +
            result['f_value'] * self.weights['value']
        )

        return result

    def analyze_fusion_gap(self, df):
        """
        分析融合差距

        Parameters:
        -----------
        df : pd.DataFrame
            带融合指数的数据

        Returns:
        --------
        dict
            融合差距分析
        """
        fusion_cols = ['f_production', 'f_supply', 'f_market', 'f_service', 'f_value']
        existing_cols = [col for col in fusion_cols if col in df.columns]

        analysis = {
            'overall_avg': df[existing_cols].mean().to_dict(),
            'overall_std': df[existing_cols].std().to_dict(),
            'gaps': {}
        }

        # 计算差距（与最优值的差距）
        for col in existing_cols:
            max_val = df[col].max()
            gap = max_val - df[col].mean()
            analysis['gaps'][col] = {
                'max': max_val,
                'avg': df[col].mean(),
                'gap': gap,
                'gap_ratio': gap / max_val if max_val > 0 else 0
            }

        # 识别短板维度
        sorted_gaps = sorted(
            analysis['gaps'].items(),
            key=lambda x: x[1]['gap_ratio'],
            reverse=True
        )
        analysis['shortcomings'] = [item[0] for item in sorted_gaps[:2]]

        return analysis

    def generate_improvement_suggestions(self, analysis_result):
        """
        生成改进建议

        Parameters:
        -----------
        analysis_result : dict
            融合差距分析结果

        Returns:
        --------
        list
            改进建议列表
        """
        suggestions = []

        suggestion_map = {
            'f_production': {
                'title': '提升生产融合度',
                'measures': [
                    '引入现代农业技术，提高机械化水平',
                    '发展规模化种植，提升土地产出效率',
                    '推广标准化生产流程，保障产品质量'
                ],
                'policies': ['农机购置补贴', '规模经营奖励', '技术培训支持']
            },
            'f_supply': {
                'title': '提升供应融合度',
                'measures': [
                    '发展农民专业合作社，提高组织化程度',
                    '建设冷链物流设施，降低流通损耗',
                    '对接农资供应商，降低采购成本'
                ],
                'policies': ['合作社扶持政策', '冷链建设补贴', '订单农业奖励']
            },
            'f_market': {
                'title': '提升市场融合度',
                'measures': [
                    '发展农产品电商，拓展销售渠道',
                    '建设品牌体系，提升产品溢价能力',
                    '参与农产品展销会，对接城市消费市场'
                ],
                'policies': ['电商扶持政策', '品牌创建奖励', '市场开拓补贴']
            },
            'f_service': {
                'title': '提升服务融合度',
                'measures': [
                    '发展农业生产社会化服务',
                    '建设农业科技服务站',
                    '完善农村金融服务体系'
                ],
                'policies': ['社会化服务补贴', '科技服务体系建设', '普惠金融支持']
            },
            'f_value': {
                'title': '提升价值融合度',
                'measures': [
                    '发展农产品精深加工',
                    '延长产业链条，提高附加值',
                    '发展休闲农业和乡村旅游'
                ],
                'policies': ['加工企业扶持', '产业融合示范项目', '乡村旅游开发支持']
            }
        }

        for shortcoming in analysis_result.get('shortcomings', []):
            if shortcoming in suggestion_map:
                suggestions.append(suggestion_map[shortcoming])

        return suggestions


class FusionIndexVisualizer:
    """融合指数可视化"""

    def __init__(self):
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_radar_chart(self, fusion_data, title='融合指数雷达图'):
        """
        绘制雷达图

        Parameters:
        -----------
        fusion_data : dict
            融合指数数据
        title : str
            图表标题

        Returns:
        --------
        matplotlib.figure.Figure
            图表对象
        """
        import matplotlib.pyplot as plt

        categories = ['生产融合', '供应融合', '市场融合', '服务融合', '价值融合']
        keys = ['production', 'supply', 'market', 'service', 'value']

        values = [fusion_data.get(f'f_{k}', 0.5) for k in keys]
        values += values[:1]  # 闭合

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title(title, size=15, pad=20)

        return fig

    def plot_dimension_comparison(self, df, group_col='state'):
        """
        绘制维度对比图

        Parameters:
        -----------
        df : pd.DataFrame
            数据
        group_col : str
            分组列

        Returns:
        --------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        fusion_cols = ['f_production', 'f_supply', 'f_market', 'f_service', 'f_value']
        existing_cols = [col for col in fusion_cols if col in df.columns]

        grouped = df.groupby(group_col)[existing_cols].mean()

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(existing_cols))
        width = 0.8 / len(grouped)

        for i, (name, row) in enumerate(grouped.iterrows()):
            ax.bar(x + i * width, row.values, width, label=str(name))

        ax.set_xlabel('融合维度')
        ax.set_ylabel('融合指数')
        ax.set_title('各群体融合维度对比')
        ax.set_xticks(x + width * (len(grouped) - 1) / 2)
        ax.set_xticklabels([col.replace('f_', '') for col in existing_cols])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        return fig

    def plot_distribution(self, df, col='fusion_index'):
        """
        绘制融合指数分布图

        Parameters:
        -----------
        df : pd.DataFrame
            数据
        col : str
            列名

        Returns:
        --------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 直方图
        axes[0].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(df[col].mean(), color='red', linestyle='--', label=f'均值: {df[col].mean():.3f}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('频数')
        axes[0].set_title(f'{col}分布')
        axes[0].legend()

        # 箱线图
        axes[1].boxplot(df[col], vert=True)
        axes[1].set_ylabel(col)
        axes[1].set_title(f'{col}箱线图')

        plt.tight_layout()
        return fig


if __name__ == '__main__':
    # 测试
    print("测试融合指数计算模块...")

    calculator = FusionIndexCalculator()

    # 测试单个农户
    base = calculator.calculate_base_fusion_index(dist_town=5, dist_road=2)
    print("\n基础融合指数:", base)

    delta = calculator.calculate_organization_delta(state=4)
    print("组织增量:", delta)

    actual = calculator.calculate_actual_fusion_index(base, delta)
    print("实际融合指数:", actual)

    comprehensive = calculator.calculate_comprehensive_index(actual)
    print("综合融合指数:", comprehensive)

    # 批量测试
    from utils.data_generator import DataGenerator
    generator = DataGenerator()
    farmer_data = generator.generate_farmer_data(100)

    result = calculator.calculate_farmer_fusion(farmer_data)
    print("\n批量计算结果:")
    print(result[['farmer_id', 'state', 'fusion_index']].head(10))

    # 差距分析
    analysis = calculator.analyze_fusion_gap(result)
    print("\n差距分析:", analysis)
