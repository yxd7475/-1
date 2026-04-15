# -*- coding: utf-8 -*-
"""
数据生成模块 - 模拟城乡产业融合数据
"""
import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import MODEL_CONFIG

fake = Faker('zh_CN')

class DataGenerator:
    """城乡产业融合数据生成器"""

    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.config = MODEL_CONFIG

    def generate_farmer_data(self, n_farmers=1000, region_name='示范县'):
        """
        生成农户数据

        Parameters:
        -----------
        n_farmers : int
            农户数量
        region_name : str
            区域名称

        Returns:
        --------
        pd.DataFrame
            农户数据
        """
        farmers = []

        for i in range(n_farmers):
            # 计算到城镇的距离
            dist_town = np.abs(np.random.normal(10, 5))
            dist_town = max(0, min(30, dist_town))

            # 距离城镇越近，参与组织的概率越高
            join_prob = max(0, 0.8 - dist_town * 0.025)

            # 农户状态
            if random.random() < join_prob:
                state = random.choices([2, 3, 4], weights=[0.5, 0.3, 0.2])[0]
            else:
                state = 1

            # 基础属性
            land_area = np.random.lognormal(mean=np.log(10), sigma=0.8)
            land_area = max(1, min(200, land_area))

            land_parcels = random.randint(1, 5)
            labor = max(1, int(np.random.normal(2.5, 1)))
            capital = np.random.lognormal(mean=np.log(50000), sigma=0.6)

            # 风险厌恶系数和学习速率
            risk_aversion = 0.5 + random.random() * 2.5
            learning_rate = 0.2 + random.random() * 0.6

            # 融合指数（基础值受距离影响）
            town_decay = self.config['abm']['town_decay']
            f_base = np.exp(-town_decay * dist_town)

            # 组织参与带来的增量
            f_pro = 0.8 * f_base
            f_sup = 0.7 * f_base
            f_mar = 0.9 * f_base
            f_ser = 0.6 * f_base
            f_val = 0.7 * f_base

            if state in [2, 4]:  # 合作社参与
                f_pro += 0.3
                f_sup += 0.4
                f_mar += 0.2
                f_ser += 0.5
                f_val += 0.6
            if state in [3, 4]:  # 企业参与
                f_pro += 0.4
                f_sup += 0.1
                f_mar += 0.6
                f_ser += 0.0
                f_val += 0.3

            # 限制在[0, 1]范围
            f_pro = min(1.0, max(0.0, f_pro))
            f_sup = min(1.0, max(0.0, f_sup))
            f_mar = min(1.0, max(0.0, f_mar))
            f_ser = min(1.0, max(0.0, f_ser))
            f_val = min(1.0, max(0.0, f_val))

            # 收入（受融合指数影响）
            base_income = 20000 + np.random.normal(0, 5000)
            fusion_bonus = (f_pro + f_sup + f_mar + f_ser + f_val) / 5 * 30000
            income = base_income + fusion_bonus + np.random.normal(0, 3000)

            # 计算综合融合指数
            fusion_index = (f_pro + f_sup + f_mar + f_ser + f_val) / 5

            farmer = {
                'farmer_id': f'F{str(i+1).zfill(6)}',
                'region': region_name,
                'name': fake.name(),
                'dist_town': round(dist_town, 2),
                'dist_road': round(np.abs(np.random.normal(5, 3)), 2),
                'land_area': round(land_area, 2),
                'land_parcels': land_parcels,
                'labor': labor,
                'capital': round(capital, 2),
                'risk_aversion': round(risk_aversion, 3),
                'learning_rate': round(learning_rate, 3),
                'state': state,
                'state_name': {1: '不参与组织', 2: '仅合作社', 3: '仅企业', 4: '复合模式'}[state],
                'tenure': random.randint(0, 10) if state != 1 else 0,
                'f_production': round(f_pro, 3),
                'f_supply': round(f_sup, 3),
                'f_market': round(f_mar, 3),
                'f_service': round(f_ser, 3),
                'f_value': round(f_val, 3),
                'fusion_index': round(fusion_index, 3),
                'annual_income': round(income, 2),
                'latitude': 30.0 + np.random.normal(0, 0.1),
                'longitude': 120.0 + np.random.normal(0, 0.1)
            }
            farmers.append(farmer)

        return pd.DataFrame(farmers)

    def generate_village_data(self, n_villages=50, region_name='示范县'):
        """
        生成行政村数据

        Parameters:
        -----------
        n_villages : int
            村庄数量
        region_name : str
            区域名称

        Returns:
        --------
        pd.DataFrame
            村庄数据
        """
        villages = []

        for i in range(n_villages):
            dist_town = np.abs(np.random.normal(8, 4))
            dist_town = max(0, min(25, dist_town))

            population = int(np.random.lognormal(mean=np.log(500), sigma=0.5))
            households = int(population / 3.5)

            # 产业数据
            crop_area = np.random.uniform(500, 5000)
            orchard_area = np.random.uniform(0, 1000)
            livestock = np.random.uniform(0, 500)

            # 企业数量
            coop_count = max(0, int(np.random.poisson(2) * (1.5 - dist_town/25)))
            firm_count = max(0, int(np.random.poisson(1) * (1.2 - dist_town/25)))
            processing_count = max(0, int(np.random.poisson(0.5) * (1 - dist_town/25)))

            # 融合指数
            fusion_index = np.random.beta(2, 2) * (1.5 - dist_town/30)

            village = {
                'village_id': f'V{str(i+1).zfill(4)}',
                'village_name': fake.street_name() + '村',
                'region': region_name,
                'dist_town': round(dist_town, 2),
                'population': population,
                'households': households,
                'crop_area': round(crop_area, 2),
                'orchard_area': round(orchard_area, 2),
                'livestock': round(livestock, 2),
                'coop_count': coop_count,
                'firm_count': firm_count,
                'processing_count': processing_count,
                'fusion_index': round(min(1.0, fusion_index), 3),
                'per_capita_income': round(np.random.uniform(15000, 40000), 2),
                'latitude': 30.0 + np.random.normal(0, 0.15),
                'longitude': 120.0 + np.random.normal(0, 0.15)
            }
            villages.append(village)

        return pd.DataFrame(villages)

    def generate_industry_data(self, n_records=200, region_name='示范县'):
        """
        生成产业数据

        Parameters:
        -----------
        n_records : int
            记录数量
        region_name : str
            区域名称

        Returns:
        --------
        pd.DataFrame
            产业数据
        """
        industries = []
        industry_types = ['粮食种植', '蔬菜种植', '水果种植', '畜牧养殖', '水产养殖',
                         '农产品加工', '食品制造', '乡村旅游', '农产品电商', '农业服务']

        for i in range(n_records):
            industry_type = random.choice(industry_types)

            # 根据产业类型确定基本属性
            if industry_type in ['粮食种植', '蔬菜种植', '水果种植']:
                primary_ratio = np.random.uniform(0.7, 0.95)
                secondary_ratio = np.random.uniform(0.02, 0.15)
                tertiary_ratio = 1 - primary_ratio - secondary_ratio
            elif industry_type in ['农产品加工', '食品制造']:
                primary_ratio = np.random.uniform(0.1, 0.3)
                secondary_ratio = np.random.uniform(0.5, 0.7)
                tertiary_ratio = 1 - primary_ratio - secondary_ratio
            else:
                primary_ratio = np.random.uniform(0.1, 0.25)
                secondary_ratio = np.random.uniform(0.1, 0.25)
                tertiary_ratio = 1 - primary_ratio - secondary_ratio

            # 融合指数
            fusion_degree = (min(primary_ratio, secondary_ratio) +
                           min(secondary_ratio, tertiary_ratio) +
                           min(primary_ratio, tertiary_ratio)) * 3

            # 收入与利润
            revenue = np.random.lognormal(mean=np.log(500), sigma=1) * 10000
            profit = revenue * np.random.uniform(0.05, 0.25)

            industry = {
                'record_id': f'I{str(i+1).zfill(5)}',
                'region': region_name,
                'industry_type': industry_type,
                'primary_ratio': round(primary_ratio, 3),
                'secondary_ratio': round(secondary_ratio, 3),
                'tertiary_ratio': round(tertiary_ratio, 3),
                'fusion_degree': round(min(1.0, fusion_degree), 3),
                'employment': int(np.random.lognormal(mean=np.log(50), sigma=0.8)),
                'revenue': round(revenue, 2),
                'profit': round(profit, 2),
                'investment': round(revenue * np.random.uniform(0.3, 0.8), 2),
                'year': random.choice([2022, 2023, 2024])
            }
            industries.append(industry)

        return pd.DataFrame(industries)

    def generate_time_series_data(self, n_periods=36, region_name='示范县'):
        """
        生成时间序列数据

        Parameters:
        -----------
        n_periods : int
            时间周期数（月）
        region_name : str
            区域名称

        Returns:
        --------
        pd.DataFrame
            时间序列数据
        """
        dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='M')
        ts_data = []

        base_gdp = 100000  # 基准GDP（万元）
        base_income = 25000  # 基准人均收入

        for i, date in enumerate(dates):
            # 添加趋势和季节性
            trend = i * 0.005
            seasonality = np.sin(i * np.pi / 6) * 0.05

            gdp = base_gdp * (1 + trend + seasonality + np.random.normal(0, 0.02))
            income = base_income * (1 + trend + np.random.normal(0, 0.01))

            record = {
                'date': date,
                'region': region_name,
                'gdp': round(gdp, 2),
                'primary_gdp': round(gdp * np.random.uniform(0.15, 0.25), 2),
                'secondary_gdp': round(gdp * np.random.uniform(0.35, 0.45), 2),
                'tertiary_gdp': round(gdp * np.random.uniform(0.35, 0.45), 2),
                'population': int(np.random.normal(50000, 1000)),
                'urbanization_rate': round(0.4 + i * 0.002 + np.random.normal(0, 0.01), 3),
                'per_capita_income': round(income, 2),
                'rural_income': round(income * 0.6, 2),
                'urban_income': round(income * 1.2, 2),
                'crop_price_index': round(100 * (1 + np.random.normal(0.02, 0.05)), 2),
                'land_transfer_area': round(np.random.uniform(5000, 15000), 2),
                'new_farmers': int(np.random.poisson(50)),
                'tourism_revenue': round(np.random.lognormal(mean=np.log(1000), sigma=0.3), 2)
            }
            ts_data.append(record)

        return pd.DataFrame(ts_data)

    def generate_policy_data(self, n_policies=20):
        """
        生成政策数据

        Parameters:
        -----------
        n_policies : int
            政策数量

        Returns:
        --------
        pd.DataFrame
            政策数据
        """
        policies = []
        policy_types = ['土地流转', '产业扶持', '金融支持', '人才引进', '基础设施建设',
                       '科技推广', '品牌建设', '生态保护', '市场开拓', '社会保障']

        for i in range(n_policies):
            policy_type = random.choice(policy_types)
            start_date = fake.date_between(start_date='-3y', end_date='today')

            policy = {
                'policy_id': f'P{str(i+1).zfill(4)}',
                'policy_name': f'{policy_type}政策_{fake.word()}',
                'policy_type': policy_type,
                'issue_date': start_date,
                'effective_period': random.choice(['1年', '3年', '5年', '长期']),
                'budget': round(np.random.uniform(100, 5000), 2),
                'coverage': random.choice(['全县', '试点乡镇', '特定产业']),
                'target_group': random.choice(['农户', '合作社', '企业', '全体']),
                'implementation_rate': round(np.random.uniform(0.3, 1.0), 3),
                'satisfaction': round(np.random.uniform(60, 95), 1)
            }
            policies.append(policy)

        return pd.DataFrame(policies)

    def generate_all_data(self, n_farmers=1000, n_villages=50, region_name='示范县'):
        """
        生成所有模拟数据

        Returns:
        --------
        dict
            包含所有数据的字典
        """
        print(f"正在生成{region_name}的模拟数据...")

        data = {
            'farmers': self.generate_farmer_data(n_farmers, region_name),
            'villages': self.generate_village_data(n_villages, region_name),
            'industries': self.generate_industry_data(200, region_name),
            'time_series': self.generate_time_series_data(36, region_name),
            'policies': self.generate_policy_data(20)
        }

        print(f"数据生成完成:")
        print(f"  - 农户数据: {len(data['farmers'])} 条")
        print(f"  - 村庄数据: {len(data['villages'])} 条")
        print(f"  - 产业数据: {len(data['industries'])} 条")
        print(f"  - 时间序列: {len(data['time_series'])} 条")
        print(f"  - 政策数据: {len(data['policies'])} 条")

        return data

    def save_data(self, data, output_dir):
        """
        保存数据到文件

        Parameters:
        -----------
        data : dict
            数据字典
        output_dir : str
            输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        for name, df in data.items():
            if isinstance(df, pd.DataFrame):
                # 保存Excel
                excel_path = os.path.join(output_dir, f'{name}.xlsx')
                df.to_excel(excel_path, index=False, engine='openpyxl')
                print(f"已保存: {excel_path}")

                # 保存CSV
                csv_path = os.path.join(output_dir, f'{name}.csv')
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"已保存: {csv_path}")


if __name__ == '__main__':
    # 测试数据生成
    generator = DataGenerator()
    data = generator.generate_all_data()
    generator.save_data(data, 'data/raw')
