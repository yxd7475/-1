# -*- coding: utf-8 -*-
"""
测试模块
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_generator import DataGenerator
from src.utils.data_processor import DataProcessor
from src.models.kmeans_model import FarmerClusterAnalyzer, IndustryClusterAnalyzer
from src.analysis.fusion_index import FusionIndexCalculator
from src.analysis.decision_support import DecisionSupportEngine


class TestDataGenerator(unittest.TestCase):
    """数据生成器测试"""

    def setUp(self):
        self.generator = DataGenerator(seed=42)

    def test_generate_farmer_data(self):
        """测试农户数据生成"""
        df = self.generator.generate_farmer_data(100)
        self.assertEqual(len(df), 100)
        self.assertIn('farmer_id', df.columns)
        self.assertIn('fusion_index', df.columns)

    def test_generate_village_data(self):
        """测试村庄数据生成"""
        df = self.generator.generate_village_data(20)
        self.assertEqual(len(df), 20)
        self.assertIn('village_id', df.columns)

    def test_generate_all_data(self):
        """测试全部数据生成"""
        data = self.generator.generate_all_data(100, 10)
        self.assertIn('farmers', data)
        self.assertIn('villages', data)
        self.assertIn('industries', data)


class TestKMeansAnalyzer(unittest.TestCase):
    """KMeans聚类分析测试"""

    def setUp(self):
        self.generator = DataGenerator(seed=42)
        self.farmer_df = self.generator.generate_farmer_data(100)

    def test_farmer_clustering(self):
        """测试农户聚类"""
        analyzer = FarmerClusterAnalyzer(n_clusters=4)
        result = analyzer.cluster_farmers(self.farmer_df)

        self.assertIn('cluster', result.columns)
        self.assertIn('cluster_name', result.columns)
        self.assertEqual(len(result), 100)

    def test_cluster_names(self):
        """测试聚类命名"""
        analyzer = FarmerClusterAnalyzer(n_clusters=4)
        result = analyzer.cluster_farmers(self.farmer_df)

        unique_names = result['cluster_name'].unique()
        self.assertTrue(len(unique_names) > 0)


class TestFusionIndexCalculator(unittest.TestCase):
    """融合指数计算测试"""

    def setUp(self):
        self.calculator = FusionIndexCalculator()
        self.generator = DataGenerator(seed=42)

    def test_base_fusion_index(self):
        """测试基础融合指数"""
        result = self.calculator.calculate_base_fusion_index(dist_town=5, dist_road=2)

        self.assertIn('f_production', result)
        self.assertIn('f_supply', result)
        self.assertIn('f_market', result)

        for value in result.values():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)

    def test_organization_delta(self):
        """测试组织增量"""
        for state in [1, 2, 3, 4]:
            delta = self.calculator.calculate_organization_delta(state)
            self.assertIn('production', delta)

    def test_farmer_fusion_calculation(self):
        """测试农户融合指数计算"""
        df = self.generator.generate_farmer_data(50)
        result = self.calculator.calculate_farmer_fusion(df)

        self.assertIn('fusion_index', result.columns)
        self.assertTrue((result['fusion_index'] >= 0).all())
        self.assertTrue((result['fusion_index'] <= 1).all())


class TestDecisionSupportEngine(unittest.TestCase):
    """决策支持引擎测试"""

    def setUp(self):
        self.engine = DecisionSupportEngine()
        self.generator = DataGenerator(seed=42)
        self.industry_df = self.generator.generate_industry_data(50)

    def test_industry_planning(self):
        """测试产业规划"""
        planning = self.engine.generate_industry_planning(self.industry_df)

        self.assertIn('diagnosis', planning)
        self.assertIn('priority_industries', planning)
        self.assertIn('policy_recommendations', planning)

    def test_risk_assessment(self):
        """测试风险评估"""
        risks = self.engine.generate_risk_assessment(self.industry_df)

        self.assertIn('overall', risks)
        self.assertIn('market_risk', risks)
        self.assertIn('level', risks['overall'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
