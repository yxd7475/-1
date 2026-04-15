# -*- coding: utf-8 -*-
"""
KMeans聚类分析核心模块
实现农户分类、产业融合度分析、区域竞争力分析等功能
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import MODEL_CONFIG


class KMeansAnalyzer:
    """KMeans聚类分析器"""

    def __init__(self, n_clusters=4, random_state=42):
        """
        初始化

        Parameters:
        -----------
        n_clusters : int
            聚类数量
        random_state : int
            随机种子
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.pca = None
        self.labels = None
        self.cluster_centers = None
        self.feature_names = None

    def find_optimal_clusters(self, X, max_k=10, method='elbow'):
        """
        寻找最优聚类数

        Parameters:
        -----------
        X : np.ndarray
            特征矩阵
        max_k : int
            最大聚类数
        method : str
            'elbow' - 肘部法则
            'silhouette' - 轮廓系数

        Returns:
        --------
        dict
            评估结果
        """
        results = {
            'k_values': list(range(2, max_k + 1)),
            'inertias': [],
            'silhouettes': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)

            results['inertias'].append(kmeans.inertia_)
            results['silhouettes'].append(silhouette_score(X, labels))
            results['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
            results['davies_bouldin'].append(davies_bouldin_score(X, labels))

        # 自动选择最优K
        if method == 'silhouette':
            optimal_k = results['k_values'][np.argmax(results['silhouettes'])]
        elif method == 'calinski':
            optimal_k = results['k_values'][np.argmax(results['calinski_harabasz'])]
        else:  # elbow
            # 使用二阶差分找肘点
            inertias = np.array(results['inertias'])
            diff = np.diff(inertias)
            diff2 = np.diff(diff)
            optimal_k = results['k_values'][np.argmax(np.abs(diff2)) + 2]

        results['optimal_k'] = optimal_k

        return results

    def fit(self, X, feature_names=None):
        """
        训练KMeans模型

        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            特征矩阵
        feature_names : list
            特征名列表

        Returns:
        --------
        self
        """
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values

        self.feature_names = feature_names

        # 创建并训练模型
        self.model = KMeans(
            n_clusters=self.n_clusters,
            max_iter=MODEL_CONFIG['kmeans']['max_iter'],
            n_init=MODEL_CONFIG['kmeans']['n_init'],
            random_state=self.random_state
        )

        self.labels = self.model.fit_predict(X)
        self.cluster_centers = self.model.cluster_centers_

        return self

    def predict(self, X):
        """
        预测新样本的聚类标签

        Parameters:
        -----------
        X : np.ndarray
            特征矩阵

        Returns:
        --------
        np.ndarray
            聚类标签
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")

        return self.model.predict(X)

    def get_cluster_stats(self, df, label_col='cluster'):
        """
        获取各聚类统计信息

        Parameters:
        -----------
        df : pd.DataFrame
            包含聚类标签的数据
        label_col : str
            聚类标签列名

        Returns:
        --------
        pd.DataFrame
            各聚类统计
        """
        stats = []

        for cluster in range(self.n_clusters):
            cluster_data = df[df[label_col] == cluster]

            stat = {
                'cluster': cluster,
                'count': len(cluster_data),
                'proportion': len(cluster_data) / len(df)
            }

            # 计算各特征的均值
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != label_col:
                    stat[f'{col}_mean'] = cluster_data[col].mean()
                    stat[f'{col}_std'] = cluster_data[col].std()

            stats.append(stat)

        return pd.DataFrame(stats)

    def analyze_cluster_characteristics(self, df, label_col='cluster'):
        """
        分析各聚类特征

        Parameters:
        -----------
        df : pd.DataFrame
            数据
        label_col : str
            聚类标签列名

        Returns:
        --------
        dict
            各聚类特征分析结果
        """
        analysis = {}

        for cluster in range(self.n_clusters):
            cluster_data = df[df[label_col] == cluster]

            # 特征重要性排名
            feature_ranking = {}
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                if col != label_col:
                    # 计算该聚类与全局均值的差异
                    cluster_mean = cluster_data[col].mean()
                    global_mean = df[col].mean()
                    global_std = df[col].std()

                    if global_std > 0:
                        z_score = (cluster_mean - global_mean) / global_std
                    else:
                        z_score = 0

                    feature_ranking[col] = {
                        'mean': cluster_mean,
                        'global_mean': global_mean,
                        'z_score': z_score,
                        'relative_diff': (cluster_mean - global_mean) / global_mean if global_mean > 0 else 0
                    }

            # 按z_score排序
            sorted_features = sorted(
                feature_ranking.items(),
                key=lambda x: abs(x[1]['z_score']),
                reverse=True
            )

            analysis[cluster] = {
                'size': len(cluster_data),
                'proportion': len(cluster_data) / len(df),
                'top_features': sorted_features[:5],
                'feature_ranking': dict(sorted_features)
            }

        return analysis

    def get_evaluation_metrics(self, X):
        """
        获取聚类评估指标

        Parameters:
        -----------
        X : np.ndarray
            特征矩阵

        Returns:
        --------
        dict
            评估指标
        """
        if self.labels is None:
            raise ValueError("模型未训练")

        metrics = {
            'inertia': self.model.inertia_,
            'silhouette_score': silhouette_score(X, self.labels),
            'calinski_harabasz_score': calinski_harabasz_score(X, self.labels),
            'davies_bouldin_score': davies_bouldin_score(X, self.labels)
        }

        return metrics

    def reduce_dimensions(self, X, n_components=2):
        """
        PCA降维

        Parameters:
        -----------
        X : np.ndarray
            特征矩阵
        n_components : int
            降维后的维度

        Returns:
        --------
        np.ndarray
            降维后的数据
        """
        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X)

        return X_reduced

    def get_feature_importance(self):
        """
        获取特征重要性（基于聚类中心的方差）

        Returns:
        --------
        dict
            特征重要性
        """
        if self.cluster_centers is None or self.feature_names is None:
            raise ValueError("模型未训练或特征名为空")

        # 计算各特征在聚类中心之间的方差
        variances = np.var(self.cluster_centers, axis=0)

        importance = {}
        for i, name in enumerate(self.feature_names):
            importance[name] = variances[i]

        # 归一化
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        # 排序
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return importance


class FarmerClusterAnalyzer(KMeansAnalyzer):
    """农户聚类分析器 - 专门用于农户分类"""

    def __init__(self, n_clusters=4, random_state=42):
        super().__init__(n_clusters, random_state)

        # 农户聚类特征
        self.clustering_features = [
            'land_area', 'capital', 'labor', 'land_parcels',
            'dist_town', 'dist_road',
            'f_production', 'f_supply', 'f_market', 'f_service', 'f_value',
            'risk_aversion', 'learning_rate'
        ]

    def cluster_farmers(self, df):
        """
        对农户进行聚类

        Parameters:
        -----------
        df : pd.DataFrame
            农户数据

        Returns:
        --------
        pd.DataFrame
            带有聚类标签的数据
        """
        # 选择存在的特征
        existing_features = [f for f in self.clustering_features if f in df.columns]

        # 提取特征矩阵
        X = df[existing_features].values

        # 标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 使用MiniBatchKMeans加速（数据量大时）
        if len(df) > 500:
            self.model = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                batch_size=min(100, len(df) // 10),
                n_init=3
            )
            self.labels = self.model.fit_predict(X_scaled)
            self.cluster_centers = self.model.cluster_centers_
            self.feature_names = existing_features
        else:
            # 小数据集使用标准KMeans
            self.fit(X_scaled, existing_features)

        # 添加聚类标签
        result_df = df.copy()
        result_df['cluster'] = self.labels

        # 为聚类命名
        self._name_clusters(result_df)

        return result_df

    def _name_clusters(self, df):
        """
        根据特征为聚类命名

        Parameters:
        -----------
        df : pd.DataFrame
            带有聚类标签的数据
        """
        # 确保融合指数列存在
        if 'fusion_index' not in df.columns:
            fusion_cols = ['f_production', 'f_supply', 'f_market', 'f_service', 'f_value']
            existing_cols = [c for c in fusion_cols if c in df.columns]
            if existing_cols:
                df['fusion_index'] = df[existing_cols].mean(axis=1)
            else:
                df['fusion_index'] = 0.5

        # 计算各聚类的特征
        cluster_profiles = {}

        for cluster in range(self.n_clusters):
            cluster_data = df[df['cluster'] == cluster]

            profile = {
                'avg_land': cluster_data['land_area'].mean() if 'land_area' in cluster_data else 0,
                'avg_capital': cluster_data['capital'].mean() if 'capital' in cluster_data else 0,
                'avg_fusion': cluster_data['fusion_index'].mean() if 'fusion_index' in cluster_data else 0,
                'avg_dist': cluster_data['dist_town'].mean() if 'dist_town' in cluster_data else 0
            }
            cluster_profiles[cluster] = profile

        # 定义聚类名称
        cluster_names = {}
        for cluster, profile in cluster_profiles.items():
            if profile['avg_fusion'] > 0.6 and profile['avg_land'] > 50:
                cluster_names[cluster] = '规模融合型农户'
            elif profile['avg_fusion'] > 0.6:
                cluster_names[cluster] = '融合发展型农户'
            elif profile['avg_fusion'] > 0.4:
                cluster_names[cluster] = '初级融合型农户'
            elif profile['avg_dist'] > 15:
                cluster_names[cluster] = '偏远传统型农户'
            else:
                cluster_names[cluster] = '近郊传统型农户'

        df['cluster_name'] = df['cluster'].map(cluster_names)
        self.cluster_names = cluster_names

    def get_transition_recommendations(self, df):
        """
        获取农户转型建议

        Parameters:
        -----------
        df : pd.DataFrame
            带聚类标签的数据

        Returns:
        --------
        dict
            各类农户的转型建议
        """
        # 确保融合指数列存在
        if 'fusion_index' not in df.columns:
            fusion_cols = ['f_production', 'f_supply', 'f_market', 'f_service', 'f_value']
            existing_cols = [c for c in fusion_cols if c in df.columns]
            if existing_cols:
                df['fusion_index'] = df[existing_cols].mean(axis=1)

        recommendations = {}

        for cluster in range(self.n_clusters):
            cluster_data = df[df['cluster'] == cluster]
            cluster_name = cluster_data['cluster_name'].iloc[0] if 'cluster_name' in cluster_data else f'聚类{cluster}'

            # 分析特征短板
            fusion_cols = ['f_production', 'f_supply', 'f_market', 'f_service', 'f_value']
            existing_fusion_cols = [c for c in fusion_cols if c in cluster_data.columns]

            if existing_fusion_cols:
                avg_fusion = cluster_data[existing_fusion_cols].mean()
                weak_dims = avg_fusion.nsmallest(2).index.tolist()
            else:
                weak_dims = []

            # 生成建议
            rec = {
                'cluster': cluster,
                'cluster_name': cluster_name,
                'size': len(cluster_data),
                'avg_income': cluster_data['annual_income'].mean() if 'annual_income' in cluster_data else 0,
                'weak_dimensions': weak_dims,
                'recommendations': self._generate_recommendations(weak_dims, cluster_data)
            }

            recommendations[cluster] = rec

        return recommendations

    def _generate_recommendations(self, weak_dims, cluster_data):
        """生成具体建议"""
        rec_map = {
            'f_production': '建议引入现代农业技术，提高生产效率；参与合作社获取规模化生产支持',
            'f_supply': '建议对接优质农资供应商，降低采购成本；发展订单农业',
            'f_market': '建议发展品牌建设，拓展销售渠道；参与电商平台销售',
            'f_service': '建议利用社会化服务资源，如农机服务、植保服务等',
            'f_value': '建议发展农产品初加工，延长产业链，提高附加值'
        }

        recommendations = []
        for dim in weak_dims:
            if dim in rec_map:
                recommendations.append({
                    'dimension': dim,
                    'suggestion': rec_map[dim]
                })

        return recommendations


class IndustryClusterAnalyzer(KMeansAnalyzer):
    """产业聚类分析器"""

    def __init__(self, n_clusters=5, random_state=42):
        super().__init__(n_clusters, random_state)

        self.clustering_features = [
            'primary_ratio', 'secondary_ratio', 'tertiary_ratio',
            'fusion_degree', 'employment', 'revenue', 'profit'
        ]

    def cluster_industries(self, df):
        """
        产业聚类分析

        Parameters:
        -----------
        df : pd.DataFrame
            产业数据

        Returns:
        --------
        pd.DataFrame
            带有聚类标签的数据
        """
        existing_features = [f for f in self.clustering_features if f in df.columns]
        X = df[existing_features].values

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 使用快速聚类
        self.model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=50,
            n_init=3
        )
        self.labels = self.model.fit_predict(X_scaled)
        self.cluster_centers = self.model.cluster_centers_
        self.feature_names = existing_features

        result_df = df.copy()
        result_df['cluster'] = self.labels

        return result_df

    def identify_advantage_industries(self, df, top_n=5):
        """
        识别优势产业

        Parameters:
        -----------
        df : pd.DataFrame
            产业数据
        top_n : int
            返回前N个

        Returns:
        --------
        pd.DataFrame
            优势产业列表
        """
        # 计算综合得分
        df['advantage_score'] = (
            df['fusion_degree'] * 0.3 +
            df['profit'] / df['profit'].max() * 0.3 +
            df['employment'] / df['employment'].max() * 0.2 +
            df['revenue'] / df['revenue'].max() * 0.2
        )

        # 按产业类型聚合
        advantage = df.groupby('industry_type').agg({
            'advantage_score': 'mean',
            'revenue': 'sum',
            'profit': 'sum',
            'employment': 'sum',
            'fusion_degree': 'mean'
        }).sort_values('advantage_score', ascending=False)

        return advantage.head(top_n)


class RegionalCompetitivenessAnalyzer(KMeansAnalyzer):
    """区域竞争力分析器"""

    def __init__(self, n_clusters=3, random_state=42):
        super().__init__(n_clusters, random_state)

        self.competitiveness_features = [
            'fusion_index', 'per_capita_income', 'population',
            'coop_count', 'firm_count', 'processing_count'
        ]

    def analyze_regional_competitiveness(self, df):
        """
        区域竞争力分析

        Parameters:
        -----------
        df : pd.DataFrame
            村庄/区域数据

        Returns:
        --------
        dict
            竞争力分析结果
        """
        existing_features = [f for f in self.competitiveness_features if f in df.columns]
        X = df[existing_features].values

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 使用快速聚类
        self.model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=20,
            n_init=3
        )
        self.labels = self.model.fit_predict(X_scaled)
        self.cluster_centers = self.model.cluster_centers_
        self.feature_names = existing_features

        result_df = df.copy()
        result_df['competitiveness_cluster'] = self.labels

        # 计算竞争力指数
        result_df['competitiveness_index'] = self._calculate_competitiveness_index(
            result_df, existing_features
        )

        return result_df

    def _calculate_competitiveness_index(self, df, features):
        """计算竞争力指数"""
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(df[features])

        # 加权平均
        weights = {
            'fusion_index': 0.25,
            'per_capita_income': 0.25,
            'population': 0.1,
            'coop_count': 0.15,
            'firm_count': 0.15,
            'processing_count': 0.1
        }

        weights_list = [weights.get(f, 0.1) for f in features]
        weights_list = np.array(weights_list) / sum(weights_list)

        index = np.dot(normalized, weights_list)

        return index

    def get_regional_ranking(self, df):
        """
        获取区域排名

        Parameters:
        -----------
        df : pd.DataFrame
            带竞争力指数的数据

        Returns:
        --------
        pd.DataFrame
            区域排名
        """
        if 'competitiveness_index' in df.columns:
            ranking = df.sort_values('competitiveness_index', ascending=False)
        else:
            ranking = df

        return ranking


if __name__ == '__main__':
    # 测试
    print("测试KMeans聚类分析模块...")

    # 生成测试数据
    from utils.data_generator import DataGenerator
    generator = DataGenerator()
    farmer_data = generator.generate_farmer_data(200)

    # 农户聚类
    analyzer = FarmerClusterAnalyzer(n_clusters=4)
    result = analyzer.cluster_farmers(farmer_data)

    print("\n聚类结果:")
    print(result['cluster_name'].value_counts())

    # 获取评估指标
    metrics = analyzer.get_evaluation_metrics(
        StandardScaler().fit_transform(
            result[analyzer.clustering_features].values
        )
    )
    print("\n评估指标:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
