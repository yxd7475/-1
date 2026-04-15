# -*- coding: utf-8 -*-
"""
数据处理模块 - 数据清洗、标准化和预处理
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """数据处理器"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.label_encoders = {}

    def clean_data(self, df):
        """
        数据清洗

        Parameters:
        -----------
        df : pd.DataFrame
            原始数据

        Returns:
        --------
        pd.DataFrame
            清洗后的数据
        """
        df_clean = df.copy()

        # 1. 去除完全重复的行
        df_clean = df_clean.drop_duplicates()

        # 2. 处理缺失值
        # 数值列：用中位数填充
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isna().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)

        # 分类列：用众数填充
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isna().sum() > 0:
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])

        # 3. 处理异常值（IQR方法）
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # 将异常值截断到边界
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

        # 4. 去除空白字符
        for col in categorical_cols:
            df_clean[col] = df_clean[col].astype(str).str.strip()

        return df_clean

    def normalize_features(self, df, columns, method='standard'):
        """
        特征标准化

        Parameters:
        -----------
        df : pd.DataFrame
            数据
        columns : list
            需要标准化的列
        method : str
            'standard' 或 'minmax'

        Returns:
        --------
        pd.DataFrame
            标准化后的数据
        """
        df_norm = df.copy()

        for col in columns:
            if col in df_norm.columns:
                if method == 'standard':
                    df_norm[f'{col}_norm'] = self.scaler.fit_transform(
                        df_norm[[col]]
                    )
                elif method == 'minmax':
                    df_norm[f'{col}_norm'] = self.minmax_scaler.fit_transform(
                        df_norm[[col]]
                    )

        return df_norm

    def encode_categorical(self, df, columns):
        """
        分类变量编码

        Parameters:
        -----------
        df : pd.DataFrame
            数据
        columns : list
            需要编码的列

        Returns:
        --------
        pd.DataFrame
            编码后的数据
        """
        df_enc = df.copy()

        for col in columns:
            if col in df_enc.columns:
                le = LabelEncoder()
                df_enc[f'{col}_encoded'] = le.fit_transform(df_enc[col].astype(str))
                self.label_encoders[col] = le

        return df_enc

    def create_features(self, df):
        """
        特征工程 - 创建新特征

        Parameters:
        -----------
        df : pd.DataFrame
            农户数据

        Returns:
        --------
        pd.DataFrame
            增加特征后的数据
        """
        df_feat = df.copy()

        # 人均土地
        if 'land_area' in df_feat.columns and 'labor' in df_feat.columns:
            df_feat['land_per_labor'] = df_feat['land_area'] / df_feat['labor']

        # 人均资本
        if 'capital' in df_feat.columns and 'labor' in df_feat.columns:
            df_feat['capital_per_labor'] = df_feat['capital'] / df_feat['labor']

        # 地块平均面积
        if 'land_area' in df_feat.columns and 'land_parcels' in df_feat.columns:
            df_feat['avg_parcel_area'] = df_feat['land_area'] / df_feat['land_parcels']

        # 综合融合指数
        fusion_cols = ['f_production', 'f_supply', 'f_market', 'f_service', 'f_value']
        existing_cols = [col for col in fusion_cols if col in df_feat.columns]
        if existing_cols:
            df_feat['fusion_index'] = df_feat[existing_cols].mean(axis=1)

        # 城镇可达性
        if 'dist_town' in df_feat.columns and 'dist_road' in df_feat.columns:
            df_feat['accessibility'] = 1 / (df_feat['dist_town'] + df_feat['dist_road'] + 1)

        # 收入风险比
        if 'annual_income' in df_feat.columns and 'risk_aversion' in df_feat.columns:
            df_feat['income_risk_ratio'] = df_feat['annual_income'] / (df_feat['risk_aversion'] + 0.1)

        return df_feat

    def prepare_clustering_data(self, df, feature_columns):
        """
        准备聚类数据

        Parameters:
        -----------
        df : pd.DataFrame
            原始数据
        feature_columns : list
            特征列

        Returns:
        --------
        tuple
            (特征矩阵, 特征列名列表)
        """
        # 选择存在的特征列
        existing_cols = [col for col in feature_columns if col in df.columns]

        # 提取特征
        X = df[existing_cols].copy()

        # 处理缺失值
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=existing_cols)

        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=existing_cols)

        return X_scaled, existing_cols

    def calculate_entropy_weights(self, df, columns):
        """
        熵值法计算权重

        Parameters:
        -----------
        df : pd.DataFrame
            数据
        columns : list
            指标列

        Returns:
        --------
        dict
            各指标权重
        """
        weights = {}
        n = len(df)

        for col in columns:
            if col in df.columns:
                # 标准化（正向化）
                x = df[col].values
                x_min, x_max = x.min(), x.max()
                if x_max - x_min > 0:
                    x_norm = (x - x_min) / (x_max - x_min)
                else:
                    x_norm = np.ones(n)

                # 避免log(0)
                x_norm = np.where(x_norm == 0, 1e-10, x_norm)
                x_norm = np.where(x_norm == 1, 1 - 1e-10, x_norm)

                # 计算熵值
                p = x_norm / x_norm.sum()
                e = -np.sum(p * np.log(p)) / np.log(n)

                # 计算权重
                d = 1 - e
                weights[col] = d

        # 归一化权重
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def process_pipeline(self, df, target_columns=None):
        """
        完整数据处理流水线

        Parameters:
        -----------
        df : pd.DataFrame
            原始数据
        target_columns : list
            目标特征列

        Returns:
        --------
        pd.DataFrame
            处理后的数据
        """
        print("开始数据处理流水线...")

        # 1. 清洗
        print("  1. 数据清洗...")
        df_clean = self.clean_data(df)

        # 2. 特征工程
        print("  2. 特征工程...")
        df_feat = self.create_features(df_clean)

        # 3. 标准化
        if target_columns:
            print("  3. 特征标准化...")
            existing_cols = [col for col in target_columns if col in df_feat.columns]
            df_final = self.normalize_features(df_feat, existing_cols, method='standard')
        else:
            df_final = df_feat

        print("数据处理完成!")
        return df_final


def load_data(file_path):
    """
    加载数据文件

    Parameters:
    -----------
    file_path : str
        文件路径

    Returns:
    --------
    pd.DataFrame
        数据
    """
    import os

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif ext == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def save_data(df, file_path):
    """
    保存数据

    Parameters:
    -----------
    df : pd.DataFrame
        数据
    file_path : str
        文件路径
    """
    import os

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.csv':
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
    elif ext == '.xlsx':
        df.to_excel(file_path, index=False, engine='openpyxl')
    elif ext == '.json':
        df.to_json(file_path, orient='records', force_ascii=False, indent=2)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

    print(f"数据已保存: {file_path}")


if __name__ == '__main__':
    # 测试
    from data_generator import DataGenerator

    generator = DataGenerator()
    data = generator.generate_farmer_data(100)

    processor = DataProcessor()
    processed = processor.process_pipeline(data)

    print(processed.head())
