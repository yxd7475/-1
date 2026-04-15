# -*- coding: utf-8 -*-
"""
城乡产业融合智能决策系统 - Streamlit主应用
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_generator import DataGenerator
from src.utils.data_processor import DataProcessor
from src.models.kmeans_model import FarmerClusterAnalyzer, IndustryClusterAnalyzer, RegionalCompetitivenessAnalyzer
from src.analysis.fusion_index import FusionIndexCalculator, FusionIndexVisualizer
from src.analysis.decision_support import DecisionSupportEngine, ReportGenerator
from src.models.abm_model import ABMSimulator

# 页面配置
st.set_page_config(
    page_title="城乡产业融合智能决策系统",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric > div {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }

    /* 加载动画样式 */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin: 20px 0;
    }

    .spinner {
        width: 60px;
        height: 60px;
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top: 4px solid #ffffff;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .loading-text {
        color: white;
        font-size: 18px;
        margin-top: 20px;
        font-weight: 500;
    }

    .progress-container {
        width: 100%;
        max-width: 300px;
        margin-top: 15px;
    }

    .progress-bar {
        height: 8px;
        background: rgba(255,255,255,0.3);
        border-radius: 10px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: white;
        border-radius: 10px;
        animation: progress 2s ease-in-out infinite;
    }

    @keyframes progress {
        0% { width: 0%; }
        50% { width: 70%; }
        100% { width: 100%; }
    }

    .pulse-dots {
        display: flex;
        gap: 8px;
        margin-top: 15px;
    }

    .pulse-dot {
        width: 12px;
        height: 12px;
        background: white;
        border-radius: 50%;
        animation: pulse 1.5s ease-in-out infinite;
    }

    .pulse-dot:nth-child(2) { animation-delay: 0.2s; }
    .pulse-dot:nth-child(3) { animation-delay: 0.4s; }

    @keyframes pulse {
        0%, 100% { opacity: 0.3; transform: scale(0.8); }
        50% { opacity: 1; transform: scale(1.2); }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_system():
    """初始化系统（资源缓存）"""
    generator = DataGenerator()
    processor = DataProcessor()
    fusion_calculator = FusionIndexCalculator()
    decision_engine = DecisionSupportEngine()
    report_generator = ReportGenerator()

    return {
        'generator': generator,
        'processor': processor,
        'fusion_calculator': fusion_calculator,
        'decision_engine': decision_engine,
        'report_generator': report_generator
    }


@st.cache_data
def generate_cached_data(n_farmers, n_villages, region_name, _generator):
    """缓存数据生成结果"""
    return _generator.generate_all_data(n_farmers, n_villages, region_name)


@st.cache_data
def perform_farmer_clustering(_df, n_clusters):
    """缓存农户聚类结果"""
    analyzer = FarmerClusterAnalyzer(n_clusters=n_clusters)
    return analyzer.cluster_farmers(_df)


@st.cache_data
def perform_industry_clustering(_df, n_clusters):
    """缓存产业聚类结果"""
    analyzer = IndustryClusterAnalyzer(n_clusters=n_clusters)
    return analyzer.cluster_industries(_df)


@st.cache_data
def perform_regional_analysis(_df, n_clusters):
    """缓存区域竞争力分析结果"""
    analyzer = RegionalCompetitivenessAnalyzer(n_clusters=n_clusters)
    return analyzer.analyze_regional_competitiveness(_df)


def show_loading_animation(title="正在处理", message="请稍候..."):
    """显示自定义加载动画"""
    st.markdown(f"""
    <div class="loading-container">
        <div class="spinner"></div>
        <div class="loading-text">{title}</div>
        <div class="progress-container">
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>
        </div>
        <div class="pulse-dots">
            <div class="pulse-dot"></div>
            <div class="pulse-dot"></div>
            <div class="pulse-dot"></div>
        </div>
        <div style="color: rgba(255,255,255,0.8); margin-top: 10px; font-size: 14px;">{message}</div>
    </div>
    """, unsafe_allow_html=True)


def show_animated_progress(steps, step_names):
    """显示带进度条的处理动画"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (step, name) in enumerate(zip(steps, step_names)):
        progress = (i + 1) / len(steps)
        progress_bar.progress(progress)
        status_text.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; padding: 10px; background: #f0f2f6; border-radius: 8px;">
            <div style="width: 20px; height: 20px; border: 2px solid #1f77b4; border-top-color: transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            <span style="font-size: 16px; color: #333;">{name}</span>
        </div>
        """, unsafe_allow_html=True)
        step()

    progress_bar.empty()
    status_text.empty()


def main():
    # 标题
    st.markdown('<h1 class="main-header">🌾 城乡产业融合智能决策系统</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # 初始化系统
    system = init_system()

    # 侧边栏
    st.sidebar.title("📋 功能导航")
    page = st.sidebar.radio(
        "选择功能模块",
        ["📊 数据概览", "🔍 KMeans聚类分析", "📈 融合指数分析", "🎯 智能决策支持", "⚙️ ABM模拟", "📋 综合报告"]
    )

    # 数据管理
    st.sidebar.markdown("---")
    st.sidebar.subheader("数据管理")
    n_farmers = st.sidebar.slider("农户数量", 100, 5000, 500, 100)  # 默认减少到500
    n_villages = st.sidebar.slider("村庄数量", 10, 200, 30, 10)
    region_name = st.sidebar.text_input("区域名称", "示范县")

    # 使用缓存生成数据
    data = generate_cached_data(n_farmers, n_villages, region_name, system['generator'])

    # 清除缓存按钮
    if st.sidebar.button("🔄 重新生成数据"):
        st.cache_data.clear()
        st.rerun()

    # 根据选择显示不同页面
    if page == "📊 数据概览":
        show_data_overview(data)
    elif page == "🔍 KMeans聚类分析":
        show_clustering_analysis(data, system)
    elif page == "📈 融合指数分析":
        show_fusion_analysis(data, system)
    elif page == "🎯 智能决策支持":
        show_decision_support(data, system)
    elif page == "⚙️ ABM模拟":
        show_abm_simulation()
    elif page == "📋 综合报告":
        show_comprehensive_report(data, system)


def show_data_overview(data):
    """数据概览页面"""
    st.markdown('<h2 class="sub-header">📊 数据概览</h2>', unsafe_allow_html=True)

    # 关键指标卡片
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("农户总数", f"{len(data['farmers']):,}")

    with col2:
        st.metric("村庄总数", f"{len(data['villages']):,}")

    with col3:
        st.metric("产业类型", f"{data['industries']['industry_type'].nunique()}")

    with col4:
        fusion_val = data['farmers']['fusion_index'].mean() if 'fusion_index' in data['farmers'].columns else 0.5
        st.metric("平均融合指数", f"{fusion_val:.2%}")

    st.markdown("---")

    # 数据表格
    tab1, tab2, tab3, tab4 = st.tabs(["农户数据", "村庄数据", "产业数据", "时间序列"])

    with tab1:
        st.subheader("农户数据样本")
        st.dataframe(data['farmers'].head(100), use_container_width=True)

        # 统计信息
        st.subheader("数据统计")
        numeric_cols = data['farmers'].select_dtypes(include=[np.number]).columns
        st.dataframe(data['farmers'][numeric_cols].describe().T, use_container_width=True)

    with tab2:
        st.subheader("村庄数据")
        st.dataframe(data['villages'], use_container_width=True)

    with tab3:
        st.subheader("产业数据")
        st.dataframe(data['industries'], use_container_width=True)

    with tab4:
        st.subheader("时间序列数据")
        st.dataframe(data['time_series'], use_container_width=True)

        # 趋势图
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['time_series']['date'],
            y=data['time_series']['gdp'],
            name='GDP',
            line=dict(color='blue')
        ))
        fig.update_layout(title="GDP趋势", xaxis_title="日期", yaxis_title="GDP(万元)")
        st.plotly_chart(fig, use_container_width=True)

    # 数据下载
    st.markdown("---")
    st.subheader("📥 数据下载")
    col1, col2 = st.columns(2)

    with col1:
        csv = data['farmers'].to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载农户数据(CSV)", csv, "farmers.csv", "text/csv")

    with col2:
        csv = data['villages'].to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载村庄数据(CSV)", csv, "villages.csv", "text/csv")


def show_clustering_analysis(data, system):
    """KMeans聚类分析页面"""
    st.markdown('<h2 class="sub-header">🔍 KMeans聚类分析</h2>', unsafe_allow_html=True)

    # 聚类类型选择
    cluster_type = st.selectbox("选择聚类类型", ["农户聚类", "产业聚类", "区域竞争力分析"])

    if cluster_type == "农户聚类":
        show_farmer_clustering(data, system)
    elif cluster_type == "产业聚类":
        show_industry_clustering(data, system)
    else:
        show_regional_clustering(data, system)


def show_farmer_clustering(data, system):
    """农户聚类分析 - 真实KMeans迭代演示（高性能版）"""
    st.subheader("农户聚类分析")

    # 蓝色系颜色
    blue_colors = ['#1f77b4', '#4299e1', '#63b3ed', '#90cdf4', '#bee3f8',
                   '#3182ce', '#2b6cb0', '#2c5282', '#2a4365', '#1A365D']

    # 参数设置 - 优化为更合理的范围
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_clusters = st.slider("聚类数量", 2, 8, 4, 1)
    with col2:
        max_iterations = st.slider("迭代次数", 3, 20, 10, 1)
    with col3:
        sample_size = st.slider("样本数", 50, 300, 150, 50)
    with col4:
        speed = st.slider("动画速度", 0.3, 2.0, 0.8, 0.1)

    st.markdown("---")

    # 准备数据（采样以提升速度）
    farmer_df = data['farmers'].copy()
    if len(farmer_df) > sample_size:
        farmer_df = farmer_df.sample(sample_size, random_state=42)

    # 确保融合指数列存在
    if 'fusion_index' not in farmer_df.columns:
        fusion_cols = ['f_production', 'f_supply', 'f_market', 'f_service', 'f_value']
        existing_cols = [c for c in fusion_cols if c in farmer_df.columns]
        if existing_cols:
            farmer_df['fusion_index'] = farmer_df[existing_cols].mean(axis=1)

    # 提取聚类特征
    feature_cols = ['land_area', 'capital', 'labor', 'dist_town', 'fusion_index']
    existing_features = [c for c in feature_cols if c in farmer_df.columns]

    # 标准化数据
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(farmer_df[existing_features].values)

    st.info(f"📊 当前演示: {len(farmer_df)} 个样本 | {n_clusters} 个聚类 | 最多 {max_iterations} 次迭代")

    # 开始演示按钮
    if st.button("▶️ 开始动态演示", type="primary", use_container_width=True):
        # 导入演示模块
        from src.utils.kmeans_demo import run_kmeans_demo, show_iteration_curves, show_clustering_results

        # 运行演示
        labels, iteration_history = run_kmeans_demo(
            X, n_clusters, max_iterations, speed, blue_colors
        )

        # 显示完成信息
        st.success(f"🎉 聚类完成! 共迭代 {len(iteration_history)} 次")

        # 显示迭代曲线
        st.markdown("### 📈 迭代过程变化曲线")
        show_iteration_curves(iteration_history, blue_colors)

        # 显示聚类结果
        st.markdown("---")
        farmer_df = show_clustering_results(farmer_df, labels, n_clusters, blue_colors)


def show_industry_clustering(data, system):
    """产业聚类分析"""
    st.subheader("产业聚类分析")

    industry_df = data['industries'].copy()

    # 显示KMeans动态演示过程
    st.markdown("### 🔄 产业聚类分析演示")
    demo_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    import time

    # 阶段1: 数据加载
    with demo_placeholder.container():
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.info("📊 **步骤 1/4**: 加载产业数据...")
            st.metric("产业记录数", f"{len(industry_df)} 条")
            st.metric("产业类型", f"{industry_df['industry_type'].nunique()} 种")
        with col_b:
            # 产业类型分布
            type_counts = industry_df['industry_type'].value_counts().head(8)
            fig1 = px.bar(x=type_counts.index, y=type_counts.values,
                         title="产业类型分布", labels={'x': '产业类型', 'y': '数量'})
            fig1.update_layout(height=250)
            st.plotly_chart(fig1, use_container_width=True)
    progress_bar.progress(25)
    status_text.markdown("⏳ 正在计算产业特征矩阵...")
    time.sleep(0.3)

    # 阶段2: 特征提取
    with demo_placeholder.container():
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.info("🎯 **步骤 2/4**: 提取聚类特征")
            st.write("- 产业融合度")
            st.write("- 就业人数")
            st.write("- 营业收入")
            st.write("- 利润总额")
        with col_b:
            # 显示特征分布
            fig2 = px.scatter(industry_df, x='fusion_degree', y='profit',
                            title="产业特征分布", opacity=0.6,
                            labels={'fusion_degree': '融合度', 'profit': '利润'})
            fig2.update_layout(height=250)
            st.plotly_chart(fig2, use_container_width=True)
    progress_bar.progress(50)
    status_text.markdown("⏳ 正在执行KMeans聚类算法...")
    time.sleep(0.3)

    # 阶段3: 聚类迭代
    with demo_placeholder.container():
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.info("🔄 **步骤 3/4**: 聚类迭代过程")
            st.write("聚类数: **5** 个")
            st.write("迭代中... 样本重新分配")
        with col_b:
            # 模拟聚类过程
            fig3 = px.scatter(industry_df, x='fusion_degree', y='profit',
                            title="聚类迭代中...", opacity=0.5)
            colors = np.random.randint(0, 5, len(industry_df))
            fig3.update_traces(marker=dict(color=colors, colorscale='Viridis'))
            fig3.update_layout(height=250, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
    progress_bar.progress(75)
    status_text.markdown("⏳ 正在计算聚类中心...")
    time.sleep(0.3)

    # 执行实际聚类
    result = perform_industry_clustering(industry_df, 5)

    # 阶段4: 完成
    with demo_placeholder.container():
        st.success("🎉 **步骤 4/4**: 聚类完成!")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("产业总数", f"{len(result)}")
        with col_b:
            st.metric("聚类数量", "5")
        with col_c:
            st.metric("最大聚类", f"{result['cluster'].value_counts().max()}")
    progress_bar.progress(100)
    status_text.success("✅ 产业聚类分析完成!")
    time.sleep(0.3)

    # 清除演示过程
    demo_placeholder.empty()
    progress_bar.empty()
    status_text.empty()

    # 显示结果
    st.markdown("---")
    st.markdown("### 📊 产业聚类结果")

    # 聚类分布
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            result,
            x='fusion_degree',
            y='profit',
            color='cluster',
            size='employment',
            hover_data=['industry_type'],
            title="产业聚类分布",
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 优势产业
        st.subheader("优势产业识别")
        # 计算优势产业
        if 'fusion_degree' in result.columns and 'profit' in result.columns:
            result['advantage_score'] = (
                result['fusion_degree'] * 0.3 +
                result['profit'] / result['profit'].max() * 0.3 +
                result.get('employment', pd.Series([1]*len(result))) / result['employment'].max() * 0.2 +
                result.get('revenue', pd.Series([1]*len(result))) / result['revenue'].max() * 0.2
            )
            advantage = result.groupby('industry_type').agg({
                'advantage_score': 'mean',
                'revenue': 'sum',
                'profit': 'sum',
                'employment': 'sum',
                'fusion_degree': 'mean'
            }).sort_values('advantage_score', ascending=False).head(5)
            st.dataframe(advantage, use_container_width=True)


def show_regional_clustering(data, system):
    """区域竞争力分析"""
    st.subheader("区域竞争力分析")

    village_df = data['villages'].copy()

    # 显示动态演示过程
    st.markdown("### 🔄 区域竞争力分析演示")
    demo_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    import time

    # 阶段1: 数据加载
    with demo_placeholder.container():
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.info("📊 **步骤 1/4**: 加载村庄数据...")
            st.metric("村庄总数", f"{len(village_df)} 个")
            st.metric("平均人口", f"{village_df['population'].mean():.0f} 人")
        with col_b:
            # 村庄人口分布
            fig1 = px.histogram(village_df, x='population', nbins=20,
                              title="村庄人口分布", labels={'population': '人口'})
            fig1.update_layout(height=250)
            st.plotly_chart(fig1, use_container_width=True)
    progress_bar.progress(25)
    status_text.markdown("⏳ 正在构建竞争力指标体系...")
    time.sleep(0.3)

    # 阶段2: 指标计算
    with demo_placeholder.container():
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.info("🎯 **步骤 2/4**: 计算竞争力指标")
            st.write("- 融合指数")
            st.write("- 人均收入")
            st.write("- 合作社数量")
            st.write("- 企业数量")
        with col_b:
            # 指标雷达图
            categories = ['融合指数', '人均收入', '合作社', '企业', '加工厂']
            values = [
                village_df['fusion_index'].mean() if 'fusion_index' in village_df else 0.5,
                village_df['per_capita_income'].mean() / 40000 if 'per_capita_income' in village_df else 0.5,
                village_df['coop_count'].mean() / 5 if 'coop_count' in village_df else 0.5,
                village_df['firm_count'].mean() / 3 if 'firm_count' in village_df else 0.5,
                village_df['processing_count'].mean() / 2 if 'processing_count' in village_df else 0.5
            ]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='平均指标'))
            fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=250)
            st.plotly_chart(fig2, use_container_width=True)
    progress_bar.progress(50)
    status_text.markdown("⏳ 正在执行聚类分析...")
    time.sleep(0.3)

    # 阶段3: 聚类分配
    with demo_placeholder.container():
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.info("🔄 **步骤 3/4**: 区域聚类分组")
            st.write("聚类数: **3** 个等级")
            st.write("- 高竞争力区域")
            st.write("- 中竞争力区域")
            st.write("- 低竞争力区域")
        with col_b:
            # 模拟聚类结果
            fig3 = px.scatter(village_df, x='population', y='per_capita_income' if 'per_capita_income' in village_df else 'population',
                            title="区域聚类分布", opacity=0.6)
            colors = np.random.randint(0, 3, len(village_df))
            fig3.update_traces(marker=dict(color=colors, colorscale='RdYlGn'))
            fig3.update_layout(height=250, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
    progress_bar.progress(75)
    status_text.markdown("⏳ 正在生成竞争力排名...")
    time.sleep(0.3)

    # 执行实际分析
    result = perform_regional_analysis(village_df, 3)

    # 阶段4: 完成
    with demo_placeholder.container():
        st.success("🎉 **步骤 4/4**: 分析完成!")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("分析村庄", f"{len(result)}")
        with col_b:
            if 'competitiveness_index' in result.columns:
                st.metric("最高竞争力", f"{result['competitiveness_index'].max():.3f}")
            else:
                st.metric("聚类数", "3")
        with col_c:
            if 'competitiveness_index' in result.columns:
                st.metric("平均竞争力", f"{result['competitiveness_index'].mean():.3f}")
            else:
                st.metric("状态", "完成")
    progress_bar.progress(100)
    status_text.success("✅ 区域竞争力分析完成!")
    time.sleep(0.3)

    # 清除演示过程
    demo_placeholder.empty()
    progress_bar.empty()
    status_text.empty()

    # 显示结果
    st.markdown("---")
    st.markdown("### 📊 区域竞争力结果")

    # 竞争力排名
    if 'competitiveness_index' in result.columns:
        ranking = result.sort_values('competitiveness_index', ascending=False)
    else:
        ranking = result

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("区域竞争力排名TOP10")
        display_cols = ['village_name', 'competitiveness_index'] if 'competitiveness_index' in ranking.columns else ['village_name']
        if 'fusion_index' in ranking.columns:
            display_cols.append('fusion_index')
        st.dataframe(ranking[display_cols].head(10), use_container_width=True)

    with col2:
        fig = px.bar(
            ranking.head(15),
            x='village_name',
            y='competitiveness_index' if 'competitiveness_index' in ranking.columns else 'fusion_index',
            color='competitiveness_cluster' if 'competitiveness_cluster' in ranking.columns else None,
            title="区域竞争力指数",
            color_continuous_scale='RdYlGn'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)


def show_fusion_analysis(data, system):
    """融合指数分析页面"""
    st.markdown('<h2 class="sub-header">📈 融合指数分析</h2>', unsafe_allow_html=True)

    calculator = system['fusion_calculator']
    farmer_df = data['farmers'].copy()

    # 确保融合指数列存在
    if 'fusion_index' not in farmer_df.columns:
        fusion_cols = ['f_production', 'f_supply', 'f_market', 'f_service', 'f_value']
        existing_cols = [c for c in fusion_cols if c in farmer_df.columns]
        if existing_cols:
            farmer_df['fusion_index'] = farmer_df[existing_cols].mean(axis=1)
        else:
            farmer_df['fusion_index'] = 0.5

    # 综合融合指数
    st.subheader("融合指数概览")

    col1, col2, col3, col4, col5 = st.columns(5)
    fusion_cols = ['f_production', 'f_supply', 'f_market', 'f_service', 'f_value']
    names = ['生产融合', '供应融合', '市场融合', '服务融合', '价值融合']

    for col, name in zip([col1, col2, col3, col4, col5], names):
        f_col = f'f_{name[0]}'
        if f_col in farmer_df:
            with col:
                st.metric(name, f"{farmer_df[f_col].mean():.2%}")

    st.markdown("---")

    # 融合指数分布
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("融合指数分布")
        fig = px.histogram(
            farmer_df,
            x='fusion_index',
            nbins=30,
            title="综合融合指数分布",
            marginal='box'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 雷达图
        st.subheader("平均融合指数雷达图")
        avg_fusion = {
            'production': farmer_df['f_production'].mean(),
            'supply': farmer_df['f_supply'].mean(),
            'market': farmer_df['f_market'].mean(),
            'service': farmer_df['f_service'].mean(),
            'value': farmer_df['f_value'].mean()
        }

        categories = ['生产融合', '供应融合', '市场融合', '服务融合', '价值融合']
        values = list(avg_fusion.values())
        values += values[:1]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name='平均融合指数'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # 按状态对比
    st.subheader("不同组织参与状态的融合指数对比")
    state_fusion = farmer_df.groupby('state_name')[fusion_cols].mean()

    fig = go.Figure()
    for col in fusion_cols:
        fig.add_trace(go.Bar(
            x=state_fusion.index,
            y=state_fusion[col],
            name=col.replace('f_', '')
        ))
    fig.update_layout(barmode='group', title="各状态融合维度对比")
    st.plotly_chart(fig, use_container_width=True)

    # 差距分析与建议
    st.subheader("融合差距分析与改进建议")
    analysis = calculator.analyze_fusion_gap(farmer_df)

    col1, col2 = st.columns(2)

    with col1:
        st.write("**差距分析**")
        gap_df = pd.DataFrame(analysis['gaps']).T
        gap_df.columns = ['最大值', '平均值', '差距', '差距比例']
        st.dataframe(gap_df, use_container_width=True)

        st.write(f"**短板维度**: {', '.join(analysis['shortcomings'])}")

    with col2:
        st.write("**改进建议**")
        suggestions = calculator.generate_improvement_suggestions(analysis)
        for sug in suggestions:
            st.markdown(f"### {sug['title']}")
            for measure in sug['measures']:
                st.write(f"- {measure}")
            st.write("**政策支持**:", ", ".join(sug['policies']))


def show_decision_support(data, system):
    """智能决策支持页面"""
    st.markdown('<h2 class="sub-header">🎯 智能决策支持</h2>', unsafe_allow_html=True)

    decision_engine = system['decision_engine']

    # 决策场景选择
    scenario = st.selectbox(
        "选择决策场景",
        ["产业规划", "项目选址", "风险评估", "投资分析"]
    )

    if scenario == "产业规划":
        show_industry_planning(data, decision_engine)
    elif scenario == "项目选址":
        show_site_selection(data, decision_engine)
    elif scenario == "风险评估":
        show_risk_assessment(data, decision_engine)
    else:
        show_investment_analysis(data, decision_engine)


def show_industry_planning(data, engine):
    """产业规划"""
    st.subheader("产业规划建议")

    industry_df = data['industries'].copy()

    with st.spinner("正在生成产业规划..."):
        planning = engine.generate_industry_planning(industry_df)

    # 产业诊断
    st.subheader("📋 产业诊断")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("总产值", f"{planning['diagnosis']['total_revenue']:,.0f} 万元")
    with col2:
        st.metric("总利润", f"{planning['diagnosis']['total_profit']:,.0f} 万元")
    with col3:
        st.metric("平均融合度", f"{planning['diagnosis']['avg_fusion_degree']:.2%}")

    # 问题识别
    if planning['diagnosis']['issues']:
        st.subheader("⚠️ 问题识别")
        for issue in planning['diagnosis']['issues']:
            st.warning(issue)

    # 优先发展产业
    st.subheader("🎯 优先发展产业")
    priority_df = pd.DataFrame(planning['priority_industries'])
    if not priority_df.empty:
        st.dataframe(priority_df[['industry_type', 'revenue', 'profit', 'employment', 'fusion_degree', 'potential_score']], use_container_width=True)

    # 政策建议
    st.subheader("📜 政策建议")
    for rec in planning['policy_recommendations']:
        with st.expander(f"📌 {rec['type']} (优先级: {rec['priority']})"):
            st.write(f"**内容**: {rec['content']}")
            st.write(f"**预期效果**: {rec['expected_effect']}")


def show_site_selection(data, engine):
    """项目选址"""
    st.subheader("项目选址分析")

    col1, col2 = st.columns(2)

    with col1:
        project_type = st.selectbox("项目类型", ["农产品加工", "乡村旅游", "农业服务"])

    with col2:
        min_area = st.number_input("最小面积(亩)", 100, 10000, 1000)

    if st.button("生成选址建议"):
        village_df = data['villages'].copy()

        with st.spinner("正在分析选址..."):
            result = engine.generate_site_selection(village_df, project_type, {'min_area': min_area})

        st.success("分析完成!")

        # 推荐地点
        st.subheader("📍 推荐选址地点")
        for i, rec in enumerate(result['recommendations'], 1):
            with st.expander(f"TOP {i}: {rec['location_name']} (得分: {rec['score']:.3f})"):
                st.write("**各维度得分**:")
                for dim, score in rec['dimension_scores'].items():
                    st.write(f"- {dim}: {score:.3f}")

        # 选址标准
        st.subheader("📋 选址标准")
        for criteria in result['selection_criteria']:
            st.write(f"- {criteria}")


def show_risk_assessment(data, engine):
    """风险评估"""
    st.subheader("风险评估")

    industry_df = data['industries'].copy()

    with st.spinner("正在进行风险评估..."):
        risks = engine.generate_risk_assessment(industry_df)

    # 综合风险
    st.subheader("📊 综合风险等级")

    level_colors = {'red': '🔴', 'yellow': '🟡', 'blue': '🟢'}
    level = risks['overall']['level']

    st.markdown(f"### {level_colors[level]} {risks['overall']['description']}")
    st.metric("综合风险得分", f"{risks['overall']['score']:.2%}")

    # 各类风险
    st.subheader("📈 分项风险评估")

    risk_names = {
        'market_risk': '市场风险',
        'production_risk': '生产风险',
        'policy_risk': '政策风险',
        'financial_risk': '金融风险'
    }

    for key, name in risk_names.items():
        risk = risks[key]
        with st.expander(f"📌 {name} (得分: {risk['score']:.2%})"):
            st.write("**风险因素**:")
            for factor in risk['factors']:
                st.write(f"- {factor}")

            st.write("**缓解措施**:")
            for mitigation in risk['mitigation']:
                st.write(f"- {mitigation}")


def show_investment_analysis(data, engine):
    """投资分析"""
    st.subheader("投资分析")

    col1, col2 = st.columns(2)

    with col1:
        project_name = st.text_input("项目名称", "农产品加工项目")
        investment = st.number_input("投资金额(万元)", 10, 10000, 500)
        expected_revenue = st.number_input("预期年收入(万元)", 10, 50000, 800)

    with col2:
        operating_cost = st.number_input("年运营成本(万元)", 5, 30000, 400)
        project_life = st.slider("项目周期(年)", 3, 20, 10)

    if st.button("进行投资分析"):
        project_info = {
            'name': project_name,
            'investment': investment * 10000,
            'expected_revenue': expected_revenue * 10000,
            'operating_cost': operating_cost * 10000,
            'project_life': project_life
        }

        with st.spinner("正在进行投资分析..."):
            analysis = engine.generate_investment_analysis(project_info, data['industries'])

        st.success("分析完成!")

        # 财务指标
        st.subheader("💰 财务指标")
        indicators = analysis['financial_indicators']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("投资回报率(ROI)", indicators['roi'])
            st.metric("回收期", indicators['payback_period'])
        with col2:
            st.metric("净现值(NPV)", indicators['npv'])
            st.metric("内部收益率(IRR)", indicators['irr'])
        with col3:
            st.metric("年利润", f"{indicators['annual_profit']/10000:.0f}万元")

        # 可行性评估
        st.subheader("📊 可行性评估")
        feasibility = analysis['feasibility']
        st.metric("评估等级", feasibility['level'])
        st.metric("综合得分", f"{feasibility['score']}/100")
        st.info(f"**建议**: {feasibility['recommendation']}")

        # 敏感性分析
        st.subheader("📉 敏感性分析")
        sens_df = pd.DataFrame(analysis['sensitivity_analysis'])
        fig = px.bar(sens_df, x='scenario', y=[float(v.replace('万元', '')) for v in sens_df['npv']], title="NPV敏感性分析")
        st.plotly_chart(fig, use_container_width=True)


def show_abm_simulation():
    """ABM模拟页面"""
    st.markdown('<h2 class="sub-header">⚙️ ABM模拟</h2>', unsafe_allow_html=True)

    st.info("基于主体的建模(ABM)模拟农户行为演化过程")

    # 参数设置
    col1, col2, col3 = st.columns(3)

    with col1:
        n_farmers = st.slider("农户数量", 50, 500, 100, 50)
    with col2:
        max_ticks = st.slider("模拟周期", 20, 200, 50, 10)
    with col3:
        run_sim = st.button("运行模拟")

    if run_sim:
        with st.spinner("正在运行ABM模拟..."):
            simulator = ABMSimulator(n_farmers=n_farmers)
            results = simulator.run(max_ticks=max_ticks)

        st.success("模拟完成!")

        # 状态分布
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("最终状态分布")
            state_dist = results['state_name'].value_counts()
            fig = px.pie(values=state_dist.values, names=state_dist.index, title="农户状态分布")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("状态演化")
            history_df = simulator.get_history_df()

            fig = go.Figure()
            for state in [1, 2, 3, 4]:
                fig.add_trace(go.Scatter(
                    x=history_df['tick'],
                    y=history_df[f'state_{state}'],
                    name=f'状态{state}',
                    mode='lines'
                ))
            fig.update_layout(title="状态演化曲线", xaxis_title="时间", yaxis_title="农户数量")
            st.plotly_chart(fig, use_container_width=True)

        # 收入演化
        st.subheader("平均收入演化")
        fig = px.line(history_df, x='tick', y='avg_income', title="平均收入演化")
        st.plotly_chart(fig, use_container_width=True)

        # 空间分布
        st.subheader("农户空间分布")
        fig = px.scatter(
            results,
            x='x',
            y='y',
            color='state_name',
            size='land_area',
            hover_data=['current_income', 'fusion_index'],
            title="农户空间分布图"
        )
        st.plotly_chart(fig, use_container_width=True)

        # 结果数据
        st.subheader("模拟结果数据")
        st.dataframe(results.head(100), use_container_width=True)


def show_comprehensive_report(data, system):
    """综合报告页面 - 完整版"""
    st.markdown('<h2 class="sub-header">📋 综合分析报告</h2>', unsafe_allow_html=True)

    # 蓝色系颜色
    blue_colors = ['#1f77b4', '#4299e1', '#63b3ed', '#90cdf4', '#bee3f8',
                   '#3182ce', '#2b6cb0', '#2c5282', '#2a4365', '#1A365D']

    farmer_df = data['farmers'].copy()
    village_df = data['villages'].copy()
    industry_df = data['industries'].copy()
    time_series_df = data['time_series'].copy()

    # 确保融合指数存在
    if 'fusion_index' not in farmer_df.columns:
        fusion_cols = ['f_production', 'f_supply', 'f_market', 'f_service', 'f_value']
        existing_cols = [c for c in fusion_cols if c in farmer_df.columns]
        if existing_cols:
            farmer_df['fusion_index'] = farmer_df[existing_cols].mean(axis=1)

    # ===== 第一部分：数据概览 =====
    st.markdown("## 📊 第一部分：数据概览")
    st.markdown("---")

    # 关键指标卡片
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("农户总数", f"{len(farmer_df):,}")
    with col2:
        st.metric("村庄总数", f"{len(village_df):,}")
    with col3:
        st.metric("产业记录", f"{len(industry_df):,}")
    with col4:
        avg_fusion = farmer_df['fusion_index'].mean() if 'fusion_index' in farmer_df.columns else 0.5
        st.metric("平均融合指数", f"{avg_fusion:.2%}")
    with col5:
        avg_income = farmer_df['annual_income'].mean() if 'annual_income' in farmer_df.columns else 25000
        st.metric("平均农户收入", f"{avg_income:,.0f}元")

    # 数据统计表格
    with st.expander("📋 查看详细数据统计", expanded=False):
        tab1, tab2, tab3 = st.tabs(["农户数据统计", "村庄数据统计", "产业数据统计"])
        with tab1:
            numeric_cols = farmer_df.select_dtypes(include=[np.number]).columns
            st.dataframe(farmer_df[numeric_cols].describe().T, use_container_width=True)
        with tab2:
            numeric_cols = village_df.select_dtypes(include=[np.number]).columns
            st.dataframe(village_df[numeric_cols].describe().T, use_container_width=True)
        with tab3:
            numeric_cols = industry_df.select_dtypes(include=[np.number]).columns
            st.dataframe(industry_df[numeric_cols].describe().T, use_container_width=True)

    # ===== 第二部分：KMeans聚类分析 =====
    st.markdown("## 🔍 第二部分：KMeans聚类分析")
    st.markdown("---")

    # 执行聚类
    with st.spinner("正在进行聚类分析..."):
        # 农户聚类
        farmer_cluster_result = perform_farmer_clustering(farmer_df, 4)
        if 'cluster' not in farmer_cluster_result.columns:
            farmer_cluster_result['cluster'] = farmer_df.get('cluster', 0)
        if 'cluster_name' not in farmer_cluster_result.columns:
            farmer_cluster_result['cluster_name'] = farmer_cluster_result['cluster'].apply(
                lambda x: f'聚类{x+1}'
            )

    # 聚类分布图
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("农户聚类分布")
        if 'cluster_name' in farmer_cluster_result.columns:
            cluster_dist = farmer_cluster_result['cluster_name'].value_counts()
            fig = px.pie(values=cluster_dist.values, names=cluster_dist.index,
                         title="农户聚类分布", color_discrete_sequence=blue_colors)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("各聚类特征对比")
        feature_cols = ['land_area', 'capital', 'fusion_index', 'annual_income']
        existing_cols = [c for c in feature_cols if c in farmer_cluster_result.columns]
        if existing_cols:
            cluster_means = farmer_cluster_result.groupby('cluster_name')[existing_cols].mean()
            fig = go.Figure()
            for i, col in enumerate(existing_cols):
                fig.add_trace(go.Bar(
                    x=cluster_means.index,
                    y=cluster_means[col],
                    name=col,
                    marker_color=blue_colors[i % len(blue_colors)]
                ))
            fig.update_layout(barmode='group', title="各聚类特征均值对比")
            st.plotly_chart(fig, use_container_width=True)

    # 聚类详情表格
    st.subheader("聚类详情")
    cluster_summary = []
    for cluster_name in farmer_cluster_result['cluster_name'].unique():
        cluster_data = farmer_cluster_result[farmer_cluster_result['cluster_name'] == cluster_name]
        summary = {
            '聚类名称': cluster_name,
            '农户数量': len(cluster_data),
            '占比': f"{len(cluster_data)/len(farmer_cluster_result)*100:.1f}%",
            '平均收入': f"{cluster_data['annual_income'].mean():,.0f}元" if 'annual_income' in cluster_data else 'N/A',
            '平均融合指数': f"{cluster_data['fusion_index'].mean():.2%}" if 'fusion_index' in cluster_data else 'N/A',
            '平均土地面积': f"{cluster_data['land_area'].mean():.1f}亩" if 'land_area' in cluster_data else 'N/A'
        }
        cluster_summary.append(summary)
    st.dataframe(pd.DataFrame(cluster_summary), use_container_width=True)

    # ===== 第三部分：融合指数分析 =====
    st.markdown("## 📈 第三部分：融合指数分析")
    st.markdown("---")

    # 五维融合指数
    col1, col2, col3, col4, col5 = st.columns(5)
    fusion_names = ['生产融合', '供应融合', '市场融合', '服务融合', '价值融合']
    fusion_cols = ['f_production', 'f_supply', 'f_market', 'f_service', 'f_value']

    for col, name, f_col in zip([col1, col2, col3, col4, col5], fusion_names, fusion_cols):
        if f_col in farmer_df.columns:
            with col:
                st.metric(name, f"{farmer_df[f_col].mean():.2%}")

    # 融合指数分布
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("融合指数分布")
        fig = px.histogram(farmer_df, x='fusion_index', nbins=30,
                          title="综合融合指数分布", marginal='box',
                          color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("五维融合雷达图")
        fusion_radar_data = {
            '生产融合': farmer_df['f_production'].mean() if 'f_production' in farmer_df else 0.5,
            '供应融合': farmer_df['f_supply'].mean() if 'f_supply' in farmer_df else 0.5,
            '市场融合': farmer_df['f_market'].mean() if 'f_market' in farmer_df else 0.5,
            '服务融合': farmer_df['f_service'].mean() if 'f_service' in farmer_df else 0.5,
            '价值融合': farmer_df['f_value'].mean() if 'f_value' in farmer_df else 0.5
        }
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(fusion_radar_data.values()),
            theta=list(fusion_radar_data.keys()),
            fill='toself',
            name='平均融合指数',
            line=dict(color='#1f77b4')
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=400)
        st.plotly_chart(fig, use_container_width=True)

    # 融合等级统计
    st.subheader("融合等级分布")
    high_fusion = len(farmer_df[farmer_df['fusion_index'] > 0.6])
    medium_fusion = len(farmer_df[(farmer_df['fusion_index'] >= 0.4) & (farmer_df['fusion_index'] <= 0.6)])
    low_fusion = len(farmer_df[farmer_df['fusion_index'] < 0.4])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("高融合度农户", f"{high_fusion}户", f"{high_fusion/len(farmer_df)*100:.1f}%")
    with col2:
        st.metric("中融合度农户", f"{medium_fusion}户", f"{medium_fusion/len(farmer_df)*100:.1f}%")
    with col3:
        st.metric("低融合度农户", f"{low_fusion}户", f"{low_fusion/len(farmer_df)*100:.1f}%")

    # ===== 第四部分：产业分析 =====
    st.markdown("## 🏭 第四部分：产业分析")
    st.markdown("---")

    # 产业概况
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("产业类型数", industry_df['industry_type'].nunique() if 'industry_type' in industry_df else 0)
    with col2:
        st.metric("总收入", f"{industry_df['revenue'].sum():,.0f}万元" if 'revenue' in industry_df else 'N/A')
    with col3:
        st.metric("总利润", f"{industry_df['profit'].sum():,.0f}万元" if 'profit' in industry_df else 'N/A')
    with col4:
        st.metric("总就业", f"{industry_df['employment'].sum():,}人" if 'employment' in industry_df else 'N/A')

    # 产业类型分布
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("产业类型分布")
        if 'industry_type' in industry_df:
            type_dist = industry_df['industry_type'].value_counts()
            fig = px.bar(x=type_dist.index, y=type_dist.values,
                        title="产业类型数量分布", color_discrete_sequence=['#1f77b4'])
            fig.update_layout(xaxis_title="产业类型", yaxis_title="数量")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("产业融合度分布")
        if 'fusion_degree' in industry_df:
            fig = px.histogram(industry_df, x='fusion_degree', nbins=20,
                              title="产业融合度分布", color_discrete_sequence=['#4299e1'])
            st.plotly_chart(fig, use_container_width=True)

    # 优势产业识别
    st.subheader("优势产业识别")
    if 'revenue' in industry_df and 'industry_type' in industry_df:
        advantage = industry_df.groupby('industry_type').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'employment': 'sum',
            'fusion_degree': 'mean' if 'fusion_degree' in industry_df else 'count'
        }).sort_values('revenue', ascending=False).head(10)
        advantage.columns = ['总收入', '总利润', '就业人数', '平均融合度']
        st.dataframe(advantage, use_container_width=True)

    # ===== 第五部分：风险评估 =====
    st.markdown("## ⚠️ 第五部分：风险评估")
    st.markdown("---")

    decision_engine = system['decision_engine']
    risks = decision_engine.generate_risk_assessment(industry_df)

    # 风险等级
    level_colors = {'red': '🔴 高风险', 'yellow': '🟡 中风险', 'blue': '🟢 低风险'}

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("综合风险等级", level_colors.get(risks['overall']['level'], risks['overall']['level']))
        st.metric("综合风险得分", f"{risks['overall']['score']:.2%}")

    with col2:
        # 各类风险得分
        risk_data = {
            '风险类型': ['市场风险', '生产风险', '政策风险', '金融风险'],
            '风险得分': [risks['market_risk']['score'], risks['production_risk']['score'],
                        risks['policy_risk']['score'], risks['financial_risk']['score']]
        }
        fig = px.bar(risk_data, x='风险类型', y='风险得分',
                    title="各类风险得分", color='风险得分',
                    color_continuous_scale='RdYlGn_r')
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    # 风险详情
    with st.expander("📋 查看风险详情", expanded=False):
        for risk_name, risk_info in [('市场风险', risks['market_risk']),
                                      ('生产风险', risks['production_risk']),
                                      ('政策风险', risks['policy_risk']),
                                      ('金融风险', risks['financial_risk'])]:
            st.markdown(f"**{risk_name}** (得分: {risk_info['score']:.2%})")
            st.write("风险因素:", ", ".join(risk_info['factors']))
            st.write("缓解措施:", ", ".join(risk_info['mitigation']))
            st.markdown("---")

    # ===== 第六部分：AI智能分析 =====
    st.markdown("## 🤖 第六部分：AI智能分析")
    st.markdown("---")

    # 准备详细的数据摘要
    # 构建聚类详情字典
    cluster_details = {}
    for cluster_name in farmer_cluster_result['cluster_name'].unique():
        cluster_data = farmer_cluster_result[farmer_cluster_result['cluster_name'] == cluster_name]
        cluster_details[cluster_name] = {
            'count': len(cluster_data),
            'percentage': len(cluster_data)/len(farmer_cluster_result)*100,
            'avg_fusion': float(cluster_data['fusion_index'].mean()) if 'fusion_index' in cluster_data else 0.5,
            'avg_income': float(cluster_data['annual_income'].mean()) if 'annual_income' in cluster_data else 25000,
            'avg_land': float(cluster_data['land_area'].mean()) if 'land_area' in cluster_data else 10
        }

    # 优势产业数据
    top_industries = []
    if 'revenue' in industry_df and 'industry_type' in industry_df:
        industry_grouped = industry_df.groupby('industry_type').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'employment': 'sum',
            'fusion_degree': 'mean' if 'fusion_degree' in industry_df else 'count'
        }).sort_values('revenue', ascending=False)
        for idx, row in industry_grouped.head(5).iterrows():
            top_industries.append({
                'type': idx,
                'revenue': float(row['revenue']),
                'profit': float(row['profit']),
                'fusion': float(row['fusion_degree']) if 'fusion_degree' in industry_df else 0.5
            })

    data_summary = {
        'farmer_count': len(farmer_df),
        'village_count': len(village_df),
        'industry_count': len(industry_df),
        'avg_fusion_index': f"{float(avg_fusion):.2%}" if isinstance(avg_fusion, (int, float, np.floating)) else str(avg_fusion),
        'avg_income': f"{float(avg_income):,.0f}元" if isinstance(avg_income, (int, float, np.floating)) else str(avg_income),
        'clustering_result': {
            'cluster_count': len(cluster_details),
            'clusters': cluster_details
        },
        'fusion_analysis': {
            'high_fusion': int(high_fusion),
            'medium_fusion': int(medium_fusion),
            'low_fusion': int(low_fusion),
            'avg_fusion': float(avg_fusion) if isinstance(avg_fusion, (int, float, np.floating)) else 0.5,
            'high_ratio': high_fusion/len(farmer_df)*100,
            'medium_ratio': medium_fusion/len(farmer_df)*100,
            'low_ratio': low_fusion/len(farmer_df)*100
        },
        'industry_analysis': {
            'total_revenue': float(industry_df['revenue'].sum()) if 'revenue' in industry_df else 0,
            'total_profit': float(industry_df['profit'].sum()) if 'profit' in industry_df else 0,
            'total_employment': int(industry_df['employment'].sum()) if 'employment' in industry_df else 0,
            'top_industries': top_industries
        },
        'risk_assessment': {
            'level': risks['overall']['level'],
            'score': f"{float(risks['overall']['score']):.2%}",
            'market_risk': float(risks['market_risk']['score']) if 'market_risk' in risks else 0.3,
            'production_risk': float(risks['production_risk']['score']) if 'production_risk' in risks else 0.3
        }
    }

    # 自动调用AI分析
    ai_analysis = None
    try:
        from src.utils.deepseek_analyzer import DeepSeekAnalyzer
        # 从环境变量或配置获取API Key
        api_key = os.environ.get("DEEPSEEK_API_KEY", "sk-044e4f1e7082434e9bc5d12df773397f")
        if api_key:
            ai_analyzer = DeepSeekAnalyzer(api_key=api_key)
            if ai_analyzer.is_available():
                with st.spinner("🤖 AI正在深入分析数据，请稍候..."):
                    ai_analysis = ai_analyzer.analyze_data(data_summary)
    except Exception as e:
        ai_analysis = None

    if ai_analysis and not ai_analysis.startswith("错误") and not ai_analysis.startswith("API"):
        # 使用容器显示AI分析结果
        st.markdown("---")
        st.markdown(ai_analysis)
    else:
        # 如果AI不可用，显示基于规则的分析
        st.markdown("### 📊 数据分析结论")
        st.markdown(f"""
        **现状诊断：**
        - 区域农户平均融合指数为 {float(avg_fusion):.2%}，处于{'较高' if avg_fusion > 0.5 else '中等' if avg_fusion > 0.3 else '较低'}水平
        - 高融合度农户占比 {high_fusion/len(farmer_df)*100:.1f}%，仍有提升空间
        - 综合风险等级为{level_colors.get(risks['overall']['level'], '中等')}，需关注风险防控

        **趋势判断：**
        - 产业融合发展趋势良好，一二三产业融合程度逐步加深
        - 新型经营主体培育效果显现，组织化程度提升
        - 农户收入水平稳步增长，城乡差距逐步缩小

        **对策建议：**
        1. 继续深化产业融合发展，延伸产业链条
        2. 加强新型经营主体培育，提升组织化程度
        3. 完善风险防控机制，保障产业稳定发展
        4. 优化政策支持体系，激发发展活力
        """)
        if ai_analysis:
            st.warning(f"AI分析遇到问题: {ai_analysis[:100]}...")

    # ===== 第七部分：综合建议 =====
    st.markdown("## 💡 第七部分：综合建议")
    st.markdown("---")

    # 生成详细建议
    recommendations = []

    # 确保数值类型
    avg_fusion_num = float(avg_fusion) if isinstance(avg_fusion, (int, float, np.floating)) else 0.5
    risk_score_num = float(risks['overall']['score']) if isinstance(risks['overall']['score'], (int, float, np.floating)) else 0.3

    # 计算更多指标
    high_ratio = high_fusion / len(farmer_df) * 100 if len(farmer_df) > 0 else 0
    low_ratio = low_fusion / len(farmer_df) * 100 if len(farmer_df) > 0 else 0

    # 产业数据
    total_revenue = industry_df['revenue'].sum() if 'revenue' in industry_df else 0
    total_profit = industry_df['profit'].sum() if 'profit' in industry_df else 0
    profit_rate = total_profit / total_revenue * 100 if total_revenue > 0 else 0

    # ========== 建议1：产业融合发展 ==========
    if avg_fusion_num < 0.5:
        rec1 = {
            'target': '产业融合发展提升工程',
            'priority': '高',
            'background': f'当前区域平均融合指数为{avg_fusion_num:.1%}，低于0.5的基准线，需重点提升一二三产业融合深度。',
            'actions': [
                f'建设农产品加工园区：投资约2000万元，引进5-8家农产品加工企业，实现初级农产品就地转化增值',
                f'发展乡村旅游休闲产业：依托{len(village_df)}个行政村资源，打造3-5条精品乡村旅游线路',
                f'培育农村电商主体：建设县级电商服务中心1个、村级服务站{min(20, len(village_df)//3)}个，培训电商人才500人次',
                f'推进农业产业链延伸：发展"种养加销"一体化模式，建设冷链物流设施，降低产后损失率15%'
            ],
            'responsible': '农业农村局、商务局、文旅局',
            'timeline': '2024-2026年（分三期实施）',
            'budget': '总投资约5000万元',
            'expected_outcome': f'预计3年内融合指数提升至{(avg_fusion_num + 0.15):.1%}，带动农户增收{avg_income * 0.2:,.0f}元/户'
        }
    else:
        rec1 = {
            'target': '产业融合提质增效工程',
            'priority': '中',
            'background': f'区域融合指数已达{avg_fusion_num:.1%}，需进一步优化产业结构，提升发展质量。',
            'actions': [
                '推进农业科技创新：建设智慧农业示范基地，推广物联网、大数据技术应用',
                '培育区域公用品牌：打造1-2个具有地方特色的农产品品牌，提升市场竞争力',
                '发展农业社会化服务：培育农机作业、植保防疫等专业服务组织'
            ],
            'responsible': '农业农村局、科技局',
            'timeline': '2024-2025年',
            'budget': '总投资约2000万元',
            'expected_outcome': f'预计融合指数提升至{(avg_fusion_num + 0.08):.1%}，品牌溢价提升20%'
        }
    recommendations.append(rec1)

    # ========== 建议2：农户分类培育 ==========
    if low_fusion > 0:
        rec2 = {
            'target': f'低融合度农户帮扶计划（涉及{low_fusion}户，占比{low_ratio:.1f}%）',
            'priority': '高',
            'background': f'当前有{low_fusion}户农户融合指数低于0.4，占总数的{low_ratio:.1f}%，是发展的薄弱环节。',
            'actions': [
                f'开展职业技能培训：组织种养殖技术、电商运营、农机操作等培训，每户至少1人参加',
                f'引导加入新型经营主体：优先推荐加入合作社或与龙头企业对接，提供"订单农业"服务',
                f'实施金融帮扶：提供贴息贷款、农业保险补贴，降低发展门槛',
                f'建立"一对一"帮扶机制：安排农技人员定期上门指导，解决生产实际问题'
            ],
            'responsible': '农业农村局、人社局、金融机构',
            'timeline': '2024-2025年（分批实施）',
            'budget': '约800万元（含培训、补贴、贷款贴息）',
            'expected_outcome': f'预计2年内{low_fusion}户农户融合指数提升至0.4以上，户均增收{avg_income * 0.25:,.0f}元'
        }
    else:
        rec2 = {
            'target': '高素质农民培育计划',
            'priority': '中',
            'background': '区域农户整体发展水平较好，需进一步提升人力资源质量。',
            'actions': [
                '培育新型职业农民：开展经营管理、市场营销等专业培训',
                '引进农业技术人才：制定人才引进激励政策',
                '搭建农民创业平台：提供创业指导、资金支持等服务'
            ],
            'responsible': '人社局、农业农村局',
            'timeline': '持续实施',
            'budget': '约300万元/年',
            'expected_outcome': '培育新型职业农民200人以上'
        }
    recommendations.append(rec2)

    # ========== 建议3：新型经营主体培育 ==========
    rec3 = {
        'target': '新型经营主体培育壮大工程',
        'priority': '高',
        'background': f'区域内现有合作社、企业等经营主体，需进一步提升带动能力和服务水平。',
        'actions': [
            '规范提升农民合作社：开展示范社创建活动，培育省级示范社3-5家',
            '扶持家庭农场发展：落实用地、用电、信贷等优惠政策，新增家庭农场20家以上',
            '引进龙头企业：引进投资额1000万元以上的农产品加工企业2-3家',
            '发展农业产业化联合体：推动龙头企业与合作社、农户建立利益联结机制'
        ],
        'responsible': '农业农村局、市场监管局、投促局',
        'timeline': '2024-2026年',
        'budget': '约1500万元（含奖补资金、基础设施配套）',
        'expected_outcome': '新增带动农户500户以上，户均增收15%'
    }
    recommendations.append(rec3)

    # ========== 建议4：基础设施建设 ==========
    rec4 = {
        'target': '农村基础设施补短板工程',
        'priority': '中',
        'background': '产业融合发展需要完善的基础设施支撑，重点解决物流、信息化等短板。',
        'actions': [
            f'完善农村道路网络：新建和改造农村公路{len(village_df)*2}公里，实现行政村通硬化路',
            '建设冷链物流体系：建设县级冷链物流中心1个，产地预冷设施{min(10, len(village_df)//5)}个',
            '推进数字乡村建设：实现行政村5G网络全覆盖，建设益农信息社{min(30, len(village_df))}个',
            '改善农田水利设施：实施高标准农田建设，新增有效灌溉面积'
        ],
        'responsible': '交通局、农业农村局、工信局、水利局',
        'timeline': '2024-2027年',
        'budget': '约8000万元',
        'expected_outcome': '物流成本降低20%，农产品损耗率降低15%'
    }
    recommendations.append(rec4)

    # ========== 建议5：风险防控体系 ==========
    rec5 = {
        'target': '农业风险防控体系建设',
        'priority': '高' if risk_score_num > 0.4 else '中',
        'background': f'当前综合风险得分为{risk_score_num:.1%}，需建立健全风险预警和应对机制。',
        'actions': [
            '建立风险监测预警系统：整合气象、市场、病虫害等数据，实现风险早发现、早预警',
            '完善农业保险体系：扩大政策性农业保险覆盖面，开发特色农产品保险产品',
            '建立风险准备金制度：县财政每年安排专项资金，用于应对重大自然灾害和市场波动',
            '发展订单农业：引导农户与龙头企业签订购销合同，降低市场风险'
        ],
        'responsible': '农业农村局、应急管理局、银保监局',
        'timeline': '2024-2025年',
        'budget': '约500万元（含系统建设、保险补贴）',
        'expected_outcome': '风险损失降低30%，农户参保率达到80%以上'
    }
    recommendations.append(rec5)

    # ========== 建议6：政策保障措施 ==========
    rec6 = {
        'target': '产业融合发展政策保障',
        'priority': '中',
        'background': '完善政策支持体系，为产业融合发展提供制度保障。',
        'actions': [
            '优化用地政策：保障设施农业用地需求，探索集体经营性建设用地入市',
            '加大财政投入：县财政每年安排产业融合发展专项资金不少于1000万元',
            '创新金融服务：开发"融合贷"等专属金融产品，降低融资成本',
            '强化科技支撑：与高校院所合作，建立专家服务团，提供技术支持'
        ],
        'responsible': '自然资源局、财政局、金融办、科技局',
        'timeline': '2024年起持续实施',
        'budget': '每年财政投入不低于1000万元',
        'expected_outcome': '政策覆盖率达到100%，农户满意度90%以上'
    }
    recommendations.append(rec6)

    # 显示建议
    for i, rec in enumerate(recommendations, 1):
        priority_color = "🔴" if rec['priority'] == '高' else "🟡" if rec['priority'] == '中' else "🟢"
        with st.expander(f"建议{i}：{priority_color} {rec['target']} (优先级: {rec['priority']})", expanded=(i<=3)):
            st.markdown(f"**📊 背景分析**：{rec['background']}")
            st.markdown("**📋 具体措施**：")
            for j, action in enumerate(rec['actions'], 1):
                st.markdown(f"  {j}. {action}")
            st.markdown(f"**🏢 责任单位**：{rec['responsible']}")
            st.markdown(f"**📅 实施周期**：{rec['timeline']}")
            st.markdown(f"**💰 资金预算**：{rec['budget']}")
            st.markdown(f"**📈 预期效果**：{rec['expected_outcome']}")

    # 显示建议汇总表
    st.markdown("### 📊 建议汇总表")
    summary_df = pd.DataFrame([
        {
            '序号': i,
            '建议名称': rec['target'],
            '优先级': rec['priority'],
            '实施周期': rec['timeline'],
            '资金预算': rec['budget']
        }
        for i, rec in enumerate(recommendations, 1)
    ])
    st.dataframe(summary_df, use_container_width=True)

    # ===== 导出报告 =====
    st.markdown("## 📥 导出报告")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # PDF报告下载
        if st.button("📄 生成PDF完整报告", type="primary", use_container_width=True):
            with st.spinner("正在生成PDF报告..."):
                try:
                    from src.utils.pdf_report_generator import PDFReportGenerator
                    pdf_gen = PDFReportGenerator()

                    # 准备报告数据 - 使用带有聚类结果的farmer_cluster_result
                    report_input = {
                        'farmers': farmer_cluster_result if 'cluster_name' in farmer_cluster_result.columns else farmer_df,
                        'villages': village_df,
                        'industries': industry_df,
                        'risk_assessment': risks,
                        'recommendations': recommendations
                    }

                    pdf_bytes = pdf_gen.create_report(report_input, ai_analysis)

                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        "📥 下载PDF报告",
                        pdf_bytes,
                        f"城乡融合分析报告_{timestamp}.pdf",
                        "application/pdf",
                        use_container_width=True
                    )
                    st.success("PDF报告生成成功！")
                except Exception as e:
                    import traceback
                    st.error(f"PDF生成失败: {str(e)}")
                    with st.expander("查看详细错误"):
                        st.code(traceback.format_exc())

    with col2:
        # JSON报告
        report_data = {
            'report_info': {
                'title': '城乡产业融合发展分析报告',
                'generated_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'region': '示范县'
            },
            'data_overview': {
                'farmer_count': len(farmer_df),
                'village_count': len(village_df),
                'industry_count': len(industry_df),
                'avg_fusion_index': float(avg_fusion) if isinstance(avg_fusion, (int, float, np.floating)) else 0.5,
                'avg_income': float(avg_income) if isinstance(avg_income, (int, float, np.floating)) else 25000
            },
            'clustering_analysis': cluster_summary,
            'fusion_analysis': {
                'high_fusion': high_fusion,
                'medium_fusion': medium_fusion,
                'low_fusion': low_fusion,
                'fusion_dimensions': float(avg_fusion) if isinstance(avg_fusion, (int, float, np.floating)) else 0.5
            },
            'industry_analysis': {
                'total_revenue': float(industry_df['revenue'].sum()) if 'revenue' in industry_df else 0,
                'total_profit': float(industry_df['profit'].sum()) if 'profit' in industry_df else 0,
                'total_employment': int(industry_df['employment'].sum()) if 'employment' in industry_df else 0
            },
            'risk_assessment': {
                'level': risks['overall']['level'],
                'score': float(risks['overall']['score']) if isinstance(risks['overall']['score'], (int, float, np.floating)) else 0.3
            },
            'recommendations': recommendations,
            'ai_analysis': ai_analysis
        }

        report_json = json.dumps(report_data, ensure_ascii=False, indent=2, default=str)
        st.download_button(
            "📊 下载JSON数据报告",
            report_json,
            f"城乡融合分析报告_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json",
            use_container_width=True
        )


if __name__ == '__main__':
    main()
