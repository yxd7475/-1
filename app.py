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

    # DeepSeek API配置
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI智能分析")
    deepseek_key = st.sidebar.text_input("DeepSeek API Key", type="password", help="输入DeepSeek API密钥以启用AI智能分析")
    enable_ai = st.sidebar.checkbox("启用AI智能分析", value=bool(deepseek_key))

    # 初始化AI分析器
    from src.utils.deepseek_analyzer import DeepSeekAnalyzer
    ai_analyzer = DeepSeekAnalyzer(api_key=deepseek_key) if deepseek_key else None

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
        avg_fusion = {
            '生产融合': farmer_df['f_production'].mean() if 'f_production' in farmer_df else 0.5,
            '供应融合': farmer_df['f_supply'].mean() if 'f_supply' in farmer_df else 0.5,
            '市场融合': farmer_df['f_market'].mean() if 'f_market' in farmer_df else 0.5,
            '服务融合': farmer_df['f_service'].mean() if 'f_service' in farmer_df else 0.5,
            '价值融合': farmer_df['f_value'].mean() if 'f_value' in farmer_df else 0.5
        }
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(avg_fusion.values()),
            theta=list(avg_fusion.keys()),
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

    # 准备数据摘要
    data_summary = {
        'farmer_count': len(farmer_df),
        'village_count': len(village_df),
        'industry_count': len(industry_df),
        'avg_fusion_index': f"{float(avg_fusion):.2%}" if isinstance(avg_fusion, (int, float, np.floating)) else str(avg_fusion),
        'avg_income': f"{float(avg_income):,.0f}元" if isinstance(avg_income, (int, float, np.floating)) else str(avg_income),
        'clustering_result': {
            'cluster_count': len(cluster_summary),
            'clusters': cluster_summary
        },
        'fusion_analysis': {
            'high_fusion': high_fusion,
            'medium_fusion': medium_fusion,
            'low_fusion': low_fusion,
            'avg_fusion': float(avg_fusion) if isinstance(avg_fusion, (int, float, np.floating)) else 0.5
        },
        'industry_analysis': {
            'total_revenue': float(industry_df['revenue'].sum()) if 'revenue' in industry_df else 0,
            'total_profit': float(industry_df['profit'].sum()) if 'profit' in industry_df else 0
        },
        'risk_assessment': {
            'level': risks['overall']['level'],
            'score': f"{float(risks['overall']['score']):.2%}"
        }
    }

    if enable_ai and ai_analyzer and ai_analyzer.is_available():
        if st.button("🤖 生成AI智能分析报告", type="primary"):
            with st.spinner("AI正在分析数据..."):
                ai_analysis = ai_analyzer.analyze_data(data_summary)

            st.markdown("### AI分析结果")
            st.markdown(ai_analysis)
    else:
        st.info("💡 请在左侧边栏输入DeepSeek API密钥以启用AI智能分析功能")

    # ===== 第七部分：综合建议 =====
    st.markdown("## 💡 第七部分：综合建议")
    st.markdown("---")

    # 生成建议
    recommendations = []

    # 确保数值类型
    avg_fusion_num = float(avg_fusion) if isinstance(avg_fusion, (int, float, np.floating)) else 0.5
    risk_score_num = float(risks['overall']['score']) if isinstance(risks['overall']['score'], (int, float, np.floating)) else 0.3

    # 基于融合指数
    if avg_fusion_num < 0.4:
        recommendations.append({
            'target': '整体融合发展',
            'priority': '高',
            'action': '实施产业融合发展工程，支持一二三产业深度融合',
            'expected_outcome': '提升融合指数15-20%'
        })

    # 基于聚类结果
    if low_fusion > len(farmer_df) * 0.3:
        recommendations.append({
            'target': f'{low_fusion}户低融合度农户',
            'priority': '高',
            'action': '组织培育与技能培训，引导加入合作社或龙头企业',
            'expected_outcome': '提升组织化程度，增加收入20%'
        })

    # 基于风险评估
    if risk_score_num > 0.5:
        recommendations.append({
            'target': '风险防控',
            'priority': '高',
            'action': '建立风险预警机制，完善农业保险体系',
            'expected_outcome': '降低风险损失15%'
        })

    # 基于产业分析
    if 'fusion_degree' in industry_df and industry_df['fusion_degree'].mean() < 0.5:
        recommendations.append({
            'target': '产业融合提升',
            'priority': '中',
            'action': '发展农产品精深加工，延伸产业链条',
            'expected_outcome': '提升产业附加值25%'
        })

    # 显示建议
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"建议{i}: 🎯 {rec['target']} (优先级: {rec['priority']})", expanded=True):
            st.write(f"**行动措施**: {rec['action']}")
            st.write(f"**预期效果**: {rec['expected_outcome']}")

    if not recommendations:
        st.success("当前区域发展态势良好，建议继续保持现有发展策略。")

    # ===== 导出报告 =====
    st.markdown("## 📥 导出报告")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
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
                'avg_fusion_index': avg_fusion,
                'avg_income': avg_income
            },
            'clustering_analysis': cluster_summary,
            'fusion_analysis': {
                'high_fusion': high_fusion,
                'medium_fusion': medium_fusion,
                'low_fusion': low_fusion,
                'fusion_dimensions': avg_fusion
            },
            'industry_analysis': {
                'total_revenue': industry_df['revenue'].sum() if 'revenue' in industry_df else 0,
                'total_profit': industry_df['profit'].sum() if 'profit' in industry_df else 0,
                'total_employment': industry_df['employment'].sum() if 'employment' in industry_df else 0
            },
            'risk_assessment': risks,
            'recommendations': recommendations
        }

        report_json = json.dumps(report_data, ensure_ascii=False, indent=2, default=str)
        st.download_button(
            "📄 下载JSON报告",
            report_json,
            f"城乡融合分析报告_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )

    with col2:
        # CSV数据导出
        csv_data = farmer_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📊 下载农户数据CSV",
            csv_data,
            f"农户数据_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

    with col3:
        # 统计摘要导出
        # 确保数值类型正确
        avg_fusion_val = float(avg_fusion) if isinstance(avg_fusion, (int, float, np.floating)) else 0.5
        avg_income_val = float(avg_income) if isinstance(avg_income, (int, float, np.floating)) else 25000
        risks_score_val = float(risks['overall']['score']) if isinstance(risks['overall']['score'], (int, float, np.floating)) else 0.3

        summary_text = f"""
城乡产业融合发展分析报告
========================
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
区域: 示范县

一、数据概览
- 农户总数: {len(farmer_df):,}
- 村庄总数: {len(village_df):,}
- 产业记录: {len(industry_df):,}
- 平均融合指数: {avg_fusion_val:.2%}
- 平均农户收入: {avg_income_val:,.0f}元

二、融合指数分析
- 高融合度农户: {high_fusion}户 ({high_fusion/len(farmer_df)*100:.1f}%)
- 中融合度农户: {medium_fusion}户 ({medium_fusion/len(farmer_df)*100:.1f}%)
- 低融合度农户: {low_fusion}户 ({low_fusion/len(farmer_df)*100:.1f}%)

三、风险评估
- 综合风险等级: {level_colors.get(risks['overall']['level'], risks['overall']['level'])}
- 综合风险得分: {risks_score_val:.2%}

四、主要建议
"""
        for i, rec in enumerate(recommendations, 1):
            summary_text += f"\n{i}. {rec['target']}: {rec['action']}"

        st.download_button(
            "📝 下载报告摘要",
            summary_text,
            f"报告摘要_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
            "text/plain"
        )


if __name__ == '__main__':
    main()
