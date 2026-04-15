# -*- coding: utf-8 -*-
"""
KMeans聚类动态演示模块 - 高性能版本
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import time


def run_kmeans_demo(X, n_clusters=4, max_iterations=10, speed=0.8, blue_colors=None):
    """
    运行KMeans动态演示 - 高性能版本

    Parameters:
    -----------
    X : ndarray
        标准化后的特征矩阵
    n_clusters : int
        聚类数量
    max_iterations : int
        最大迭代次数
    speed : float
        动画速度（秒）
    blue_colors : list
        颜色列表

    Returns:
    --------
    tuple
        (labels, iteration_history, farmer_df)
    """
    if blue_colors is None:
        blue_colors = ['#1f77b4', '#4299e1', '#63b3ed', '#90cdf4', '#bee3f8',
                       '#3182ce', '#2b6cb0', '#2c5282', '#2a4365', '#1A365D']

    n_samples = len(X)
    X_2d = X[:, :2]  # 用于2D可视化

    # 创建占位符
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    progress_bar = st.progress(0)

    np.random.seed(42)

    # ====== 步骤1: 随机初始化聚类中心 ======
    center_indices = np.random.choice(n_samples, n_clusters, replace=False)
    centers = X[center_indices].copy()

    # 初始分配（向量化计算）
    distances = np.sqrt(((X[:, np.newaxis, :] - centers) ** 2).sum(axis=2))
    labels = np.argmin(distances, axis=1)

    # 记录历史
    iteration_history = []

    # ====== 步骤2: 迭代过程 ======
    for iteration in range(max_iterations):
        # 更新聚类中心
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                new_centers[k] = X[mask].mean(axis=0)
            else:
                new_centers[k] = centers[k]

        # 计算中心移动距离
        center_shift = np.linalg.norm(new_centers - centers)
        centers = new_centers

        # 重新分配标签（向量化计算）
        distances = np.sqrt(((X[:, np.newaxis, :] - centers) ** 2).sum(axis=2))
        new_labels = np.argmin(distances, axis=1)

        # 统计变化
        changed_count = np.sum(new_labels != labels)
        labels = new_labels

        # 记录历史
        iteration_history.append({
            'iteration': iteration + 1,
            'center_shift': float(center_shift),
            'changed_count': int(changed_count),
            'cluster_sizes': [int(np.sum(labels == k)) for k in range(n_clusters)]
        })

        # ====== 实时可视化 ======
        with chart_placeholder.container():
            fig = go.Figure()

            for k in range(n_clusters):
                mask = labels == k
                fig.add_trace(go.Scatter(
                    x=X_2d[mask, 0],
                    y=X_2d[mask, 1],
                    mode='markers',
                    marker=dict(
                        color=blue_colors[k],
                        size=10,
                        opacity=0.8,
                        line=dict(color='white', width=1)
                    ),
                    name=f'聚类{k+1} ({np.sum(mask)}个)',
                    showlegend=True
                ))

            fig.update_layout(
                title=dict(text=f"🔄 迭代 {iteration + 1}/{max_iterations}", font=dict(size=20)),
                xaxis_title="特征1",
                yaxis_title="特征2",
                height=450,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

        # 更新指标
        with metrics_placeholder.container():
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("迭代步数", f"{iteration + 1}/{max_iterations}")
            with col_m2:
                st.metric("样本变化", f"{changed_count}个", f"{changed_count/n_samples*100:.1f}%")
            with col_m3:
                st.metric("中心移动", f"{center_shift:.4f}")
            with col_m4:
                if center_shift < 0.001:
                    st.metric("状态", "✅ 收敛")
                else:
                    st.metric("状态", "🔄 进行中")

        progress_bar.progress((iteration + 1) / max_iterations)

        # 检查收敛
        if center_shift < 0.001:
            break

        time.sleep(speed)

    # 清理
    chart_placeholder.empty()
    metrics_placeholder.empty()
    progress_bar.empty()

    return labels, iteration_history


def show_iteration_curves(iteration_history, blue_colors):
    """显示迭代过程曲线"""
    history_df = pd.DataFrame(iteration_history)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df['iteration'],
            y=history_df['center_shift'],
            mode='lines+markers+text',
            name='中心移动距离',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10),
            text=[f'{x:.3f}' for x in history_df['center_shift']],
            textposition='top center'
        ))
        fig.update_layout(title="聚类中心移动距离", xaxis_title="迭代次数", yaxis_title="距离")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df['iteration'],
            y=history_df['changed_count'],
            mode='lines+markers+text',
            name='样本变化数',
            line=dict(color='#4299e1', width=3),
            marker=dict(size=10),
            text=[str(x) for x in history_df['changed_count']],
            textposition='top center'
        ))
        fig.update_layout(title="每步样本归属变化", xaxis_title="迭代次数", yaxis_title="变化数量")
        st.plotly_chart(fig, use_container_width=True)


def show_clustering_results(farmer_df, labels, n_clusters, blue_colors):
    """显示聚类结果"""
    farmer_df['cluster'] = labels

    # 为聚类命名
    cluster_names = {}
    for k in range(n_clusters):
        mask = farmer_df['cluster'] == k
        if 'fusion_index' in farmer_df.columns:
            avg_fusion = farmer_df.loc[mask, 'fusion_index'].mean()
        else:
            avg_fusion = 0.5

        if 'land_area' in farmer_df.columns:
            avg_land = farmer_df.loc[mask, 'land_area'].mean()
        else:
            avg_land = 10

        if avg_fusion > 0.6 and avg_land > 50:
            cluster_names[k] = '规模融合型农户'
        elif avg_fusion > 0.6:
            cluster_names[k] = '融合发展型农户'
        elif avg_fusion > 0.4:
            cluster_names[k] = '初级融合型农户'
        else:
            cluster_names[k] = '传统型农户'

    farmer_df['cluster_name'] = farmer_df['cluster'].map(cluster_names)

    # 显示最终结果
    st.markdown("### 📊 聚类分析结果")

    # 聚类分布
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("聚类分布")
        cluster_dist = farmer_df['cluster_name'].value_counts()
        fig = px.pie(values=cluster_dist.values, names=cluster_dist.index, title="农户聚类分布",
                     color_discrete_sequence=blue_colors)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("各聚类特征对比")
        feature_cols = ['land_area', 'capital', 'fusion_index', 'annual_income']
        existing_cols = [c for c in feature_cols if c in farmer_df.columns]
        if existing_cols:
            cluster_means = farmer_df.groupby('cluster_name')[existing_cols].mean()
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

    # 聚类详情
    st.subheader("聚类详情")
    for cluster_name in farmer_df['cluster_name'].unique():
        with st.expander(f"📌 {cluster_name}"):
            cluster_data = farmer_df[farmer_df['cluster_name'] == cluster_name]
            st.write(f"**数量**: {len(cluster_data)} 户")
            st.write(f"**占比**: {len(cluster_data)/len(farmer_df)*100:.1f}%")

            if 'annual_income' in cluster_data:
                st.write(f"**平均收入**: {cluster_data['annual_income'].mean():,.0f} 元")
            if 'fusion_index' in cluster_data:
                st.write(f"**平均融合指数**: {cluster_data['fusion_index'].mean():.2%}")

    # 散点图
    st.subheader("聚类可视化")
    fig = px.scatter(
        farmer_df,
        x='dist_town',
        y='annual_income',
        color='cluster_name',
        size='land_area',
        hover_data=['farmer_id', 'state_name'],
        title="农户聚类分布图（距离-收入）",
        color_discrete_sequence=blue_colors
    )
    st.plotly_chart(fig, use_container_width=True)

    return farmer_df
