# -*- coding: utf-8 -*-
"""
PDF报告生成模块 - 使用fpdf2
生成包含图表和分析的完整PDF报告
"""
import os
import io
import base64
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from fpdf import FPDF

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class PDF(FPDF):
    """自定义PDF类，支持中文"""

    def __init__(self):
        super().__init__()
        # 设置更小的边距
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)
        try:
            self.add_font('SimHei', '', 'C:/Windows/Fonts/simhei.ttf')
        except:
            pass

    def footer(self):
        self.set_y(-15)
        self.set_font('SimHei', '', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'第 {self.page_no()} 页', 0, 0, 'C')


class PDFReportGenerator:
    """PDF报告生成器"""

    def __init__(self):
        self.blue_colors = ['#1f77b4', '#4299e1', '#63b3ed', '#90cdf4', '#bee3f8',
                           '#3182ce', '#2b6cb0', '#2c5282', '#2a4365', '#1A365D']

    def create_report(self, data, ai_analysis=None, output_path=None):
        """生成完整的PDF报告"""

        farmer_df = data.get('farmers', pd.DataFrame())
        village_df = data.get('villages', pd.DataFrame())
        industry_df = data.get('industries', pd.DataFrame())
        risks = data.get('risk_assessment', {})
        recommendations = data.get('recommendations', [])

        # 计算统计数据
        avg_fusion = farmer_df['fusion_index'].mean() if 'fusion_index' in farmer_df.columns else 0.5
        avg_income = farmer_df['annual_income'].mean() if 'annual_income' in farmer_df.columns else 25000

        high_fusion = len(farmer_df[farmer_df['fusion_index'] > 0.6]) if 'fusion_index' in farmer_df.columns else 0
        medium_fusion = len(farmer_df[(farmer_df['fusion_index'] >= 0.4) & (farmer_df['fusion_index'] <= 0.6)]) if 'fusion_index' in farmer_df.columns else 0
        low_fusion = len(farmer_df[farmer_df['fusion_index'] < 0.4]) if 'fusion_index' in farmer_df.columns else 0

        # 创建PDF
        pdf = PDF()
        pdf.add_page()

        # ===== 封面 =====
        pdf.set_font('SimHei', '', 28)
        pdf.ln(50)
        pdf.cell(0, 20, '城乡融合分析报告', 0, 1, 'C')
        pdf.ln(20)
        pdf.set_font('SimHei', '', 14)
        pdf.cell(0, 10, f'生成日期: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
        pdf.cell(0, 10, '分析区域: 示范县', 0, 1, 'C')
        pdf.ln(30)
        pdf.set_font('SimHei', '', 12)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 8, '城乡产业融合智能决策系统', 0, 1, 'C')
        pdf.set_text_color(0, 0, 0)

        # ===== 第一部分：数据概览 =====
        pdf.add_page()
        pdf.set_font('SimHei', '', 18)
        pdf.set_text_color(31, 119, 180)
        pdf.cell(0, 15, '一、数据概览', 0, 1, 'L')
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('SimHei', '', 12)

        overview_items = [
            f'农户总数: {len(farmer_df):,} 户',
            f'村庄总数: {len(village_df):,} 个',
            f'产业记录: {len(industry_df):,} 条',
            f'平均融合指数: {avg_fusion:.2%}',
            f'户均年收入: {avg_income:,.0f} 元'
        ]
        for item in overview_items:
            pdf.cell(0, 10, f'  {item}', 0, 1, 'L')

        pdf.ln(10)

        # 村庄概览图
        if len(village_df) > 0:
            village_img = self._create_village_overview_chart(village_df)
            if village_img:
                pdf.image(village_img, x=20, w=170)
                pdf.ln(5)

        # ===== 第二部分：KMeans聚类分析 =====
        pdf.set_font('SimHei', '', 18)
        pdf.set_text_color(31, 119, 180)
        pdf.cell(0, 15, '二、KMeans聚类分析', 0, 1, 'L')
        pdf.set_text_color(0, 0, 0)

        # 聚类分布图
        if 'cluster_name' in farmer_df.columns:
            cluster_img = self._create_cluster_pie_chart(farmer_df)
            if cluster_img:
                pdf.image(cluster_img, x=30, w=150)
                pdf.ln(5)

        # 聚类统计表
        if 'cluster_name' in farmer_df.columns:
            pdf.set_font('SimHei', '', 12)
            pdf.cell(0, 10, '聚类统计:', 0, 1, 'L')
            cluster_stats = farmer_df.groupby('cluster_name').agg({
                'farmer_id': 'count',
                'fusion_index': 'mean',
                'annual_income': 'mean'
            }).round(2)

            for idx, row in cluster_stats.iterrows():
                cluster_name = str(idx)[:15]  # 限制长度
                pdf.cell(0, 8, f'  {cluster_name}: {int(row["farmer_id"])} 户, 融合指数: {row["fusion_index"]:.2%}', 0, 1, 'L')

        # 聚类特征对比图
        if 'cluster_name' in farmer_df.columns:
            cluster_feature_img = self._create_cluster_feature_chart(farmer_df)
            if cluster_feature_img:
                pdf.add_page()
                pdf.set_font('SimHei', '', 14)
                pdf.cell(0, 10, '聚类特征对比:', 0, 1, 'L')
                pdf.image(cluster_feature_img, x=15, w=180)

        # 农户分布散点图
        scatter_img = self._create_farmer_scatter_chart(farmer_df)
        if scatter_img:
            pdf.add_page()
            pdf.set_font('SimHei', '', 14)
            pdf.cell(0, 10, '农户分布图:', 0, 1, 'L')
            pdf.image(scatter_img, x=15, w=180)

        # ===== 第三部分：融合指数分析 =====
        pdf.add_page()
        pdf.set_font('SimHei', '', 18)
        pdf.set_text_color(31, 119, 180)
        pdf.cell(0, 15, '三、融合指数分析', 0, 1, 'L')
        pdf.set_text_color(0, 0, 0)

        # 五维融合指标
        pdf.set_font('SimHei', '', 12)
        fusion_dims = ['f_production', 'f_supply', 'f_market', 'f_service', 'f_value']
        fusion_names = ['生产融合', '供应融合', '市场融合', '服务融合', '价值融合']

        pdf.cell(0, 10, '五维融合指标:', 0, 1, 'L')
        for dim, name in zip(fusion_dims, fusion_names):
            if dim in farmer_df.columns:
                pdf.cell(0, 8, f'  {name}: {farmer_df[dim].mean():.2%}', 0, 1, 'L')

        pdf.ln(5)

        # 融合等级分布
        pdf.cell(0, 10, '融合等级分布:', 0, 1, 'L')
        pdf.cell(0, 8, f'  高融合度: {high_fusion} 户 ({high_fusion/len(farmer_df)*100:.1f}%)', 0, 1, 'L')
        pdf.cell(0, 8, f'  中融合度: {medium_fusion} 户 ({medium_fusion/len(farmer_df)*100:.1f}%)', 0, 1, 'L')
        pdf.cell(0, 8, f'  低融合度: {low_fusion} 户 ({low_fusion/len(farmer_df)*100:.1f}%)', 0, 1, 'L')

        pdf.ln(10)

        # 雷达图
        radar_img = self._create_radar_chart(farmer_df, fusion_dims, fusion_names)
        if radar_img:
            pdf.image(radar_img, x=30, w=150)

        # 融合指数分布图
        fusion_dist_img = self._create_fusion_dist_chart(farmer_df)
        if fusion_dist_img:
            pdf.add_page()
            pdf.set_font('SimHei', '', 14)
            pdf.cell(0, 10, '融合指数分布:', 0, 1, 'L')
            pdf.image(fusion_dist_img, x=20, w=170)

        # ===== 第四部分：产业分析 =====
        pdf.add_page()
        pdf.set_font('SimHei', '', 18)
        pdf.set_text_color(31, 119, 180)
        pdf.cell(0, 15, '四、产业分析', 0, 1, 'L')
        pdf.set_text_color(0, 0, 0)

        if 'industry_type' in industry_df.columns:
            # 产业类型分布图
            industry_img = self._create_industry_chart(industry_df)
            if industry_img:
                pdf.image(industry_img, x=20, w=170)
                pdf.ln(5)

            # 产业统计
            pdf.set_font('SimHei', '', 12)
            pdf.cell(0, 10, '主要产业类型:', 0, 1, 'L')

            industry_stats = industry_df.groupby('industry_type').agg({
                'revenue': 'sum',
                'profit': 'sum',
                'employment': 'sum'
            }).sort_values('revenue', ascending=False).head(10)

            for idx, row in industry_stats.iterrows():
                ind_name = str(idx)[:12]
                pdf.cell(0, 8, f'  {ind_name}: 营收 {row["revenue"]:,.0f}元, 就业 {int(row["employment"])}人', 0, 1, 'L')

            # 产业收入利润图
            industry_rev_img = self._create_industry_revenue_chart(industry_df)
            if industry_rev_img:
                pdf.add_page()
                pdf.set_font('SimHei', '', 14)
                pdf.cell(0, 10, '产业收入与利润:', 0, 1, 'L')
                pdf.image(industry_rev_img, x=15, w=180)

            # 产业融合度分布图
            if 'fusion_degree' in industry_df.columns:
                fusion_deg_img = self._create_industry_fusion_chart(industry_df)
                if fusion_deg_img:
                    pdf.add_page()
                    pdf.set_font('SimHei', '', 14)
                    pdf.cell(0, 10, '产业融合度分布:', 0, 1, 'L')
                    pdf.image(fusion_deg_img, x=20, w=170)

        # ===== 第五部分：风险评估 =====
        pdf.add_page()
        pdf.set_font('SimHei', '', 18)
        pdf.set_text_color(31, 119, 180)
        pdf.cell(0, 15, '五、风险评估', 0, 1, 'L')
        pdf.set_text_color(0, 0, 0)

        # 风险得分图
        if risks:
            risk_img = self._create_risk_chart(risks)
            if risk_img:
                pdf.image(risk_img, x=30, w=150)
                pdf.ln(5)

        # 风险详情
        pdf.set_font('SimHei', '', 12)
        if risks:
            risk_items = [
                ('市场风险', risks.get('market_risk', {})),
                ('生产风险', risks.get('production_risk', {})),
                ('政策风险', risks.get('policy_risk', {})),
                ('金融风险', risks.get('financial_risk', {}))
            ]
            for name, info in risk_items:
                score = info.get('score', 0)
                if score > 0.6:
                    level = '高风险'
                elif score > 0.3:
                    level = '中等风险'
                else:
                    level = '低风险'
                pdf.cell(0, 10, f'{name}: {score:.2%} ({level})', 0, 1, 'L')

        # ===== 第六部分：AI智能分析 =====
        pdf.add_page()
        pdf.set_font('SimHei', '', 18)
        pdf.set_text_color(31, 119, 180)
        pdf.cell(0, 15, '六、AI智能分析', 0, 1, 'L')
        pdf.set_text_color(0, 0, 0)

        if ai_analysis and not ai_analysis.startswith("错误") and not ai_analysis.startswith("API"):
            # 解析并格式化AI分析内容
            lines = ai_analysis.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    pdf.ln(3)
                    continue

                # 处理标题 (## 开头)
                if line.startswith('## '):
                    pdf.ln(5)
                    pdf.set_font('SimHei', '', 14)
                    pdf.set_text_color(31, 119, 180)
                    title = line.replace('## ', '').replace('#', '')
                    pdf.multi_cell(0, 8, title)
                    pdf.set_text_color(0, 0, 0)
                    pdf.set_font('SimHei', '', 11)
                elif line.startswith('### '):
                    pdf.ln(3)
                    pdf.set_font('SimHei', '', 12)
                    title = line.replace('### ', '').replace('#', '')
                    pdf.multi_cell(0, 7, title)
                    pdf.set_font('SimHei', '', 11)
                elif line.startswith('# '):
                    pdf.ln(5)
                    pdf.set_font('SimHei', '', 16)
                    pdf.set_text_color(31, 119, 180)
                    title = line.replace('# ', '').replace('#', '')
                    pdf.multi_cell(0, 10, title)
                    pdf.set_text_color(0, 0, 0)
                    pdf.set_font('SimHei', '', 11)
                elif line.startswith('---'):
                    pdf.ln(3)
                    pdf.set_draw_color(200, 200, 200)
                    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                    pdf.ln(3)
                elif line.startswith('- ') or line.startswith('* '):
                    # 列表项
                    item = line[2:]
                    # 移除markdown加粗标记
                    item = item.replace('**', '')
                    pdf.set_font('SimHei', '', 11)
                    pdf.multi_cell(0, 6, f'  • {item}')
                elif line.startswith('  - ') or line.startswith('    - '):
                    # 子列表项
                    item = line.strip()[2:]
                    item = item.replace('**', '')
                    pdf.set_font('SimHei', '', 10)
                    pdf.multi_cell(0, 5, f'      - {item}')
                    pdf.set_font('SimHei', '', 11)
                else:
                    # 普通文本
                    # 移除markdown标记
                    clean_line = line.replace('**', '').replace('*', '').replace('`', '')
                    pdf.set_font('SimHei', '', 11)
                    try:
                        pdf.multi_cell(0, 6, clean_line)
                    except:
                        # 如果遇到编码问题，尝试简化
                        pdf.multi_cell(0, 6, clean_line.encode('utf-8', errors='ignore').decode('utf-8'))
        else:
            # 默认分析 - 使用中文
            pdf.set_font('SimHei', '', 11)
            pdf.cell(0, 10, '1. 现状分析:', 0, 1, 'L')
            pdf.cell(0, 8, f'   平均融合指数: {avg_fusion:.2%}', 0, 1, 'L')
            pdf.cell(0, 8, f'   高融合度农户占比: {high_fusion/len(farmer_df)*100:.1f}%', 0, 1, 'L')
            pdf.ln(3)

            pdf.cell(0, 10, '2. 趋势分析:', 0, 1, 'L')
            pdf.cell(0, 8, '   产业融合发展态势良好', 0, 1, 'L')
            pdf.cell(0, 8, '   组织化程度逐步提升', 0, 1, 'L')
            pdf.ln(3)

            pdf.cell(0, 10, '3. 发展建议:', 0, 1, 'L')
            pdf.cell(0, 8, '   继续深化产业融合', 0, 1, 'L')
            pdf.cell(0, 8, '   培育壮大新型经营主体', 0, 1, 'L')
            pdf.cell(0, 8, '   完善风险防控机制', 0, 1, 'L')

        # ===== 第七部分：综合建议 =====
        pdf.add_page()
        pdf.set_font('SimHei', '', 18)
        pdf.set_text_color(31, 119, 180)
        pdf.cell(0, 15, '七、综合建议', 0, 1, 'L')
        pdf.set_text_color(0, 0, 0)

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                # 建议标题
                pdf.set_font('SimHei', '', 14)
                pdf.set_text_color(31, 119, 180)
                priority = rec.get('priority', '中')
                priority_mark = '【高优先级】' if priority == '高' else '【中优先级】' if priority == '中' else '【低优先级】'
                target = str(rec.get("target", ""))[:40]
                pdf.multi_cell(0, 10, f'{i}. {priority_mark}{target}')
                pdf.set_text_color(0, 0, 0)
                pdf.set_font('SimHei', '', 11)

                # 背景分析
                background = rec.get('background', '')
                if background:
                    pdf.multi_cell(0, 7, f'背景分析：{background}')
                    pdf.ln(2)

                # 具体措施
                actions = rec.get('actions', [])
                if actions:
                    pdf.set_font('SimHei', '', 11)
                    pdf.cell(0, 7, '具体措施：', 0, 1, 'L')
                    pdf.set_font('SimHei', '', 10)
                    for j, action in enumerate(actions, 1):
                        # 处理长文本，自动换行
                        action_text = str(action)
                        pdf.multi_cell(0, 6, f'  {j}. {action_text}')
                    pdf.ln(2)

                # 责任单位
                responsible = rec.get('responsible', '')
                if responsible:
                    pdf.set_font('SimHei', '', 10)
                    pdf.multi_cell(0, 6, f'责任单位：{responsible}')

                # 实施周期
                timeline = rec.get('timeline', '')
                if timeline:
                    pdf.multi_cell(0, 6, f'实施周期：{timeline}')

                # 资金预算
                budget = rec.get('budget', '')
                if budget:
                    pdf.multi_cell(0, 6, f'资金预算：{budget}')

                # 预期效果
                outcome = rec.get('expected_outcome', '')
                if outcome:
                    pdf.set_font('SimHei', '', 10)
                    pdf.set_text_color(0, 100, 0)
                    pdf.multi_cell(0, 6, f'预期效果：{outcome}')
                    pdf.set_text_color(0, 0, 0)

                pdf.ln(5)

                # 每3条建议后换页
                if i % 3 == 0 and i < len(recommendations):
                    pdf.add_page()
        else:
            # 默认建议
            pdf.set_font('SimHei', '', 12)
            default_recs = [
                ('产业融合发展', '推动一二三产业深度融合，延伸产业链条，提升附加值'),
                ('新型经营主体培育', '支持农民合作社和家庭农场发展，提高组织化程度'),
                ('基础设施建设', '加强农村道路、冷链物流、信息化等基础设施建设'),
                ('科技支撑体系', '推广农业新技术应用，建设智慧农业示范基地'),
                ('政策保障机制', '完善产业扶持政策体系，优化发展环境')
            ]
            for i, (title, content) in enumerate(default_recs, 1):
                pdf.cell(0, 10, f'{i}. {title}', 0, 1, 'L')
                pdf.set_font('SimHei', '', 10)
                pdf.multi_cell(0, 6, f'   {content}')
                pdf.set_font('SimHei', '', 12)
                pdf.ln(3)

        # 建议汇总表
        if recommendations:
            pdf.add_page()
            pdf.set_font('SimHei', '', 14)
            pdf.set_text_color(31, 119, 180)
            pdf.cell(0, 12, '建议汇总表', 0, 1, 'L')
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('SimHei', '', 10)

            # 表头
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(10, 8, '序号', 1, 0, 'C', True)
            pdf.cell(70, 8, '建议名称', 1, 0, 'C', True)
            pdf.cell(20, 8, '优先级', 1, 0, 'C', True)
            pdf.cell(50, 8, '实施周期', 1, 0, 'C', True)
            pdf.cell(40, 8, '资金预算', 1, 1, 'C', True)

            # 表格内容
            for i, rec in enumerate(recommendations, 1):
                target = str(rec.get('target', ''))[:18]
                priority = rec.get('priority', '中')
                timeline = str(rec.get('timeline', ''))[:15]
                budget = str(rec.get('budget', ''))[:12]

                pdf.cell(10, 7, str(i), 1, 0, 'C')
                pdf.cell(70, 7, target, 1, 0, 'L')
                pdf.cell(20, 7, priority, 1, 0, 'C')
                pdf.cell(50, 7, timeline, 1, 0, 'C')
                pdf.cell(40, 7, budget, 1, 1, 'C')

        # 输出PDF
        if output_path:
            pdf.output(output_path)
            return output_path
        else:
            # 返回字节
            buffer = io.BytesIO()
            pdf.output(buffer)
            buffer.seek(0)
            return buffer.getvalue()

    def _create_cluster_pie_chart(self, farmer_df):
        """创建聚类饼图"""
        try:
            fig, ax = plt.subplots(figsize=(10, 7))
            cluster_dist = farmer_df['cluster_name'].value_counts()
            colors = self.blue_colors[:len(cluster_dist)]
            wedges, texts, autotexts = ax.pie(
                cluster_dist.values, labels=cluster_dist.index,
                autopct='%1.1f%%', colors=colors, startangle=90
            )
            ax.set_title('农户聚类分布', fontsize=14, fontweight='bold')

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plt.close()
            return buffer
        except:
            return None

    def _create_cluster_feature_chart(self, farmer_df):
        """创建聚类特征对比图"""
        try:
            feature_cols = ['land_area', 'capital', 'fusion_index', 'annual_income']
            existing_cols = [c for c in feature_cols if c in farmer_df.columns]
            if not existing_cols or 'cluster_name' not in farmer_df.columns:
                return None

            cluster_means = farmer_df.groupby('cluster_name')[existing_cols].mean()

            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(cluster_means.index))
            width = 0.2

            # 中文列名映射
            col_names = {
                'land_area': '土地面积',
                'capital': '资本',
                'fusion_index': '融合指数',
                'annual_income': '年收入'
            }

            for i, col in enumerate(existing_cols):
                ax.bar(x + i * width, cluster_means[col], width,
                      label=col_names.get(col, col), color=self.blue_colors[i % len(self.blue_colors)])

            ax.set_xticks(x + width * (len(existing_cols) - 1) / 2)
            ax.set_xticklabels(cluster_means.index, rotation=45, ha='right')
            ax.set_title('聚类特征对比', fontsize=14, fontweight='bold')
            ax.legend()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plt.close()
            return buffer
        except:
            return None

    def _create_radar_chart(self, df, dims, names):
        """创建雷达图"""
        try:
            values = [df[d].mean() if d in df.columns else 0.5 for d in dims]
            values += values[:1]

            angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
            ax.fill(angles, values, alpha=0.25, color='#1f77b4')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(names)
            ax.set_ylim(0, 1)
            ax.set_title('五维融合指数雷达图', fontsize=14, fontweight='bold')

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plt.close()
            return buffer
        except:
            return None

    def _create_fusion_dist_chart(self, farmer_df):
        """创建融合指数分布图"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(farmer_df['fusion_index'], bins=30, color='#1f77b4', edgecolor='white', alpha=0.8)
            ax.set_title('融合指数分布', fontsize=14, fontweight='bold')
            ax.set_xlabel('融合指数')
            ax.set_ylabel('农户数量')
            ax.axvline(farmer_df['fusion_index'].mean(), color='red', linestyle='--',
                      label=f'均值: {farmer_df["fusion_index"].mean():.2f}')
            ax.legend()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plt.close()
            return buffer
        except:
            return None

    def _create_industry_chart(self, industry_df):
        """创建产业分析图"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            type_counts = industry_df['industry_type'].value_counts()
            colors = self.blue_colors[:len(type_counts)]
            bars = ax.bar(range(len(type_counts)), type_counts.values, color=colors)
            ax.set_xticks(range(len(type_counts)))
            ax.set_xticklabels(type_counts.index, rotation=45, ha='right')
            ax.set_title('产业类型分布', fontsize=14, fontweight='bold')
            ax.set_ylabel('数量')

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plt.close()
            return buffer
        except:
            return None

    def _create_risk_chart(self, risks):
        """创建风险图"""
        try:
            risk_names = ['市场风险', '生产风险', '政策风险', '金融风险']
            risk_scores = [
                risks.get('market_risk', {}).get('score', 0),
                risks.get('production_risk', {}).get('score', 0),
                risks.get('policy_risk', {}).get('score', 0),
                risks.get('financial_risk', {}).get('score', 0)
            ]

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#e74c3c' if s > 0.6 else '#f39c12' if s > 0.3 else '#27ae60' for s in risk_scores]
            bars = ax.bar(risk_names, risk_scores, color=colors)
            ax.set_ylim(0, 1)
            ax.set_title('风险评估得分', fontsize=14, fontweight='bold')
            ax.set_ylabel('风险得分')

            for bar, score in zip(bars, risk_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{score:.1%}', ha='center', va='bottom', fontsize=11)

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plt.close()
            return buffer
        except:
            return None

    def _create_industry_revenue_chart(self, industry_df):
        """创建产业收入利润图"""
        try:
            if 'industry_type' not in industry_df.columns:
                return None

            industry_stats = industry_df.groupby('industry_type').agg({
                'revenue': 'sum',
                'profit': 'sum'
            }).sort_values('revenue', ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(industry_stats.index))
            width = 0.35

            ax.bar(x - width/2, industry_stats['revenue'], width, label='营收', color='#1f77b4')
            ax.bar(x + width/2, industry_stats['profit'], width, label='利润', color='#4299e1')

            ax.set_xticks(x)
            ax.set_xticklabels(industry_stats.index, rotation=45, ha='right')
            ax.set_title('主要产业营收与利润', fontsize=14, fontweight='bold')
            ax.legend()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plt.close()
            return buffer
        except:
            return None

    def _create_industry_fusion_chart(self, industry_df):
        """创建产业融合度分布图"""
        try:
            if 'fusion_degree' not in industry_df.columns:
                return None

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(industry_df['fusion_degree'], bins=20, color='#4299e1', edgecolor='white', alpha=0.8)
            ax.set_title('产业融合度分布', fontsize=14, fontweight='bold')
            ax.set_xlabel('融合度')
            ax.set_ylabel('数量')

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plt.close()
            return buffer
        except:
            return None

    def _create_village_overview_chart(self, village_df):
        """创建村庄概览图"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # 村庄人口分布
            if 'population' in village_df.columns:
                axes[0].hist(village_df['population'], bins=20, color='#1f77b4', edgecolor='white', alpha=0.8)
                axes[0].set_title('村庄人口分布', fontsize=12, fontweight='bold')
                axes[0].set_xlabel('人口数量')
                axes[0].set_ylabel('村庄数量')
            else:
                axes[0].text(0.5, 0.5, '暂无人口数据', ha='center', va='center')

            # 村庄收入分布
            if 'avg_income' in village_df.columns:
                axes[1].hist(village_df['avg_income'], bins=20, color='#4299e1', edgecolor='white', alpha=0.8)
                axes[1].set_title('村庄收入分布', fontsize=12, fontweight='bold')
                axes[1].set_xlabel('平均收入')
                axes[1].set_ylabel('村庄数量')
            else:
                axes[1].text(0.5, 0.5, '暂无收入数据', ha='center', va='center')

            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plt.close()
            return buffer
        except:
            return None

    def _create_farmer_scatter_chart(self, farmer_df):
        """创建农户分布散点图"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            if 'cluster_name' in farmer_df.columns:
                # 按聚类着色
                for cluster in farmer_df['cluster_name'].unique():
                    cluster_data = farmer_df[farmer_df['cluster_name'] == cluster]
                    x = cluster_data['dist_town'] if 'dist_town' in cluster_data else np.random.rand(len(cluster_data)) * 20
                    y = cluster_data['annual_income'] if 'annual_income' in cluster_data else np.random.rand(len(cluster_data)) * 50000
                    sizes = cluster_data['land_area'] if 'land_area' in cluster_data else 10
                    ax.scatter(x, y, s=sizes, alpha=0.6, label=cluster)

                ax.legend()
            else:
                x = farmer_df['dist_town'] if 'dist_town' in farmer_df else np.random.rand(len(farmer_df)) * 20
                y = farmer_df['annual_income'] if 'annual_income' in farmer_df else np.random.rand(len(farmer_df)) * 50000
                ax.scatter(x, y, alpha=0.6, color='#1f77b4')

            ax.set_title('农户分布图 (距离-收入)', fontsize=14, fontweight='bold')
            ax.set_xlabel('距镇中心距离 (公里)')
            ax.set_ylabel('年收入 (元)')

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plt.close()
            return buffer
        except:
            return None
