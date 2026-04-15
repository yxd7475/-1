# -*- coding: utf-8 -*-
"""
智能决策支持模块
基于聚类结果生成产业规划、项目选址、招商策略等决策建议
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import MODEL_CONFIG, DECISION_SCENARIOS, RISK_LEVELS


class DecisionSupportEngine:
    """智能决策支持引擎"""

    def __init__(self):
        self.config = MODEL_CONFIG
        self.scenarios = DECISION_SCENARIOS
        self.risk_levels = RISK_LEVELS

    def generate_industry_planning(self, df, region_name='示范县'):
        """
        生成产业规划建议

        Parameters:
        -----------
        df : pd.DataFrame
            区域产业数据
        region_name : str
            区域名称

        Returns:
        --------
        dict
            产业规划建议
        """
        planning = {
            'region': region_name,
            'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'diagnosis': self._diagnose_industry(df),
            'priority_industries': self._identify_priority_industries(df),
            'layout_suggestions': self._generate_layout_suggestions(df),
            'policy_recommendations': self._generate_policy_recommendations(df)
        }

        return planning

    def _diagnose_industry(self, df):
        """产业诊断"""
        diagnosis = {
            'total_revenue': df['revenue'].sum() if 'revenue' in df else 0,
            'total_profit': df['profit'].sum() if 'profit' in df else 0,
            'total_employment': df['employment'].sum() if 'employment' in df else 0,
            'avg_fusion_degree': df['fusion_degree'].mean() if 'fusion_degree' in df else 0,
            'industry_structure': {}
        }

        # 产业结构分析
        if 'industry_type' in df:
            for industry, group in df.groupby('industry_type'):
                diagnosis['industry_structure'][industry] = {
                    'count': len(group),
                    'revenue_share': group['revenue'].sum() / diagnosis['total_revenue'] if diagnosis['total_revenue'] > 0 else 0,
                    'avg_fusion': group['fusion_degree'].mean() if 'fusion_degree' in group else 0
                }

        # 问题识别
        diagnosis['issues'] = []
        if diagnosis['avg_fusion_degree'] < 0.4:
            diagnosis['issues'].append('产业融合程度偏低，产业链条较短')
        if diagnosis['total_profit'] / diagnosis['total_revenue'] < 0.1 if diagnosis['total_revenue'] > 0 else True:
            diagnosis['issues'].append('产业盈利能力较弱，需提升附加值')

        return diagnosis

    def _identify_priority_industries(self, df):
        """识别优先发展产业"""
        if 'industry_type' not in df:
            return []

        # 计算综合发展潜力得分
        industry_scores = df.groupby('industry_type').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'employment': 'sum',
            'fusion_degree': 'mean'
        }).reset_index()

        # 标准化
        for col in ['revenue', 'profit', 'employment', 'fusion_degree']:
            max_val = industry_scores[col].max()
            if max_val > 0:
                industry_scores[f'{col}_norm'] = industry_scores[col] / max_val

        # 综合得分
        industry_scores['potential_score'] = (
            industry_scores['revenue_norm'] * 0.25 +
            industry_scores['profit_norm'] * 0.25 +
            industry_scores['employment_norm'] * 0.2 +
            industry_scores['fusion_degree_norm'] * 0.3
        )

        # 排序
        priority = industry_scores.nlargest(5, 'potential_score')

        return priority.to_dict('records')

    def _generate_layout_suggestions(self, df):
        """生成布局建议"""
        suggestions = []

        if 'industry_type' in df:
            for industry in df['industry_type'].unique():
                industry_data = df[df['industry_type'] == industry]

                suggestion = {
                    'industry': industry,
                    'current_scale': industry_data['revenue'].sum(),
                    'development_stage': self._assess_development_stage(industry_data),
                    'suggested_actions': self._get_development_actions(industry, industry_data)
                }
                suggestions.append(suggestion)

        return suggestions

    def _assess_development_stage(self, industry_data):
        """评估发展阶段"""
        avg_fusion = industry_data['fusion_degree'].mean() if 'fusion_degree' in industry_data else 0
        avg_revenue = industry_data['revenue'].mean() if 'revenue' in industry_data else 0

        if avg_fusion < 0.3:
            return '起步阶段'
        elif avg_fusion < 0.6:
            return '发展阶段'
        else:
            return '成熟阶段'

    def _get_development_actions(self, industry, industry_data):
        """获取发展行动建议"""
        actions_map = {
            '起步阶段': [
                '加大政策扶持力度，培育市场主体',
                '引进龙头企业，带动产业发展',
                '加强基础设施建设，改善发展条件'
            ],
            '发展阶段': [
                '延伸产业链条，发展精深加工',
                '培育区域品牌，提升市场影响力',
                '发展产业联合体，提高组织化程度'
            ],
            '成熟阶段': [
                '推进产业升级，发展高端业态',
                '拓展新业态，发展休闲农业、农村电商',
                '输出技术和模式，带动周边发展'
            ]
        }

        stage = self._assess_development_stage(industry_data)
        return actions_map.get(stage, [])

    def _generate_policy_recommendations(self, df):
        """生成政策建议"""
        recommendations = []

        # 基于融合度差距
        if 'fusion_degree' in df:
            avg_fusion = df['fusion_degree'].mean()
            if avg_fusion < 0.4:
                recommendations.append({
                    'type': '产业融合',
                    'priority': '高',
                    'content': '实施产业融合发展工程，支持一二三产业融合',
                    'expected_effect': '提升融合指数15-20%'
                })

        # 基于就业
        if 'employment' in df:
            total_employment = df['employment'].sum()
            if total_employment < 1000:
                recommendations.append({
                    'type': '就业带动',
                    'priority': '中',
                    'content': '发展劳动密集型产业，增加就业岗位',
                    'expected_effect': '新增就业岗位200+'
                })

        return recommendations

    def generate_site_selection(self, df, project_type, requirements):
        """
        生成项目选址建议

        Parameters:
        -----------
        df : pd.DataFrame
            区域/村庄数据
        project_type : str
            项目类型
        requirements : dict
            项目需求

        Returns:
        --------
        dict
            选址建议
        """
        # 权重设置
        weights = {
            '农产品加工': {
                'raw_material': 0.3,
                'transportation': 0.25,
                'labor': 0.2,
                'market': 0.15,
                'policy': 0.1
            },
            '乡村旅游': {
                'resources': 0.3,
                'accessibility': 0.25,
                'scenery': 0.25,
                'infrastructure': 0.2
            },
            '农业服务': {
                'coverage': 0.3,
                'demand': 0.25,
                'transportation': 0.25,
                'policy': 0.2
            }
        }

        project_weights = weights.get(project_type, {})

        # 计算各候选地点得分
        scores = []
        for idx, row in df.iterrows():
            score = self._calculate_site_score(row, project_type, project_weights, requirements)
            scores.append({
                'location_id': row.get('village_id', idx),
                'location_name': row.get('village_name', f'地点{idx}'),
                'score': score['total'],
                'dimension_scores': score['dimensions'],
                'rank': 0
            })

        # 排序
        scores_df = pd.DataFrame(scores)
        scores_df = scores_df.sort_values('score', ascending=False)
        scores_df['rank'] = range(1, len(scores_df) + 1)

        return {
            'project_type': project_type,
            'requirements': requirements,
            'recommendations': scores_df.head(5).to_dict('records'),
            'selection_criteria': self._get_selection_criteria(project_type)
        }

    def _calculate_site_score(self, location, project_type, weights, requirements):
        """计算选址得分"""
        dimensions = {}

        # 原材料供应（针对加工类）
        if 'raw_material' in weights:
            crop_area = location.get('crop_area', 0)
            orchard_area = location.get('orchard_area', 0)
            dimensions['raw_material'] = min(1.0, (crop_area + orchard_area) / 5000)

        # 交通可达性
        if 'transportation' in weights:
            dist_town = location.get('dist_town', 20)
            dimensions['transportation'] = max(0, 1 - dist_town / 30)

        # 劳动力
        if 'labor' in weights:
            population = location.get('population', 0)
            dimensions['labor'] = min(1.0, population / 1000)

        # 市场
        if 'market' in weights:
            dimensions['market'] = location.get('fusion_index', 0.5)

        # 资源（针对旅游类）
        if 'resources' in weights:
            dimensions['resources'] = location.get('fusion_index', 0.5)

        # 可达性（针对旅游类）
        if 'accessibility' in weights:
            dist_town = location.get('dist_town', 20)
            dimensions['accessibility'] = max(0, 1 - dist_town / 30)

        # 计算总分
        total = sum(dimensions.get(k, 0) * v for k, v in weights.items())

        return {'total': total, 'dimensions': dimensions}

    def _get_selection_criteria(self, project_type):
        """获取选址标准"""
        criteria = {
            '农产品加工': [
                '距离原料产地较近，降低运输成本',
                '交通便利，便于产品外运',
                '劳动力资源充足',
                '配套设施完善'
            ],
            '乡村旅游': [
                '自然或人文资源丰富',
                '交通便利，可达性好',
                '环境优美，适合休闲',
                '基础设施完善'
            ],
            '农业服务': [
                '服务覆盖范围广',
                '需求集中，市场潜力大',
                '交通便利',
                '政策支持力度大'
            ]
        }
        return criteria.get(project_type, [])

    def generate_risk_assessment(self, df, assessment_type='comprehensive'):
        """
        生成风险评估

        Parameters:
        -----------
        df : pd.DataFrame
            数据
        assessment_type : str
            评估类型

        Returns:
        --------
        dict
            风险评估结果
        """
        risks = {
            'market_risk': self._assess_market_risk(df),
            'production_risk': self._assess_production_risk(df),
            'policy_risk': self._assess_policy_risk(df),
            'financial_risk': self._assess_financial_risk(df)
        }

        # 计算综合风险等级
        overall_score = np.mean([r['score'] for r in risks.values()])

        if overall_score >= 0.7:
            risk_level = 'red'
        elif overall_score >= 0.4:
            risk_level = 'yellow'
        else:
            risk_level = 'blue'

        risks['overall'] = {
            'level': risk_level,
            'score': overall_score,
            'description': self.risk_levels[risk_level]['desc']
        }

        return risks

    def _assess_market_risk(self, df):
        """评估市场风险"""
        score = 0.3  # 基础风险

        if 'revenue' in df:
            revenue_std = df['revenue'].std()
            revenue_mean = df['revenue'].mean()
            if revenue_mean > 0:
                cv = revenue_std / revenue_mean
                score += min(0.4, cv * 0.5)

        return {
            'score': min(1.0, score),
            'factors': ['价格波动', '市场需求变化', '竞争加剧'],
            'mitigation': ['发展订单农业', '拓展销售渠道', '建立品牌壁垒']
        }

    def _assess_production_risk(self, df):
        """评估生产风险"""
        score = 0.2

        if 'fusion_degree' in df:
            avg_fusion = df['fusion_degree'].mean()
            score += (1 - avg_fusion) * 0.5

        return {
            'score': min(1.0, score),
            'factors': ['自然灾害', '病虫害', '技术风险'],
            'mitigation': ['购买农业保险', '完善预警体系', '加强技术培训']
        }

    def _assess_policy_risk(self, df):
        """评估政策风险"""
        return {
            'score': 0.2,
            'factors': ['政策调整', '补贴变化', '土地政策'],
            'mitigation': ['关注政策动态', '多元发展规划', '合法合规经营']
        }

    def _assess_financial_risk(self, df):
        """评估金融风险"""
        score = 0.25

        if 'profit' in df and 'revenue' in df:
            profit_margin = df['profit'].sum() / df['revenue'].sum() if df['revenue'].sum() > 0 else 0
            if profit_margin < 0.1:
                score += 0.3
            elif profit_margin < 0.2:
                score += 0.15

        return {
            'score': min(1.0, score),
            'factors': ['资金周转', '融资成本', '收益不稳定'],
            'mitigation': ['优化资金管理', '拓展融资渠道', '发展订单农业']
        }

    def generate_investment_analysis(self, project_info, df):
        """
        生成投资分析

        Parameters:
        -----------
        project_info : dict
            项目信息
        df : pd.DataFrame
            相关数据

        Returns:
        --------
        dict
            投资分析报告
        """
        investment = project_info.get('investment', 1000000)
        expected_revenue = project_info.get('expected_revenue', 2000000)
        operating_cost = project_info.get('operating_cost', 800000)
        project_life = project_info.get('project_life', 10)

        # 计算财务指标
        annual_profit = expected_revenue - operating_cost
        roi = annual_profit / investment if investment > 0 else 0
        payback_period = investment / annual_profit if annual_profit > 0 else float('inf')
        npv = self._calculate_npv(investment, annual_profit, project_life, 0.08)
        irr = self._calculate_irr(investment, annual_profit, project_life)

        analysis = {
            'project_name': project_info.get('name', '项目'),
            'financial_indicators': {
                'total_investment': investment,
                'annual_revenue': expected_revenue,
                'annual_profit': annual_profit,
                'roi': f'{roi*100:.1f}%',
                'payback_period': f'{payback_period:.1f}年',
                'npv': f'{npv:.0f}万元',
                'irr': f'{irr*100:.1f}%'
            },
            'feasibility': self._assess_feasibility(roi, payback_period, npv, irr),
            'risk_factors': self._identify_risk_factors(project_info),
            'sensitivity_analysis': self._sensitivity_analysis(investment, annual_profit, project_life)
        }

        return analysis

    def _calculate_npv(self, investment, annual_profit, years, discount_rate):
        """计算净现值"""
        npv = -investment
        for t in range(1, years + 1):
            npv += annual_profit / (1 + discount_rate) ** t
        return npv / 10000  # 转换为万元

    def _calculate_irr(self, investment, annual_profit, years):
        """计算内部收益率"""
        from scipy.optimize import brentq

        def npv_func(r):
            npv = -investment
            for t in range(1, years + 1):
                npv += annual_profit / (1 + r) ** t
            return npv

        try:
            irr = brentq(npv_func, -0.99, 10)
        except:
            irr = 0

        return irr

    def _assess_feasibility(self, roi, payback_period, npv, irr):
        """评估可行性"""
        score = 0

        if roi > 0.15:
            score += 25
        elif roi > 0.1:
            score += 15

        if payback_period < 5:
            score += 25
        elif payback_period < 8:
            score += 15

        if npv > 0:
            score += 25
        else:
            score -= 10

        if irr > 0.12:
            score += 25
        elif irr > 0.08:
            score += 15

        if score >= 80:
            return {'level': '高度可行', 'score': score, 'recommendation': '建议立即启动'}
        elif score >= 60:
            return {'level': '基本可行', 'score': score, 'recommendation': '建议优化后启动'}
        else:
            return {'level': '需谨慎评估', 'score': score, 'recommendation': '建议重新规划'}

    def _identify_risk_factors(self, project_info):
        """识别风险因素"""
        return [
            {'factor': '市场风险', 'probability': '中', 'impact': '高'},
            {'factor': '技术风险', 'probability': '低', 'impact': '中'},
            {'factor': '政策风险', 'probability': '低', 'impact': '中'},
            {'factor': '管理风险', 'probability': '中', 'impact': '中'}
        ]

    def _sensitivity_analysis(self, investment, annual_profit, years):
        """敏感性分析"""
        base_npv = self._calculate_npv(investment, annual_profit, years, 0.08)

        scenarios = []
        for change in [-20, -10, 0, 10, 20]:
            adjusted_profit = annual_profit * (1 + change / 100)
            npv = self._calculate_npv(investment, adjusted_profit, years, 0.08)
            scenarios.append({
                'scenario': f'收益{change:+d}%',
                'npv': f'{npv:.0f}万元',
                'change': change
            })

        return scenarios


class ReportGenerator:
    """报告生成器"""

    def __init__(self):
        self.engine = DecisionSupportEngine()

    def generate_comprehensive_report(self, farmer_df, village_df, industry_df, region_name='示范县'):
        """
        生成综合报告

        Parameters:
        -----------
        farmer_df : pd.DataFrame
            农户数据
        village_df : pd.DataFrame
            村庄数据
        industry_df : pd.DataFrame
            产业数据
        region_name : str
            区域名称

        Returns:
        --------
        dict
            综合报告
        """
        report = {
            'report_info': {
                'title': f'{region_name}城乡产业融合发展分析报告',
                'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'region': region_name
            },
            'executive_summary': self._generate_executive_summary(farmer_df, village_df, industry_df),
            'data_overview': {
                'farmers': len(farmer_df),
                'villages': len(village_df),
                'industries': len(industry_df)
            },
            'clustering_analysis': self._analyze_clustering_results(farmer_df),
            'fusion_analysis': self._analyze_fusion_status(farmer_df, village_df),
            'industry_analysis': self.engine.generate_industry_planning(industry_df, region_name),
            'risk_assessment': self.engine.generate_risk_assessment(industry_df),
            'recommendations': self._generate_recommendations(farmer_df, village_df, industry_df)
        }

        return report

    def _generate_executive_summary(self, farmer_df, village_df, industry_df):
        """生成执行摘要"""
        summary = {
            'key_findings': [],
            'main_indicators': {}
        }

        # 关键指标
        if 'fusion_index' in farmer_df:
            summary['main_indicators']['avg_farmer_fusion'] = f"{farmer_df['fusion_index'].mean():.2%}"

        if 'per_capita_income' in village_df:
            summary['main_indicators']['avg_income'] = f"{village_df['per_capita_income'].mean():,.0f}元"

        if 'revenue' in industry_df:
            summary['main_indicators']['total_revenue'] = f"{industry_df['revenue'].sum():,.0f}万元"

        # 关键发现
        if 'state' in farmer_df:
            state_dist = farmer_df['state'].value_counts(normalize=True)
            no_participation = state_dist.get(1, 0)
            if no_participation > 0.5:
                summary['key_findings'].append('农户组织化程度较低，需加强合作社培育')

        return summary

    def _analyze_clustering_results(self, df):
        """分析聚类结果"""
        if 'cluster' not in df:
            return None

        analysis = {
            'cluster_distribution': df['cluster'].value_counts().to_dict(),
            'cluster_characteristics': {}
        }

        for cluster in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster]
            analysis['cluster_characteristics'][int(cluster)] = {
                'size': len(cluster_data),
                'avg_fusion': cluster_data['fusion_index'].mean() if 'fusion_index' in cluster_data else 0,
                'avg_income': cluster_data['annual_income'].mean() if 'annual_income' in cluster_data else 0
            }

        return analysis

    def _analyze_fusion_status(self, farmer_df, village_df):
        """分析融合现状"""
        analysis = {}

        if 'fusion_index' in farmer_df:
            analysis['farmer_fusion'] = {
                'mean': farmer_df['fusion_index'].mean(),
                'std': farmer_df['fusion_index'].std(),
                'distribution': {
                    'high': len(farmer_df[farmer_df['fusion_index'] > 0.6]),
                    'medium': len(farmer_df[(farmer_df['fusion_index'] >= 0.4) & (farmer_df['fusion_index'] <= 0.6)]),
                    'low': len(farmer_df[farmer_df['fusion_index'] < 0.4])
                }
            }

        if 'fusion_index' in village_df:
            analysis['village_fusion'] = {
                'mean': village_df['fusion_index'].mean(),
                'std': village_df['fusion_index'].std(),
                'top_villages': village_df.nlargest(5, 'fusion_index')[['village_name', 'fusion_index']].to_dict('records') if 'village_name' in village_df else []
            }

        return analysis

    def _generate_recommendations(self, farmer_df, village_df, industry_df):
        """生成综合建议"""
        recommendations = []

        # 基于农户聚类
        if 'cluster_name' in farmer_df:
            for cluster_name, group in farmer_df.groupby('cluster_name'):
                if '传统' in cluster_name:
                    recommendations.append({
                        'target': cluster_name,
                        'priority': '高',
                        'action': '组织培育与技能培训',
                        'expected_outcome': '提升组织化程度，增加收入'
                    })

        # 基于村庄融合度
        if 'fusion_index' in village_df:
            low_fusion_villages = village_df[village_df['fusion_index'] < 0.4]
            if len(low_fusion_villages) > 0:
                recommendations.append({
                    'target': f'{len(low_fusion_villages)}个低融合村庄',
                    'priority': '高',
                    'action': '重点帮扶与基础设施建设',
                    'expected_outcome': '提升区域整体融合水平'
                })

        return recommendations

    def export_report(self, report, output_path):
        """导出报告"""
        import json

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"报告已导出: {output_path}")


if __name__ == '__main__':
    print("测试智能决策支持模块...")

    from utils.data_generator import DataGenerator
    generator = DataGenerator()

    farmer_df = generator.generate_farmer_data(200)
    village_df = generator.generate_village_data(20)
    industry_df = generator.generate_industry_data(50)

    engine = DecisionSupportEngine()

    # 测试产业规划
    planning = engine.generate_industry_planning(industry_df)
    print("\n产业规划:")
    print(f"  总收入: {planning['diagnosis']['total_revenue']:.0f}")
    print(f"  优先产业数: {len(planning['priority_industries'])}")

    # 测试选址建议
    site_result = engine.generate_site_selection(
        village_df,
        '农产品加工',
        {'min_area': 1000}
    )
    print("\n选址建议:")
    print(f"  推荐地点数: {len(site_result['recommendations'])}")

    # 测试风险评估
    risks = engine.generate_risk_assessment(industry_df)
    print("\n风险评估:")
    print(f"  综合风险等级: {risks['overall']['level']}")
