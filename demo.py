# -*- coding: utf-8 -*-
"""
城乡产业融合智能决策系统 - 演示脚本
用于快速测试和展示系统功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_generator import DataGenerator
from src.models.kmeans_model import FarmerClusterAnalyzer, IndustryClusterAnalyzer
from src.analysis.fusion_index import FusionIndexCalculator
from src.analysis.decision_support import DecisionSupportEngine


def demo_kmeans_clustering():
    """演示KMeans聚类"""
    print("\n" + "="*60)
    print("【演示1】KMeans农户聚类分析")
    print("="*60)

    # 生成数据
    print("\n1. 生成模拟数据...")
    generator = DataGenerator(seed=42)
    farmer_data = generator.generate_farmer_data(500)
    print(f"   生成 {len(farmer_data)} 条农户数据")

    # 执行聚类
    print("\n2. 执行KMeans聚类...")
    analyzer = FarmerClusterAnalyzer(n_clusters=4)
    result = analyzer.cluster_farmers(farmer_data)

    # 显示结果
    print("\n3. 聚类结果:")
    print("-" * 40)
    cluster_dist = result['cluster_name'].value_counts()
    for name, count in cluster_dist.items():
        pct = count / len(result) * 100
        print(f"   {name}: {count} 户 ({pct:.1f}%)")

    # 各聚类特征
    print("\n4. 各聚类特征对比:")
    print("-" * 40)
    feature_cols = ['land_area', 'capital', 'fusion_index', 'annual_income']
    cluster_means = result.groupby('cluster_name')[feature_cols].mean()
    print(cluster_means.to_string())

    # 转型建议
    print("\n5. 转型建议:")
    print("-" * 40)
    recommendations = analyzer.get_transition_recommendations(result)
    for cluster, rec in recommendations.items():
        print(f"\n   【{rec['cluster_name']}】")
        print(f"   规模: {rec['size']} 户")
        print(f"   短板: {', '.join(rec['weak_dimensions'])}")
        for sug in rec['recommendations'][:2]:
            print(f"   建议: {sug['suggestion']}")


def demo_fusion_index():
    """演示融合指数计算"""
    print("\n" + "="*60)
    print("【演示2】城乡融合指数计算")
    print("="*60)

    # 生成数据
    print("\n1. 生成农户数据...")
    generator = DataGenerator(seed=42)
    farmer_data = generator.generate_farmer_data(300)

    # 计算融合指数
    print("\n2. 计算融合指数...")
    calculator = FusionIndexCalculator()
    result = calculator.calculate_farmer_fusion(farmer_data)

    # 显示结果
    print("\n3. 融合指数统计:")
    print("-" * 40)
    print(f"   平均融合指数: {result['fusion_index'].mean():.2%}")
    print(f"   标准差: {result['fusion_index'].std():.2%}")
    print(f"   最小值: {result['fusion_index'].min():.2%}")
    print(f"   最大值: {result['fusion_index'].max():.2%}")

    # 各维度指数
    print("\n4. 各维度融合指数:")
    print("-" * 40)
    dimensions = ['f_production', 'f_supply', 'f_market', 'f_service', 'f_value']
    for dim in dimensions:
        if dim in result:
            print(f"   {dim.replace('f_', '')}: {result[dim].mean():.2%}")

    # 差距分析
    print("\n5. 融合差距分析:")
    print("-" * 40)
    analysis = calculator.analyze_fusion_gap(result)
    print(f"   短板维度: {', '.join(analysis['shortcomings'])}")

    # 改进建议
    print("\n6. 改进建议:")
    suggestions = calculator.generate_improvement_suggestions(analysis)
    for sug in suggestions[:2]:
        print(f"\n   【{sug['title']}】")
        print(f"   措施: {sug['measures'][0]}")


def demo_decision_support():
    """演示智能决策支持"""
    print("\n" + "="*60)
    print("【演示3】智能决策支持")
    print("="*60)

    # 生成数据
    print("\n1. 生成产业数据...")
    generator = DataGenerator(seed=42)
    industry_data = generator.generate_industry_data(100)

    # 创建决策引擎
    engine = DecisionSupportEngine()

    # 产业规划
    print("\n2. 产业规划分析:")
    print("-" * 40)
    planning = engine.generate_industry_planning(industry_data)
    print(f"   总产值: {planning['diagnosis']['total_revenue']:,.0f} 万元")
    print(f"   总利润: {planning['diagnosis']['total_profit']:,.0f} 万元")
    print(f"   平均融合度: {planning['diagnosis']['avg_fusion_degree']:.2%}")

    # 优势产业
    print("\n3. 优先发展产业 (TOP 5):")
    print("-" * 40)
    for i, ind in enumerate(planning['priority_industries'][:5], 1):
        print(f"   {i}. {ind['industry_type']}: 潜力得分 {ind['potential_score']:.3f}")

    # 风险评估
    print("\n4. 风险评估:")
    print("-" * 40)
    risks = engine.generate_risk_assessment(industry_data)
    print(f"   综合风险等级: {risks['overall']['level']}")
    print(f"   风险得分: {risks['overall']['score']:.2%}")

    # 投资分析
    print("\n5. 投资分析示例:")
    print("-" * 40)
    project = {
        'name': '农产品加工项目',
        'investment': 5000000,
        'expected_revenue': 8000000,
        'operating_cost': 6000000,
        'project_life': 10
    }
    analysis = engine.generate_investment_analysis(project, industry_data)
    print(f"   项目: {analysis['project_name']}")
    print(f"   投资回报率: {analysis['financial_indicators']['roi']}")
    print(f"   回收期: {analysis['financial_indicators']['payback_period']}")
    print(f"   可行性: {analysis['feasibility']['level']}")


def demo_full_workflow():
    """完整工作流演示"""
    print("\n" + "="*60)
    print("【完整工作流演示】")
    print("="*60)

    # 1. 数据生成
    print("\n步骤1: 数据生成")
    generator = DataGenerator(seed=42)
    data = generator.generate_all_data(n_farmers=500, n_villages=30)
    print(f"   农户: {len(data['farmers'])} 条")
    print(f"   村庄: {len(data['villages'])} 条")
    print(f"   产业: {len(data['industries'])} 条")

    # 2. KMeans聚类
    print("\n步骤2: KMeans聚类分析")
    farmer_analyzer = FarmerClusterAnalyzer(n_clusters=4)
    clustered = farmer_analyzer.cluster_farmers(data['farmers'])
    print(f"   聚类分布: {clustered['cluster_name'].value_counts().to_dict()}")

    # 3. 融合指数
    print("\n步骤3: 融合指数计算")
    calculator = FusionIndexCalculator()
    fusion_result = calculator.calculate_farmer_fusion(data['farmers'])
    print(f"   平均融合指数: {fusion_result['fusion_index'].mean():.2%}")

    # 4. 决策支持
    print("\n步骤4: 智能决策支持")
    engine = DecisionSupportEngine()
    planning = engine.generate_industry_planning(data['industries'])
    print(f"   优先产业数: {len(planning['priority_industries'])}")

    print("\n" + "="*60)
    print("演示完成!")
    print("="*60)


def main():
    """主函数"""
    print("\n" + "="*60)
    print("城乡产业融合智能决策系统 - 功能演示")
    print("="*60)

    print("\n请选择演示内容:")
    print("  1. KMeans农户聚类分析")
    print("  2. 融合指数计算")
    print("  3. 智能决策支持")
    print("  4. 完整工作流演示")
    print("  5. 全部演示")
    print("  0. 退出")

    choice = input("\n请输入选项 (0-5): ").strip()

    if choice == '1':
        demo_kmeans_clustering()
    elif choice == '2':
        demo_fusion_index()
    elif choice == '3':
        demo_decision_support()
    elif choice == '4':
        demo_full_workflow()
    elif choice == '5':
        demo_kmeans_clustering()
        demo_fusion_index()
        demo_decision_support()
        demo_full_workflow()
    elif choice == '0':
        print("\n退出演示程序。")
        return
    else:
        print("\n无效选项，请重新运行程序。")

    print("\n提示: 运行 'streamlit run app.py' 启动Web界面")


if __name__ == '__main__':
    main()
