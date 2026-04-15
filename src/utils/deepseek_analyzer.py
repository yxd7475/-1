# -*- coding: utf-8 -*-
"""
DeepSeek AI 智能分析模块
提供基于大语言模型的智能分析能力
"""
import requests
import json
from typing import Dict, List, Optional
import os


class DeepSeekAnalyzer:
    """DeepSeek AI分析器"""

    def __init__(self, api_key: str = None, base_url: str = "https://api.deepseek.com/v1"):
        """
        初始化DeepSeek分析器

        Parameters:
        -----------
        api_key : str
            DeepSeek API密钥，如未提供将从环境变量获取
        base_url : str
            API基础URL
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url
        self.model = "deepseek-chat"

    def set_api_key(self, api_key: str):
        """设置API密钥"""
        self.api_key = api_key

    def is_available(self) -> bool:
        """检查API是否可用"""
        return bool(self.api_key)

    def chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 4000) -> str:
        """
        调用DeepSeek聊天API

        Parameters:
        -----------
        messages : List[Dict]
            消息列表
        temperature : float
            温度参数
        max_tokens : int
            最大token数

        Returns:
        --------
        str
            AI回复内容
        """
        if not self.api_key:
            return "错误：未配置DeepSeek API密钥"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # 禁用代理
        proxies = {
            "http": None,
            "https": None,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
                proxies=proxies
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"API调用失败: {response.status_code} - {response.text}"

        except Exception as e:
            return f"API调用异常: {str(e)}"

    def analyze_data(self, data_summary: Dict, analysis_type: str = "comprehensive") -> str:
        """
        分析数据并生成智能报告

        Parameters:
        -----------
        data_summary : Dict
            数据摘要
        analysis_type : str
            分析类型

        Returns:
        --------
        str
            分析结果
        """
        prompt = self._build_analysis_prompt(data_summary, analysis_type)

        messages = [
            {"role": "system", "content": "你是一位专业的城乡产业融合发展分析专家，擅长数据分析、政策研究和战略规划。请用专业的角度进行分析，给出具体、可操作的建议。"},
            {"role": "user", "content": prompt}
        ]

        return self.chat(messages)

    def _build_analysis_prompt(self, data_summary: Dict, analysis_type: str) -> str:
        """构建分析提示词"""
        prompt = f"""
请基于以下数据进行城乡产业融合发展的专业分析：

## 数据概览
- 农户总数: {data_summary.get('farmer_count', 'N/A')}
- 村庄总数: {data_summary.get('village_count', 'N/A')}
- 产业记录数: {data_summary.get('industry_count', 'N/A')}
- 平均融合指数: {data_summary.get('avg_fusion_index', 'N/A')}
- 平均农户收入: {data_summary.get('avg_income', 'N/A')}

## 聚类分析结果
{json.dumps(data_summary.get('clustering_result', {}), ensure_ascii=False, indent=2)}

## 融合指数分析
{json.dumps(data_summary.get('fusion_analysis', {}), ensure_ascii=False, indent=2)}

## 产业分析
{json.dumps(data_summary.get('industry_analysis', {}), ensure_ascii=False, indent=2)}

## 风险评估
{json.dumps(data_summary.get('risk_assessment', {}), ensure_ascii=False, indent=2)}

请从以下几个方面进行深入分析：
1. **现状诊断**：当前区域城乡产业融合发展的主要特征和问题
2. **趋势判断**：基于数据的发展趋势预判
3. **瓶颈识别**：制约融合发展的关键瓶颈
4. **对策建议**：针对性的政策建议和实施路径
5. **风险预警**：潜在风险点和应对措施

请用专业、简洁的语言进行分析，每部分给出3-5个要点。
"""
        return prompt

    def generate_policy_recommendations(self, context: Dict) -> str:
        """生成政策建议"""
        prompt = f"""
基于以下城乡产业融合发展数据，请生成具体的政策建议：

区域概况：
- 农户数: {context.get('farmer_count', 0)}
- 平均融合指数: {context.get('avg_fusion', 0):.2%}
- 主要问题: {context.get('issues', [])}

请从以下维度提出政策建议：
1. 产业扶持政策
2. 人才引进政策
3. 金融支持政策
4. 基础设施建设
5. 社会保障政策

每个维度给出2-3条具体措施，包括政策名称、实施主体、预期效果。
"""
        messages = [
            {"role": "system", "content": "你是政府政策研究专家，专注于农业农村政策制定。"},
            {"role": "user", "content": prompt}
        ]

        return self.chat(messages)

    def explain_clustering_result(self, cluster_info: Dict) -> str:
        """解释聚类结果"""
        prompt = f"""
请解释以下农户聚类分析结果：

聚类信息：
{json.dumps(cluster_info, ensure_ascii=False, indent=2)}

请分析：
1. 每个聚类群体的特征和定位
2. 各群体间的发展差距
3. 针对各群体的差异化扶持策略
"""
        messages = [
            {"role": "system", "content": "你是数据分析专家，擅长解读聚类分析结果。"},
            {"role": "user", "content": prompt}
        ]

        return self.chat(messages)

    def generate_executive_summary(self, report_data: Dict) -> str:
        """生成执行摘要"""
        prompt = f"""
请为以下城乡产业融合发展分析报告撰写执行摘要：

报告数据：
{json.dumps(report_data, ensure_ascii=False, indent=2)}

要求：
1. 总结关键发现（3-5条）
2. 核心指标概览
3. 主要建议（3条以内）
4. 风险提示

篇幅控制在500字以内，语言精练专业。
"""
        messages = [
            {"role": "system", "content": "你是报告撰写专家，擅长撰写简洁有力的执行摘要。"},
            {"role": "user", "content": prompt}
        ]

        return self.chat(messages)


class AIAnalyzerFactory:
    """AI分析器工厂类"""

    @staticmethod
    def create_analyzer(provider: str = "deepseek", **kwargs):
        """
        创建AI分析器

        Parameters:
        -----------
        provider : str
            服务提供商
        **kwargs : dict
            额外参数

        Returns:
        --------
        analyzer
            AI分析器实例
        """
        if provider == "deepseek":
            return DeepSeekAnalyzer(**kwargs)
        else:
            raise ValueError(f"不支持的AI服务提供商: {provider}")


if __name__ == "__main__":
    # 测试
    analyzer = DeepSeekAnalyzer()

    # 模拟数据
    test_data = {
        "farmer_count": 1000,
        "village_count": 50,
        "avg_fusion_index": 0.45,
        "avg_income": 28000,
        "clustering_result": {
            "cluster_1": {"size": 300, "avg_fusion": 0.65, "type": "融合发展型"},
            "cluster_2": {"size": 400, "avg_fusion": 0.45, "type": "初级融合型"},
            "cluster_3": {"size": 300, "avg_fusion": 0.25, "type": "传统型"}
        }
    }

    if analyzer.is_available():
        result = analyzer.analyze_data(test_data)
        print("AI分析结果:")
        print(result)
    else:
        print("DeepSeek API未配置，请设置DEEPSEEK_API_KEY环境变量")
