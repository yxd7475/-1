# -*- coding: utf-8 -*-
"""
城乡产业融合智能决策系统 - FastAPI后端服务
提供RESTful API接口
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import uvicorn
import os
import sys
import json
from datetime import datetime
import io

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_generator import DataGenerator
from src.utils.data_processor import DataProcessor
from src.models.kmeans_model import FarmerClusterAnalyzer, IndustryClusterAnalyzer, RegionalCompetitivenessAnalyzer
from src.analysis.fusion_index import FusionIndexCalculator
from src.analysis.decision_support import DecisionSupportEngine, ReportGenerator

# 创建应用
app = FastAPI(
    title="城乡产业融合智能决策系统API",
    description="提供数据分析、聚类、决策支持等API接口",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局实例
generator = DataGenerator()
processor = DataProcessor()
fusion_calculator = FusionIndexCalculator()
decision_engine = DecisionSupportEngine()
report_generator = ReportGenerator()


# ==================== 数据模型 ====================

class FarmerData(BaseModel):
    """农户数据模型"""
    farmer_id: str
    region: str
    dist_town: float
    land_area: float
    labor: int
    capital: float
    state: int = 1


class ClusteringRequest(BaseModel):
    """聚类请求"""
    data: List[Dict[str, Any]]
    n_clusters: int = 4
    features: Optional[List[str]] = None


class SiteSelectionRequest(BaseModel):
    """选址请求"""
    villages: List[Dict[str, Any]]
    project_type: str
    requirements: Dict[str, Any]


class InvestmentRequest(BaseModel):
    """投资分析请求"""
    project_name: str
    investment: float
    expected_revenue: float
    operating_cost: float
    project_life: int = 10


# ==================== API端点 ====================

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "城乡产业融合智能决策系统API",
        "version": "1.0.0",
        "endpoints": [
            "/api/data/generate",
            "/api/clustering/farmers",
            "/api/clustering/industries",
            "/api/fusion/calculate",
            "/api/decision/planning",
            "/api/decision/site-selection",
            "/api/decision/risk-assessment",
            "/api/decision/investment",
            "/api/report/generate"
        ]
    }


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ==================== 数据生成API ====================

@app.post("/api/data/generate")
async def generate_data(
    n_farmers: int = Query(1000, ge=100, le=10000),
    n_villages: int = Query(50, ge=10, le=500),
    region_name: str = Query("示范县")
):
    """
    生成模拟数据

    Parameters:
    - n_farmers: 农户数量
    - n_villages: 村庄数量
    - region_name: 区域名称
    """
    try:
        data = generator.generate_all_data(n_farmers, n_villages, region_name)

        return {
            "success": True,
            "message": "数据生成成功",
            "data": {
                "farmers": data['farmers'].to_dict('records')[:100],  # 返回前100条
                "villages": data['villages'].to_dict('records'),
                "statistics": {
                    "total_farmers": len(data['farmers']),
                    "total_villages": len(data['villages']),
                    "avg_fusion_index": float(data['farmers']['fusion_index'].mean())
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/upload")
async def upload_data(file: UploadFile = File(...)):
    """
    上传数据文件

    支持 CSV, Excel 格式
    """
    try:
        contents = await file.read()

        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="不支持的文件格式")

        return {
            "success": True,
            "message": "文件上传成功",
            "data": {
                "filename": file.filename,
                "rows": len(df),
                "columns": list(df.columns),
                "preview": df.head(10).to_dict('records')
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 聚类分析API ====================

@app.post("/api/clustering/farmers")
async def cluster_farmers(request: ClusteringRequest):
    """
    农户聚类分析

    Parameters:
    - data: 农户数据列表
    - n_clusters: 聚类数量
    - features: 特征列名
    """
    try:
        df = pd.DataFrame(request.data)

        analyzer = FarmerClusterAnalyzer(n_clusters=request.n_clusters)
        result = analyzer.cluster_farmers(df)

        # 获取评估指标
        from sklearn.preprocessing import StandardScaler
        feature_cols = [f for f in analyzer.clustering_features if f in df.columns]
        X_scaled = StandardScaler().fit_transform(df[feature_cols].values)
        metrics = analyzer.get_evaluation_metrics(X_scaled)

        return {
            "success": True,
            "message": "聚类分析完成",
            "data": {
                "cluster_distribution": result['cluster_name'].value_counts().to_dict(),
                "cluster_centers": result.groupby('cluster_name')[feature_cols].mean().to_dict(),
                "evaluation_metrics": metrics,
                "clustered_data": result.to_dict('records')[:200]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clustering/industries")
async def cluster_industries(request: ClusteringRequest):
    """
    产业聚类分析
    """
    try:
        df = pd.DataFrame(request.data)

        analyzer = IndustryClusterAnalyzer(n_clusters=request.n_clusters)
        result = analyzer.cluster_industries(df)

        advantage = analyzer.identify_advantage_industries(result)

        return {
            "success": True,
            "message": "产业聚类完成",
            "data": {
                "cluster_distribution": result['cluster'].value_counts().to_dict(),
                "advantage_industries": advantage.to_dict('records'),
                "clustered_data": result.to_dict('records')
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 融合指数API ====================

@app.post("/api/fusion/calculate")
async def calculate_fusion_index(data: List[Dict[str, Any]]):
    """
    计算融合指数
    """
    try:
        df = pd.DataFrame(data)
        result = fusion_calculator.calculate_farmer_fusion(df)

        analysis = fusion_calculator.analyze_fusion_gap(result)

        return {
            "success": True,
            "message": "融合指数计算完成",
            "data": {
                "fusion_index": {
                    "mean": float(result['fusion_index'].mean()),
                    "std": float(result['fusion_index'].std()),
                    "min": float(result['fusion_index'].min()),
                    "max": float(result['fusion_index'].max())
                },
                "dimension_averages": {
                    "production": float(result['f_production'].mean()),
                    "supply": float(result['f_supply'].mean()),
                    "market": float(result['f_market'].mean()),
                    "service": float(result['f_service'].mean()),
                    "value": float(result['f_value'].mean())
                },
                "gap_analysis": analysis,
                "calculated_data": result.to_dict('records')[:200]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 决策支持API ====================

@app.post("/api/decision/planning")
async def generate_industry_planning(
    industries: List[Dict[str, Any]],
    region_name: str = Query("示范县")
):
    """
    生成产业规划
    """
    try:
        df = pd.DataFrame(industries)
        planning = decision_engine.generate_industry_planning(df, region_name)

        return {
            "success": True,
            "message": "产业规划生成完成",
            "data": planning
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/decision/site-selection")
async def generate_site_selection(request: SiteSelectionRequest):
    """
    项目选址建议
    """
    try:
        df = pd.DataFrame(request.villages)
        result = decision_engine.generate_site_selection(df, request.project_type, request.requirements)

        return {
            "success": True,
            "message": "选址分析完成",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/decision/risk-assessment")
async def generate_risk_assessment(data: List[Dict[str, Any]]):
    """
    风险评估
    """
    try:
        df = pd.DataFrame(data)
        risks = decision_engine.generate_risk_assessment(df)

        return {
            "success": True,
            "message": "风险评估完成",
            "data": risks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/decision/investment")
async def analyze_investment(request: InvestmentRequest):
    """
    投资分析
    """
    try:
        project_info = {
            'name': request.project_name,
            'investment': request.investment,
            'expected_revenue': request.expected_revenue,
            'operating_cost': request.operating_cost,
            'project_life': request.project_life
        }

        # 使用模拟产业数据
        industry_df = generator.generate_industry_data(50)
        analysis = decision_engine.generate_investment_analysis(project_info, industry_df)

        return {
            "success": True,
            "message": "投资分析完成",
            "data": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 报告API ====================

@app.post("/api/report/generate")
async def generate_comprehensive_report(
    n_farmers: int = Query(1000),
    n_villages: int = Query(50),
    region_name: str = Query("示范县")
):
    """
    生成综合报告
    """
    try:
        # 生成数据
        data = generator.generate_all_data(n_farmers, n_villages, region_name)

        # 生成报告
        report = report_generator.generate_comprehensive_report(
            data['farmers'],
            data['villages'],
            data['industries'],
            region_name
        )

        return {
            "success": True,
            "message": "综合报告生成完成",
            "data": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 启动服务器 ====================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
