#!/bin/bash
# 城乡产业融合智能决策系统启动脚本

echo "========================================"
echo "城乡产业融合智能决策系统"
echo "========================================"
echo

# 检查Python
if ! command -v python &> /dev/null; then
    echo "[错误] 未检测到Python，请先安装Python 3.8+"
    exit 1
fi

echo "[信息] Python环境检测通过"
echo

# 安装依赖
echo "[步骤1] 安装依赖包..."
pip install -r requirements.txt -q

echo
echo "[步骤2] 启动Streamlit应用..."
echo
echo "========================================"
echo "系统启动中，请在浏览器中访问:"
echo "http://localhost:8501"
echo "========================================"
echo

streamlit run app.py --server.port=8501 --server.address=localhost
