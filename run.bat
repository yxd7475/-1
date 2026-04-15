@echo off
chcp 65001 >nul
echo ========================================
echo 城乡产业融合智能决策系统
echo ========================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo [信息] Python环境检测通过
echo.

REM 安装依赖
echo [步骤1] 安装依赖包...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [警告] 部分依赖安装失败，尝试继续...
)

echo.
echo [步骤2] 启动Streamlit应用...
echo.
echo ========================================
echo 系统启动中，请在浏览器中访问:
echo http://localhost:8501
echo ========================================
echo.

streamlit run app.py --server.port=8501 --server.address=localhost

pause
