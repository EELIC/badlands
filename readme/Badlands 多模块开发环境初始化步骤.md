✅ 一览：Badlands 多模块开发环境初始化步骤
我们分为以下七步：

css
复制
编辑
1️⃣ 创建 Conda 环境
2️⃣ 安装主模块 badlands
3️⃣ 安装 companion 辅助工具包
4️⃣ 安装开发工具包（pip editable）
5️⃣ VS Code 配置（解释器 + 调试）
6️⃣ 测试运行示例
7️⃣ 后续建议（脚本组织/二次开发）
🥇 1️⃣ 创建 Conda 虚拟环境（推荐 Python 3.8）
bash
复制
编辑
conda create -n badlands python=3.8
conda activate badlands
📦 2️⃣ 安装主包 badlands（源码在 badlands-master）
进入 badlands-master/badlands/：

bash
复制
编辑
cd badlands\badlands-master\badlands
pip install -e .
💡 说明：

-e . 表示开发模式安装（源码变化立即生效）

安装时请确保 pyproject.toml 中 未使用 meson（否则会失败）

如遇 meson 报错，请将 pyproject.toml 替换为：

toml
复制
编辑
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"
📦 3️⃣ 安装 badlands-companion
进入 badlands-companion/：

bash
复制
编辑
cd ..\..\badlands-companion
pip install -e .
📦 4️⃣ 安装其他依赖包（来自 requirements 和 environment）
建议合并所有依赖包，如需完整推荐版本我可为你生成。先快速安装：

bash
复制
编辑
pip install -r requirements.txt
conda install numpy=1.21 scipy=1.9.3 pandas h5py scikit-image shapely cmocean jupyterlab
pip install matplotlib==3.3.4 plotly triangle gFlex lavavu piglet pyvirtualdisplay descartes
pip install git+https://github.com/kinnala/meshplex
🧪 5️⃣ VS Code 配置开发环境
✅ 打开 VS Code：
打开项目根目录（例如 badlands/），推荐在 VS Code 中使用命令行运行：

bash
复制
编辑
code .
✅ 设置 Python 解释器：
Ctrl + Shift + P → 选择 Python: Select Interpreter

选择你的环境 badlands，路径如：

makefile
复制
编辑
C:\Users\eason\anaconda3\envs\badlands\python.exe
✅ 安装插件：
✅ Python

✅ Jupyter

✅ Pylance（智能补全）

✅ GitLens（代码版本控制）

✅ IntelliCode（代码建议）

🚀 6️⃣ 测试运行一个示例（basin.xml）
创建文件 run_basin.py：

python
复制
编辑
from badlands.model import Model

model = Model()
model.load("..\\badlands-workshop\\examples\\basin\\basin.xml")
model.run()
然后在终端运行：

bash
复制
编辑
python run_basin.py
或在 VS Code 中点击 ▶ 运行。

🧠 7️⃣ 开发建议（结构管理）
📁 badlands-setting/ 中可写你自己的脚本、测试用例、调试笔记

🧪 推荐创建：

pgsql
复制
编辑
dev/
  test_run.py
  plot_tools.py
  experiment_01.xml
notebooks/
  exploratory.ipynb
✅ 用 pip install -e . 模式，开发时改代码无需重新安装

✅ 用 conda env export > environment.yml 保存环境备份


📁 1️⃣ install_badlands.bat 脚本（简洁版）
bat
复制
编辑
@echo off
setlocal

rem === Configuration ===
set ENV_NAME=badlands
set CONDA_PATH=%USERPROFILE%\anaconda3
set WORKSPACE=D:\badlands
set MAIN_PKG=%WORKSPACE%\badlands-master\badlands
set COMPANION_PKG=%WORKSPACE%\badlands-companion

rem === Activate environment ===
call "%CONDA_PATH%\Scripts\activate.bat" %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment: %ENV_NAME%
    pause
    exit /b 1
)

rem === Install badlands main package ===
cd /d %MAIN_PKG%
echo Installing badlands main package...
pip install -e .

rem === Install companion package ===
cd /d %COMPANION_PKG%
echo Installing badlands companion package...
pip install -e .

rem === Install dependencies ===
echo Installing required packages...
conda install -y numpy=1.21 scipy=1.9.3 pandas h5py scikit-image shapely cmocean jupyterlab
pip install matplotlib==3.3.4 plotly triangle gFlex lavavu piglet pyvirtualdisplay descartes
pip install git+https://github.com/kinnala/meshplex

echo [OK] Installation completed.
pause
📄 2️⃣ dev_run.py 测试脚本
python
复制
编辑
from badlands.model import Model
import os

# Locate the example XML file
example_path = os.path.join("..", "badlands-workshop", "examples", "basin", "basin.xml")
print("Loading example model from:", example_path)

model = Model()
model.load(example_path)
model.run()

print("Model run completed.")
🧷 3️⃣ .vscode/launch.json 调试配置
json
复制
编辑
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run Badlands Example",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/dev_run.py",
      "console": "integratedTerminal"
    }
  ]
}
✅ 使用说明
保存 install_badlands.bat 到任意目录，右键 → 以管理员身份运行

运行成功后，用 VS Code 打开你的根目录 badlands/

按 F5 或点击 ▶️ 运行 dev_run.py

输出无报错即成功！

📊 1️⃣ plot_topography.py – 读取 Badlands 输出并绘图
此脚本读取 badlands.nc（默认输出文件），展示最终地形图（topo 变量）：

python
复制
编辑
import xarray as xr
import matplotlib.pyplot as plt
import os

# 路径配置：可根据实际调整
nc_path = os.path.join("..", "badlands-workshop", "examples", "basin", "badlands.nc")

# 读取 NetCDF 文件
ds = xr.open_dataset(nc_path)
topo = ds['topo'][-1, :, :]  # 最后一个时间步的地形

# 绘图
plt.figure(figsize=(8, 6))
plt.imshow(topo, cmap='terrain')
plt.title("Final Topography")
plt.colorbar(label="Elevation (m)")
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
plt.show()
你可以在 badlands 虚拟环境中运行它来快速可视化模拟结果。

📓 2️⃣ badlands_analysis_template.ipynb – Jupyter 分析模板
请保存以下为：badlands_analysis_template.ipynb，或复制粘贴至新 notebook。

json
复制
编辑
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🌍 Badlands 输出分析模板\n",
    "\n",
    "本 Notebook 读取并分析 Badlands 模拟输出 (`badlands.nc`)，包括地形、流域、沉积等。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 导入库\n",
    "import xarray as xr\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 加载 NetCDF 输出文件\n",
    "nc_path = os.path.join(\"..\", \"badlands-workshop\", \"examples\", \"basin\", \"badlands.nc\")\n",
    "ds = xr.open_dataset(nc_path)\n",
    "ds"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🎨 最终地形图"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 获取最后时间步地形\n",
    "topo = ds['topo'][-1, :, :]\n",
    "\n",
    "plt.figure(figsize=(8, 6))\n",
    "plt.imshow(topo, cmap='terrain')\n",
    "plt.title(\"Final Topography\")\n",
    "plt.colorbar(label='Elevation (m)')\n",
    "plt.xlabel(\"X\")\n",
    "plt.ylabel(\"Y\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (badlands)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
📌 如果你在 VS Code 中使用 Jupyter 插件，只需打开 .ipynb 文件并点击单元格即可逐步执行分析。
✅ 下一步建议
你可以将 dev_run.py、plot_topography.py 和 Notebook 一起放在 badlands-setting/ 或 dev/ 文件夹中统一管理。

若你希望增加：

流域分析（如 catchment mask）

侵蚀剖面图

时间序列变化图（例如某点高程随时间）


下面是一个可运行的 Python 脚本（或 Jupyter 单元），用于生成 Badlands 模拟输出中的侵蚀剖面图（沿某条线查看地形变化），支持对比多个时间步：

📁 脚本名建议：plot_erosion_profile.py
python
复制
编辑
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置路径（按需修改）
nc_path = os.path.join("..", "badlands-workshop", "examples", "basin", "badlands.nc")
ds = xr.open_dataset(nc_path)

# 基本参数
topos = ds['topo']        # 地形变量: [time, y, x]
times = ds['time'].values # 时间轴
profile_line = 50         # 沿 y=50 行（横向剖面），你也可以改为 x=某列
axis = 'row'              # 'row' = 横向，'col' = 纵向剖面

# 选取时间点（例如：初始 / 中期 / 最终）
time_indices = [0, len(times)//2, len(times)-1]

# 取剖面数据
profiles = []
for t in time_indices:
    if axis == 'row':
        profile = topos[t, profile_line, :].values
    else:
        profile = topos[t, :, profile_line].values
    profiles.append(profile)

# 绘图
plt.figure(figsize=(10, 5))
for i, p in enumerate(profiles):
    label = f"Time = {times[time_indices[i]]:.2f} Myr"
    plt.plot(p, label=label)

plt.title(f"Erosion Profile at {'Y' if axis=='row' else 'X'} = {profile_line}")
plt.xlabel("X Distance (grid units)" if axis == 'row' else "Y Distance (grid units)")
plt.ylabel("Elevation (m)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
📌 说明
参数	含义
profile_line = 50	第 50 行（y=50）作为剖面，可改成任意值
axis = 'row'	'row' 为横剖面（沿 x），'col' 为纵剖面（沿 y）
time_indices	选择要对比的时间点，可任意组合
topos[t, y, x]	访问某一时间步的地形网格

📓 用法拓展（在 Jupyter 中使用）
可以将这段代码粘贴到你已有的 Notebook 中一个单元格里，也可以放入你自己的 badlands-setting 开发文件夹中运行。

