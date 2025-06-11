
# 🧭 Badlands 安装与使用指南（Windows + Anaconda + VS Code）

适用于以下项目结构：

```
D:\code-python\badlands\
├── badlands-master\          ← 主程序
├── badlands-companion\       ← 可视化支持工具
├── badlands-workshop\        ← 示例模型
```

---

## 📦 1. 创建 Conda 环境

打开 **Anaconda PowerShell Prompt**：

```powershell
conda create -n badlands python=3.8 -y
conda activate badlands
```

---

### 使用 Conda 创建一个名为 badlands 的独立 Python 环境
命令：

bash
复制
编辑
conda create -n badlands python=3.8 -y
是使用 Conda 创建一个名为 badlands 的独立 Python 环境，具体解释如下：

🔍 参数逐项详解
参数	说明
conda	使用 Conda 工具（来自 Anaconda 或 Miniconda）
create	创建新环境的操作
-n badlands	-n 是 --name 的简写，表示新环境的名称为 badlands
python=3.8	指定 Python 解释器的版本为 3.8（Badlands 当前推荐使用）
-y	自动确认，不再提示 [y/n] 安装确认

📦 此命令的作用
创建隔离环境 badlands，不会污染你的主 Python 安装；

安装 Python 3.8 作为基础；

为后续安装 Badlands 及其依赖打好基础；

可通过 conda activate badlands 进入该环境；

所有在此环境中的包、路径、设置彼此隔离，便于管理多个项目或版本。

📝 补充说明
Conda 环境相当于“虚拟沙箱”，适合科研和多项目开发；

badlands 环境名你可以自定义，比如：conda create -n geo python=3.8；

若你已安装了 Anaconda 或 Miniconda，即可使用本命令；

创建后执行：

bash
复制
编辑
conda activate badlands
### conda env create -f environment.yml
📁 1. environment.yml：Conda 环境配置文件
✅ 作用：
用于创建或重建一个完整的 Conda 虚拟环境。

可以包括：

Python 版本

Conda 依赖（numpy, scipy 等）

pip 依赖（通过 pip: 子项）

安装源（channel）

环境名称

📌 使用方式：
bash
复制
编辑
conda env create -f environment.yml
或更新已有环境：

bash
复制
编辑
conda env update -f environment.yml --prune

## 📚 2. 安装依赖包

```powershell
conda install numpy=1.21 scipy=1.9.3 pandas h5py scikit-image shapely cmocean jupyterlab -c conda-forge -y

pip install matplotlib==3.3.4 plotly triangle gFlex lavavu piglet pyvirtualdisplay descartes
pip install git+https://github.com/kinnala/meshplex
```

---

## 🛠 3. 安装 `badlands` 主程序（开发模式）

```powershell
cd D:\code-python\badlands\badlands-master\badlands\
pip install -e .
```

> 注意：务必在 `badlands-master\badlands/` 目录执行 `pip install -e .`，否则将出错！

---

## 🧰 4. 安装 companion 工具模块

```powershell
cd D:\code-python\badlands\badlands-companion
pip install -e .
```

如遇 `No module named 'distutils.msvccompiler'`：

```powershell
pip install setuptools==59.5.0
pip install distutils
```

---

## ✅ 5. 测试模块是否安装成功

```powershell
python -c "import badlands; print('Badlands OK')"
python -c "import badlands_companion; print('Companion OK')"
```

---

## 🚀 6. 运行一个示例模型

```powershell
cd D:\code-python\badlands\badlands-workshop
set PYTHONPATH=D:\code-python\badlands\badlands-master
python -m badlands.model run -i examples/basin/basin.xml
```

输出 `.nc` 文件后，可用 `plot_topography.py` 进行可视化。

---

## 🧪 7. VS Code 开发环境配置

### 启动 VS Code 并选择解释器

```powershell
code D:\code-python\badlands
```

按 `Ctrl+Shift+P`，输入 `Python: Select Interpreter`，选择：

```
Anaconda3/envs/badlands
```

### 配置 `.vscode/launch.json`

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run dev_run.py",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/dev_run.py",
      "console": "integratedTerminal"
    }
  ]
}
```

---

## 📊 8. 模型输出可视化脚本（示例）

### `plot_topography.py` – 绘制最终地形图

```python
import xarray as xr
import matplotlib.pyplot as plt
import os

nc_path = os.path.join("badlands-workshop", "examples", "basin", "badlands.nc")
ds = xr.open_dataset(nc_path)
topo = ds['topo'][-1, :, :]

plt.imshow(topo, cmap='terrain')
plt.title("Final Topography")
plt.colorbar()
plt.show()
```

---

## 📌 推荐开发结构

```
D:\code-python\badlands\
├── dev_run.py              ← 运行模型
├── plot_topography.py      ← 绘图工具
├── plot_erosion_profile.py ← 剖面分析
├── badlands-master\
├── badlands-companion\
├── badlands-workshop\
├── .vscode\
│   └── launch.json
```

---

## 📝 附注

- 可用 `jupyter lab` 分析输出 `.nc` 文件
- 所有模块使用 `pip install -e .` 支持源码级修改与调试
- 可加入版本控制（如 Git）管理模型与工具脚本
