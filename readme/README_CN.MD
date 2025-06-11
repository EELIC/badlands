
badlands 安装

git clone https://github.com/badlands-model/badlands-companion.git
git clone https://github.com/badlands-model/badlands.git
git clone https://github.com/badlands-model/badlands-workshop.git
安装涉及重要文件及目录
badlands/
├── badlands-companion/    ← 可视化支持工具
│   ├── badlands-companion/
│   │   ├── __init__.py
│   │   └── ...其他...
│   ├── README.md
│   ├── requirements.txt
│   ├── setup.cfg
│   ├── setup.cfg
│   └── ...其他...
├── badlands-master/    ← 主程序             
│   ├── badlands/
│   │   ├── badlands/   ← 源码主包 
│   │   │    ├── __init__.py
│   │   │    ├── model.py
│   │   │    └── ...其他...
│   │   ├── MANIFEST.in
│   │   ├── meson.build
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── setup.cfg
│   │   └── setup.py
│   ├── environment.yml
│   ├── README.md
│   └── ...其他...
├── badlands-workshop/  ← 示例
│   ├── companion/
│   ├── examples/
│   │   ├── basin/
│   │   │   ├── data/
│   │   │   ├── images/
│   │   │   ├── basin.ipynb
│   │   │   ├── basin.xml
│   │   │   └── ...其他...
│   │   └── ...其他...
│   ├── workshop/
│   └── ...其他...
├── badlands-setting/
│   └── ...其他自己编写的辅助学习说明文件...
├── dev_run.py             ← ✔️ 运行模型测试脚本
├── install_badlands.bat   ← ✔️ 一键环境安装
├── .vscode/
│   └── launch.json        ← ✔️ 调试配置
└──其它文件

# 批处理安装
## Anaconda PowerShell 运行 
在 badlands 虚拟环境中执行：
conda install setuptools
conda install -c anaconda python=3.8
pip install distutils

## 安装失败后重新安装需要清理
重新运行 .bat 之前，清理旧失败残留（可选）：
conda deactivate
rd /s /q D:\code-python\badlands\badlands-master\build
rd /s /q D:\code-python\badlands\badlands-master\badlands.egg-info

最后检查
## 成功后运行：

conda activate badlands
python -c "import badlands; print('Badlands is installed.')"

## 再测试运行：
python dev_run.py

# 错误集锦
## badlands-companion 安装失败（缺少 distutils）
❌ 错误 2：badlands-companion 安装失败（缺少 distutils）
💡 错误提示：
vbnet
复制
编辑
ModuleNotFoundError: No module named 'distutils.msvccompiler'
✅ 原因：
Anaconda 默认并不安装 distutils。该模块在 Python 3.10+ 被拆分出 stdlib，如果 companion 项目仍使用老式 setup.py + numpy.distutils，你需要手动安装：

✅ 解决办法：
在 badlands 虚拟环境中执行：

bash
复制
编辑
conda install setuptools
conda install -c anaconda python=3.8
pip install distutils
或者更稳妥：

bash
复制
编辑
pip install setuptools==59.5.0



# ✅ 安装依赖
   必要依赖：
   Python 3.8
   VSCode 编辑器
   Anaconda

1 vscode 虚拟环境 python 版本 3.8
2 下载并安装 Anaconda3-2024.10-1-Windows-x86_64.exe
链接: https://www.anaconda.com/download/success
3 验证 conda 安装:

# 📥 克隆 Badlands 项目
vscode 终端运行:

进入目录 ruolin\bandlands-master

## badlands 主程序
git clone https://github.com/EELIC/badlands.git
git clone https://github.com/badlands-model/badlands.git
## badlands_companion  
- badlands_companion 是 Badlands 团队开发的 辅助工具库，默认并不包含在 badlands 主库中，它是一个单独的 GitHub 仓库：
  目的 提供网格生成、可视化、剖面分析等功能
  下载文件
git clone https://github.com/badlands-model/badlands-companion.git
 
进入文件夹 badlands-master\badlands>
复制 environment.yml到文件夹

## 安装示例文件 badlands-workshop
git clone https://github.com/badlands-model/badlands-workshop.git

# windows修改目录结构

badlands-master/
├── badlands/                 ← 源码主包，统一放这里
│   ├── __init__.py
│   ├── model.py
│   └── ...其他模块...
├── setup.py
├── setup.cfg
├── README.md
├── environment.yml
├── work/
│   └── examples/
│       └── basin/
│           └── basin.xml
└── companion/

### 修改说明
所有代码放在 badlands/ 包中：符合 setuptools 推荐做法；

安装路径清晰，避免 ../README.md 非法跨路径；

使用 setup.cfg + 最简 setup.py：兼容性强、避免 dash-deprecated 警告；

install_requires 放在 setup.cfg 里更清晰，避免手动管理 requirements.txt。

# 尝试自动批处理安装
ruolin\badlands-setting> setup_badlands_full.bat

# 🐍 创建并激活虚拟环境 
## 整合并测试兼容性 后的最小可行 environment.yml，基于：

Python 3.8（兼容 Badlands）
Numpy 1.21
SciPy 限制为 1.9.3（最后兼容 3.8 的稳定版本）
避免使用 m2w64-gcc-libs 等 Windows 下易冲突包
尽量使用 conda-forge 提供的包（更稳定）

## 使用conda自带python编译器 创建兼容的 Conda 环境 
打开Anaconda PowerShell Prompt
### 查看所有环境
conda env list
### 如果存在 badlands 进入卸载
## 卸载安装(安装失败,重新安装时需要先卸载原有环境)
conda env remove -n badlands
## conda deactivate 禁用环境

## 📦 正常安装 badlands Python 依赖
Anaconda PowerShell Prompt 进入路径： ruolin\bandlands-master\main\ 
conda env create --file environment.yml  

## (出现错误步骤) 激活已有环境并安装 environment.yml 中的依赖(如果发现某些依赖缺失的话--通常是外网下包的网络连接问题)
conda env update --file environment.yml --prune

## 激活 badlands环境
conda activate badlands

## 验证安装成功
conda env list 
![conda环境安装验证成功图片](/res/conda_env_success.png "conda_env_success")

## ⚙️ 编译 badlands
python setup.py install
![py安装验证成功图片](/res/py_install_success.png "py_install_success")

## 验证安装
运行以下命令检查是否成功安装：
python -c "import badlands; print(badlands.__file__)"
如果路径正确输出 .py 或 .pyc 文件路径，则一切就绪。

成功提示 : Processing dependencies for badlands==2.0
Finished processing dependencies for badlands==2.0

# 可能存在的问题

## 安装 badlands_companion.toolGeo 脚本依赖的可视化库 plotly

conda install -c plotly plotly

本地克隆 + 加入路径 克隆源码，手动添加到 sys.path

([ 当前路径](badlands-master\badlands-companion))
python
import sys
sys.path.append('../badlands-companion')
import badlands_companion.toolGeo as simple

### ：降级 NumPy 到 1.26.4
pip install numpy==1.26.4
确认安装版本： 
python -c "import numpy; print(numpy.__version__)"
****** 输出应为：1.26.4

# 运行
## 激活现有环境：

conda activate badlands
回应:
**********************************************************************
** Visual Studio 2019 Developer Command Prompt v16.11.16
** Copyright (c) 2021 Microsoft Corporation
**********************************************************************
[vcvarsall.bat] Environment initialized for: 'x64'

## 检查现有环境是否已安装好 Badlands

python -c "import badlands; print('Badlands is installed.')"
如果已经激活 回应 Badlands is installed.

## 运行参考示例

切换目录 badlands-master\
python -m badlands.badlands.model run -i badlands-workshop/examples/basin/basin.xml
python -m badlands.badlands.model run -i badlands-workshop/examples/basin/basin.xml

该命令的含义是：

-m badlands.model：通过模块方式运行；

run：执行模型；

-i basin.xml：输入 XML 模型配置。

若执行后目录下出现 .h5 输出文件，如 basin.nc、basin_topo.h5，表示运行成功。


# 参考资料

## Conda
🐍 什么是 Conda？
Conda 是一个开源的跨平台包管理器和环境管理工具，用于：

安装、运行和更新 Python（或其他语言）的包；

创建、隔离、管理虚拟环境；

自动解决依赖冲突，比 pip 更智能。

🧩 Conda VS pip 的区别
功能	Conda	pip
包管理	支持 Python + 非 Python（如 C库）	仅支持 Python 包
依赖管理能力	强，自动解决依赖	弱，容易冲突
安装速度	快（用预编译的二进制包）	慢（需源码编译）
环境隔离能力	强	需要借助 virtualenv 或 venv

🛠️ Conda 常用命令
🌱 环境管理
bash
复制
编辑
### 创建新环境
conda create -n myenv python=3.9

### 激活环境
conda activate myenv

### 退出环境
conda deactivate

### 删除环境
conda remove -n myenv --all

### 克隆环境
conda create --name myenv2 --clone myenv

### 查看所有环境
conda env list
📦 包管理
bash
复制
编辑
### 安装包
conda install numpy

### 指定通道安装
conda install -c conda-forge matplotlib

### 升级包
conda update numpy

### 删除包
conda remove scipy

### 搜索包
conda search pandas
📄 使用 environment.yml 文件
bash
复制
编辑
### 从 YML 文件创建环境
conda env create -f environment.yml

### 更新已有环境
conda env update -f environment.yml --prune

### 导出当前环境
conda env export > environment.yml
🔗 多通道（源）
默认 Conda 使用官方源，建议添加 conda-forge：

复制
编辑
conda config --add channels conda-forge
conda config --set channel_priority strict

## pyproject.toml 
pyproject.toml  是 Python 项目的标准构建配置文件，用于声明如何构建、打包和安装你的项目。