import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ydata_profiling as pandas_profiling
import os


plt.rcParams['font.sans-serif'] = ['SimHei']

# 定义输出目录
output_dir = r'F:\数据分析项目合集\心脏病数据集分析'
os.makedirs(os.path.join(output_dir, 'figs'), exist_ok=True)

# 1. 读取数据
df = pd.read_csv('心脏病数据集分析\dataset\Medicaldataset.csv')

# 2. 数据基本信息
profile = pandas_profiling.ProfileReport(df, title="心脏病数据集分析报告", explorative=True)
profile.to_file(os.path.join(output_dir, "report.html"))

# 3. 生成主要可视化图片

# （1）阳性/阴性分布
plt.figure(figsize=(6,4))
sns.countplot(x='Result', data=df)
plt.title('心脏病诊断结果分布')
plt.savefig(os.path.join(output_dir, 'figs/result_dist.png'))
plt.close()

# （2）年龄分布
plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('年龄分布')
plt.savefig(os.path.join(output_dir, 'figs/age_dist.png'))
plt.close()

# （3）性别与心脏病关系
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', hue='Result', data=df)
plt.title('性别与心脏病诊断结果关系')
plt.xticks([0, 1], ['女性', '男性'])
plt.savefig(os.path.join(output_dir, 'figs/gender_result.png'))
plt.close()

# （4）年龄与心脏病关系
plt.figure(figsize=(8,6))
sns.boxplot(x='Result', y='Age', data=df)
plt.title('年龄与心脏病诊断结果关系')
plt.savefig(os.path.join(output_dir, 'figs/age_result.png'))
plt.close()

# （5）各特征与结果关系
features = ['Age', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']
for feat in features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Result', y=feat, data=df)
    plt.title(f'{feat}与诊断结果关系')
    plt.savefig(os.path.join(output_dir, f'figs/{feat}_box.png'))
    plt.close()

# （6）相关性热力图
plt.figure(figsize=(10,8))
corr = df.drop('Result', axis=1).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('特征相关性热力图')
plt.savefig(os.path.join(output_dir, 'figs/corr_heatmap.png'))
plt.close()

# （7）风险因素分布
risk_factors = ['Systolic blood pressure', 'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']
plt.figure(figsize=(12,8))
for i, factor in enumerate(risk_factors, 1):
    plt.subplot(2,3,i)
    sns.histplot(data=df, x=factor, hue='Result', multiple="stack")
    plt.title(f'{factor}分布')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figs/risk_factors_dist.png'))
plt.close()

# 4. 生成HTML报告（用Jinja2模板）
from jinja2 import Template
html_template = '''
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>心脏病数据集分析报告</title>
    <style>
        body { font-family: SimHei, Arial, sans-serif; margin: 40px; }
        h1, h2, h3 { color: #2c3e50; }
        img { max-width: 600px; margin: 20px 0; }
        .section { margin: 30px 0; }
        .highlight { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>心脏病数据集分析报告</h1>
    
    <div class="section">
        <h2>1. 数据集基本信息</h2>
        <p>样本数：{{ n_samples }}，特征数：{{ n_features }}</p>
        <p>特征列表：{{ features }}</p>
    </div>

    <div class="section">
        <h2>2. 人口统计学分析</h2>
        <h3>2.1 诊断结果分布</h3>
        <img src="figs/result_dist.png">
        <h3>2.2 性别与心脏病关系</h3>
        <img src="figs/gender_result.png">
        <h3>2.3 年龄分布</h3>
        <img src="figs/age_dist.png">
        <h3>2.4 年龄与心脏病关系</h3>
        <img src="figs/age_result.png">
    </div>

    <div class="section">
        <h2>3. 关键指标分析</h2>
        <h3>3.1 各特征与诊断结果关系</h3>
        {% for feat in features_plot %}
            <h4>{{ feat }}</h4>
            <img src="figs/{{ feat }}_box.png">
        {% endfor %}
    </div>

    <div class="section">
        <h2>4. 风险因素分析</h2>
        <h3>4.1 风险因素分布</h3>
        <img src="figs/risk_factors_dist.png">
        <h3>4.2 特征相关性热力图</h3>
        <img src="figs/corr_heatmap.png">
    </div>

    <div class="section highlight">
        <h2>5. 主要发现</h2>
        <ul>
            <li>人口统计学特征（年龄、性别）与心脏病的关系</li>
            <li>关键生理指标（血压、心率）的分布特征</li>
            <li>心肌标志物（CK-MB、肌钙蛋白）的诊断价值</li>
            <li>各风险因素之间的相关性</li>
        </ul>
    </div>

    <div class="section">
        <h2>6. 自动化数据分析报告</h2>
        <p>详见附录：ydata-profiling自动生成的分析报告（report.html）。</p>
    </div>
</body>
</html>
'''

# 渲染HTML
report_html = Template(html_template).render(
    n_samples=df.shape[0],
    n_features=df.shape[1]-1,
    features=', '.join(df.columns[:-1]),
    features_plot=features
)
with open(os.path.join(output_dir, 'report_main.html'), 'w', encoding='utf-8') as f:
    f.write(report_html)

print('分析完成，已生成 report_main.html（主报告）和 report.html（详细分析）') 