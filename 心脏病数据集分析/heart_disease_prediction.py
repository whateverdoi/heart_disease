import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = r'F:\数据分析项目合集\心脏病数据集分析'
os.makedirs(os.path.join(output_dir, 'figs'), exist_ok=True)

# 读取数据
df = pd.read_csv(os.path.join(output_dir, 'dataset\Medicaldataset.csv'))

# 将结果转换为数值型（positive=1, negative=0）
df['Result'] = df['Result'].map({'positive': 1, 'negative': 0})

# 准备特征和目标变量
X = df.drop('Result', axis=1)
y = df['Result']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建并训练模型
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 预测
y_pred = model.predict(X_test_scaled)

# 评估模型
print("模型评估报告：")
print("\n准确率：", accuracy_score(y_test, y_pred))
print("\n分类报告：")
print(classification_report(y_test, y_pred))

# 特征重要性
feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('重要性', ascending=False)

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
sns.barplot(x='重要性', y='特征', data=feature_importance)
plt.title('特征重要性分析')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figs/feature_importance.png'))
plt.close()

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.savefig(os.path.join(output_dir, 'figs/confusion_matrix.png'))
plt.close()

# 保存模型评估结果
with open(os.path.join(output_dir, 'model_evaluation.txt'), 'w', encoding='utf-8') as f:
    f.write("心脏病风险预测模型评估报告\n")
    f.write("="*50 + "\n\n")
    f.write(f"模型准确率: {accuracy_score(y_test, y_pred):.4f}\n\n")
    f.write("分类报告:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\n特征重要性:\n")
    f.write(feature_importance.to_string())

print("\n模型评估结果已保存到 'model_evaluation.txt'")
print("特征重要性图已保存到 'figs/feature_importance.png'")
print("混淆矩阵图已保存到 'figs/confusion_matrix.png'") 