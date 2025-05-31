import akshare as ak
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import datetime
import warnings
import pickle
import joblib
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据保存路径
DATA_DIR = "F:\\数据分析项目合集\\量化交易实例\\data"
# 图片保存路径
PICS_DIR = "F:\\数据分析项目合集\\量化交易实例\\pics"
# 模型保存路径
MODELS_DIR = "F:\\数据分析项目合集\\量化交易实例\\models"
# 报告保存路径
REPORTS_DIR = "F:\\数据分析项目合集\\量化交易实例\\reports"

# 创建必要的目录
for directory in [DATA_DIR, PICS_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

def get_file_path(ticker, start_date, end_date):
    """生成数据文件路径"""
    filename = f"{ticker}_{start_date}_{end_date}.csv"
    return os.path.join(DATA_DIR, filename)

def save_stock_data(df, ticker, start_date, end_date):
    """保存股票数据到CSV文件"""
    file_path = get_file_path(ticker, start_date, end_date)
    df.to_csv(file_path)
    print(f"数据已保存到: {file_path}")
    return file_path

def load_stock_data(ticker, start_date, end_date):
    """从CSV文件加载股票数据"""
    file_path = get_file_path(ticker, start_date, end_date)
    if os.path.exists(file_path):
        print(f"正在从本地文件加载数据: {file_path}")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df, True
    return pd.DataFrame(), False

def fetch_stock_data(ticker, start_date, end_date):
    """获取股票历史数据 (使用 akshare)"""
    try:
        # akshare 的日期格式通常是 YYYYMMDD
        start_date_ak = start_date.replace("-", "")
        end_date_ak = end_date.replace("-", "")
        
        print(f"正在通过akshare获取 {ticker} 从 {start_date} 到 {end_date} 的数据...")
        
        # 获取A股日度历史数据（后复权）
        df = ak.stock_zh_a_hist(symbol=ticker, 
                                 period="daily", 
                                 start_date=start_date_ak, 
                                 end_date=end_date_ak, 
                                 adjust="hfq") # hfq: 后复权, qfq: 前复权, '': 不复权
        
        if df is None or df.empty:
            print(f"使用 akshare 未能获取到 {ticker} 的数据。")
            return pd.DataFrame()
            
        # akshare 返回的列名与 yfinance 不同，进行重命名以兼容后续代码
        # '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率'
        df.rename(columns={
            '日期': 'Date',
            '开盘': 'Open',
            '收盘': 'Close',
            '最高': 'High',
            '最低': 'Low',
            '成交量': 'Volume'
        }, inplace=True)
        
        # 将 Date 列转换为 datetime 对象并设为索引
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 确保数据按日期升序排列
        df.sort_index(ascending=True, inplace=True)
        
        # 保存数据
        save_stock_data(df, ticker, start_date, end_date)
        
        return df
        
    except Exception as e:
        print(f"获取 {ticker} 数据时发生错误: {e}")
        return pd.DataFrame()

def create_features(df):
    """创建特征"""
    if df.empty:
        return pd.DataFrame()
    df_copy = df.copy()
    
    # 确保 'Close' 列是数值类型
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # 计算每日收益率
    df_copy['Return'] = df_copy['Close'].pct_change()
    
    # 创建移动平均线特征
    df_copy['SMA_5'] = df_copy['Close'].rolling(window=5).mean()
    df_copy['SMA_10'] = df_copy['Close'].rolling(window=10).mean()
    df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean()
    df_copy['SMA_60'] = df_copy['Close'].rolling(window=60).mean()
    
    # 计算价格与移动平均线的差距
    df_copy['Price_SMA5_Ratio'] = df_copy['Close'] / df_copy['SMA_5']
    df_copy['Price_SMA20_Ratio'] = df_copy['Close'] / df_copy['SMA_20']
    
    # 计算波动率
    df_copy['Volatility_5'] = df_copy['Return'].rolling(window=5).std()
    df_copy['Volatility_20'] = df_copy['Return'].rolling(window=20).std()
    
    # RSI指标
    delta = df_copy['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df_copy['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD指标
    exp1 = df_copy['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_copy['Close'].ewm(span=26, adjust=False).mean()
    df_copy['MACD'] = exp1 - exp2
    df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
    df_copy['MACD_Hist'] = df_copy['MACD'] - df_copy['MACD_Signal']
    
    # 创建未来N天的价格变动方向作为标签 (1: 上涨, 0: 下跌或持平)
    df_copy['Target'] = (df_copy['Close'].shift(-1) > df_copy['Close']).astype(int)
    
    # 移除包含NaN值的行
    df_copy = df_copy.dropna()
    
    return df_copy

def plot_price_history(df, ticker):
    """绘制股价历史走势图"""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='收盘价')
    plt.plot(df.index, df['SMA_20'], label='20日均线', alpha=0.7)
    plt.plot(df.index, df['SMA_60'], label='60日均线', alpha=0.7)
    
    plt.title(f'{ticker} 股价走势图')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存到指定路径
    save_path = os.path.join(PICS_DIR, f"{ticker}_price_history.png")
    plt.savefig(save_path)
    print(f"股价走势图已保存到: {save_path}")
    

def plot_returns_distribution(df, ticker):
    """绘制收益率分布图"""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Return'].dropna(), kde=True, bins=50)
    plt.title(f'{ticker} 日收益率分布')
    plt.xlabel('日收益率')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存到指定路径
    save_path = os.path.join(PICS_DIR, f"{ticker}_returns_distribution.png")
    plt.savefig(save_path)
    print(f"收益率分布图已保存到: {save_path}")
    

def plot_feature_importance(model, feature_names, model_name, ticker):
    """绘制特征重要性图"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print(f"{model_name} 模型不支持特征重要性可视化")
        return
        
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'{ticker} - {model_name} 特征重要性')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('相对重要性')
    plt.tight_layout()
    
    # 保存到指定路径
    save_path = os.path.join(PICS_DIR, f"{ticker}_{model_name}_feature_importance.png")
    plt.savefig(save_path)
    print(f"{model_name}特征重要性图已保存到: {save_path}")
    

def plot_confusion_matrix(y_true, y_pred, model_name, ticker):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{ticker} - {model_name} 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    
    # 保存到指定路径
    save_path = os.path.join(PICS_DIR, f"{ticker}_{model_name}_confusion_matrix.png")
    plt.savefig(save_path)
    print(f"{model_name}混淆矩阵已保存到: {save_path}")
    

def save_model(model, model_name, ticker):
    """保存模型到本地文件"""
    # 确保目录存在
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 保存模型
    today = datetime.datetime.now().strftime("%Y%m%d")
    model_path = os.path.join(MODELS_DIR, f"{ticker}_{model_name}_{today}.pkl")
    
    try:
        joblib.dump(model, model_path)
        print(f"模型已保存到: {model_path}")
        return True
    except Exception as e:
        print(f"保存模型时发生错误: {e}")
        return False

def train_and_evaluate_models(df, ticker):
    """训练多种模型并评估"""
    if df.empty or len(df) < 25: 
        print("数据不足，无法训练模型。")
        return

    # 选择特征
    features = ['Return', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_60', 
                'Price_SMA5_Ratio', 'Price_SMA20_Ratio', 
                'Volatility_5', 'Volatility_20', 
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
                'Open', 'High', 'Low', 'Close', 'Volume']
    
    # 过滤出存在的特征列
    features = [f for f in features if f in df.columns]
    
    if not features:
        print("没有可用的特征列。")
        return
        
    X = df[features]
    y = df['Target']

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False, random_state=42)

    # 定义模型
    models = {
        '随机森林': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        '逻辑回归': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    predictions = {}
    best_accuracy = 0
    best_model_name = ''
    trained_models = {}  # 存储训练好的模型
    
    # 训练和评估各个模型
    for model_name, model in models.items():
        print(f"\n开始训练 {model_name} 模型...")
        model.fit(X_train, y_train)
        trained_models[model_name] = model  # 保存训练好的模型
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        predictions[model_name] = y_pred
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
        
        print(f"{model_name} 模型测试集准确率: {accuracy:.4f}")
        
        # 如果是随机森林，显示特征重要性
        if model_name == '随机森林':
            importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            print(f"\n{model_name} 特征重要性:")
            print(importances)
            
        # 绘制混淆矩阵
        plot_confusion_matrix(y_test, y_pred, model_name, ticker)
        
        # 绘制特征重要性（如果支持）
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            plot_feature_importance(model, X.columns, model_name, ticker)
    
    # 返回结果用于生成报告和保存模型
    return {
        'ticker': ticker,
        'features': features,
        'results': results,
        'best_model': best_model_name,
        'best_accuracy': best_accuracy,
        'test_period': (X_test.index[0].strftime('%Y-%m-%d'), X_test.index[-1].strftime('%Y-%m-%d')),
        'data_size': {
            'total': len(df),
            'train': len(X_train),
            'test': len(X_test)
        },
        'models': trained_models,  # 保存训练好的模型
        'scaler': scaler  # 保存标准化器以便后续使用
    }

def generate_report(report_data):
    """生成预测结果总结报告并保存到文件"""
    if not report_data:
        print("无报告数据生成")
        return
    
    ticker = report_data['ticker']
    best_model = report_data['best_model']
    best_accuracy = report_data['best_accuracy']
    test_period = report_data['test_period']
    
    # 创建报告内容
    report_content = []
    report_content.append("="*80)
    report_content.append(f"股票 {ticker} 预测结果总结报告")
    report_content.append("="*80)
    
    report_content.append(f"\n测试周期: {test_period[0]} 至 {test_period[1]}")
    report_content.append(f"数据总量: {report_data['data_size']['total']} 条")
    report_content.append(f"训练集: {report_data['data_size']['train']} 条")
    report_content.append(f"测试集: {report_data['data_size']['test']} 条")
    
    report_content.append(f"\n最佳模型: {best_model}")
    report_content.append(f"最佳准确率: {best_accuracy:.4f}")
    
    report_content.append("\n所有模型性能比较:")
    model_accuracies = [(model, data['accuracy']) for model, data in report_data['results'].items()]
    model_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    for model, accuracy in model_accuracies:
        precision = report_data['results'][model]['classification_report']['1']['precision']
        recall = report_data['results'][model]['classification_report']['1']['recall']
        f1 = report_data['results'][model]['classification_report']['1']['f1-score']
        
        report_content.append(f"- {model}: 准确率 {accuracy:.4f}, 上涨精确率 {precision:.4f}, 上涨召回率 {recall:.4f}, F1分数 {f1:.4f}")
    
    report_content.append("\n使用的特征:")
    for feature in report_data['features']:
        report_content.append(f"- {feature}")
    
    report_content.append("\n建议:")
    if best_accuracy > 0.65:
        report_content.append("模型表现良好，可以考虑实际应用。")
    elif best_accuracy > 0.55:
        report_content.append("模型表现一般，建议继续优化或增加更多特征。")
    else:
        report_content.append("模型表现不佳，可能需要重新设计特征或考虑其他模型。")
    
    report_content.append("="*80)
    
    # 将内容保存到文件
    today = datetime.datetime.now().strftime("%Y%m%d")
    report_filename = f"{ticker}_预测报告_{today}.txt"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        for line in report_content:
            f.write(line + "\n")
    
    print(f"报告已保存到: {report_path}")
    
    # 同时在终端显示报告
    for line in report_content:
        print(line)

def main():
    """主函数"""
    # 显示已保存的数据文件
    print(f"\n{DATA_DIR} 目录中的文件：")
    if os.path.exists(DATA_DIR):
        files = os.listdir(DATA_DIR)
        if files:
            for file in files:
                print(f"- {file}")
        else:
            print("目前没有文件")
    else:
        print("数据目录不存在")
    
    # 询问用户是否使用保存的数据
    use_saved_data = input("是否使用已保存的数据？(y/n): ").strip().lower() == 'y'
    
    if use_saved_data:
        # 询问股票代码和日期范围
        ticker = input("请输入股票代码(例如: 000001): ").strip()
        start_date = input("请输入开始日期(格式: YYYY-MM-DD): ").strip()
        end_date = input("请输入结束日期(格式: YYYY-MM-DD): ").strip()
        
        # 尝试加载已保存的数据
        stock_df, success = load_stock_data(ticker, start_date, end_date)
        
        if not success:
            print(f"未找到股票 {ticker} 在 {start_date} 至 {end_date} 期间的保存数据。")
            use_akshare = input("是否使用akshare重新获取数据？(y/n): ").strip().lower() == 'y'
            
            if use_akshare:
                stock_df = fetch_stock_data(ticker, start_date, end_date)
            else:
                print("程序退出。")
                return
    else:
        # 询问股票代码和日期范围
        ticker = input("请输入股票代码(例如: 000001): ").strip()
        start_date = input("请输入开始日期(格式: YYYY-MM-DD): ").strip()
        end_date = input("请输入结束日期(格式: YYYY-MM-DD): ").strip()
        
        # 通过akshare获取数据
        stock_df = fetch_stock_data(ticker, start_date, end_date)
    
    if stock_df.empty:
        print("未能获取数据，程序退出。")
        return
    
    print(f"获取到 {len(stock_df)} 条数据。")
    
    # 创建特征
    print("\n正在创建特征...")
    featured_df = create_features(stock_df)
    
    if featured_df.empty:
        print("创建特征后数据为空，请检查数据或特征工程逻辑。")
        return
        
    print(f"特征创建完成，特征数据共 {len(featured_df)} 条。")
    
    # 可视化股价历史
    plot_price_history(featured_df, ticker)
    
    # 可视化收益率分布
    plot_returns_distribution(featured_df, ticker)
    
    # 训练和评估模型
    report_data = train_and_evaluate_models(featured_df, ticker)
    
    # 生成报告
    if report_data:
        generate_report(report_data)
        
        # 询问用户是否保存模型
        save_models = input("\n是否需要保存模型？(y/n): ").strip().lower() == 'y'
        
        if save_models:
            available_models = list(report_data['models'].keys())
            print(f"\n可用模型: {', '.join(available_models)}")
            print(f"最佳模型: {report_data['best_model']} (准确率: {report_data['best_accuracy']:.4f})")
            
            model_to_save = input("请输入要保存的模型名称(输入'best'选择最佳模型，输入'all'保存所有模型): ").strip()
            
            if model_to_save.lower() == 'all':
                # 保存所有模型
                for name, model in report_data['models'].items():
                    save_model(model, name, ticker)
                # 保存标准化器
                scaler_path = os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")
                joblib.dump(report_data['scaler'], scaler_path)
                print(f"标准化器已保存到: {scaler_path}")
                
            elif model_to_save.lower() == 'best':
                # 保存最佳模型
                best_model_name = report_data['best_model']
                save_model(report_data['models'][best_model_name], best_model_name, ticker)
                # 保存标准化器
                scaler_path = os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")
                joblib.dump(report_data['scaler'], scaler_path)
                print(f"标准化器已保存到: {scaler_path}")
                
            elif model_to_save in available_models:
                # 保存指定模型
                save_model(report_data['models'][model_to_save], model_to_save, ticker)
                # 保存标准化器
                scaler_path = os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")
                joblib.dump(report_data['scaler'], scaler_path)
                print(f"标准化器已保存到: {scaler_path}")
                
            else:
                print(f"未找到名为 '{model_to_save}' 的模型，未保存任何模型。")
    
    print("\n分析完成。")

if __name__ == "__main__":
    main() 