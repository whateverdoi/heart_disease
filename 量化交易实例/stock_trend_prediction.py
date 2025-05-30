import akshare as ak
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def fetch_stock_data(ticker, start_date, end_date):
    """获取股票历史数据 (使用 akshare)"""
    try:
        # akshare 的日期格式通常是 YYYYMMDD
        start_date_ak = start_date.replace("-", "")
        end_date_ak = end_date.replace("-", "")
        
        # 获取A股日度历史数据（后复权）
        df = ak.stock_zh_a_hist(symbol=ticker, 
                                 period="daily", 
                                 start_date=start_date_ak, 
                                 end_date=end_date_ak, 
                                 adjust="hfq") # hfq: 后复权, qfq: 前复权, '': 不复权
        
        if df is None or df.empty:
            print(f"使用 akshare 未能获取到 {ticker} 的数据。")
            return pd.DataFrame()
            
        # akshare 返回的列名与 yfinance 不同，进行重命名以尽量兼容后续代码
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
    df_copy['Close'] = pd.to_numeric(df_copy['Close'], errors='coerce')

    # 计算每日收益率
    df_copy['Return'] = df_copy['Close'].pct_change()
    
    # 创建简单的移动平均线特征
    df_copy['SMA_5'] = df_copy['Close'].rolling(window=5).mean()
    df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean()
    
    # 创建未来N天的价格变动方向作为标签 (1: 上涨, 0: 下跌或持平)
    df_copy['Target'] = (df_copy['Close'].shift(-1) > df_copy['Close']).astype(int)
    
    # 移除包含NaN值的行
    df_copy = df_copy.dropna()
    
    return df_copy

def train_and_evaluate_model(df):
    """训练模型并评估"""
    if df.empty or len(df) < 25: 
        print("数据不足，无法训练模型。")
        return

    # 选择特征和目标
    # 'Open', 'High', 'Low', 'Close', 'Volume' 已经在 fetch_stock_data 中重命名
    features = ['Return', 'SMA_5', 'SMA_20', 'Open', 'High', 'Low', 'Close', 'Volume']
    features = [f for f in features if f in df.columns]
    
    if not features:
        print("没有可用的特征列。")
        return
        
    X = df[features]
    y = df['Target']

    if len(X) != len(y):
        print("特征和标签长度不匹配。")
        return
        
    if len(X) == 0:
        print("没有可用于训练的数据。")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    if len(X_train) == 0 or len(X_test) == 0:
        print("训练集或测试集为空。")
        return

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    print("\n开始训练模型...")
    model.fit(X_train, y_train)
    print("模型训练完成。")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n模型在测试集上的准确率: {accuracy:.4f}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\n特征重要性:")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importances)

if __name__ == "__main__":
    # 定义股票代码和日期范围
    # akshare 通常使用纯数字代码，如 '000001' 代表平安银行
    ticker_symbol = "000001" # 平安银行
    start_date = "2010-01-01"
    end_date = "2023-12-31"

    print(f"正在获取 {ticker_symbol} 从 {start_date} 到 {end_date} 的数据 (使用 akshare)...")
    stock_df = fetch_stock_data(ticker_symbol, start_date, end_date)

    if not stock_df.empty:
        print(f"获取到 {len(stock_df)} 条数据。")
        print("\n正在创建特征...")
        featured_df = create_features(stock_df)
        
        if not featured_df.empty:
            print("特征创建完成。")
            train_and_evaluate_model(featured_df)
        else:
            print("创建特征后数据为空，请检查数据或特征工程逻辑。")
    else:
        print(f"未能获取到 {ticker_symbol} 的数据，请检查股票代码或日期范围或 akshare 是否正常工作。")

    print("\n示例运行结束。") 