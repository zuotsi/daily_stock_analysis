# -*- coding: utf-8 -*-
"""
===================================
趋势交易分析器 V2.0 (Pro版)
===================================

升级策略核心：多因子共振
1. 趋势因子：均线多头排列 (MA20向上)
2. 动能因子：MACD 金叉或处于零轴上方
3. 情绪因子：RSI < 70 (未超买)，回调不破 RSI 50
4. 波动因子：利用 ATR 动态计算止损位

"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# --- 扩展的数据类 ---

class TrendStatus(Enum):
    STRONG_BULL = "强势多头"
    BULL = "多头趋势"
    CONSOLIDATION = "震荡整理"
    BEAR = "空头趋势"
    STRONG_BEAR = "极度弱势"

class BuySignal(Enum):
    STRONG_BUY = "★ 强力买入"     # 趋势+动能+量能共振
    BUY = "买入信号"              # 趋势良好，回踩支撑
    HOLD = "持股待涨"             # 趋势未坏
    WAIT = "空仓观望"             # 无机会
    RISK_WARNING = "⚠️ 风险警示"  # 顶背离或超买
    SELL = "卖出信号"             # 趋势破坏

@dataclass
class TrendAnalysisResult:
    """分析结果数据类 (V2.0)"""
    code: str
    
    # 核心判断
    trend_status: TrendStatus = TrendStatus.CONSOLIDATION
    buy_signal: BuySignal = BuySignal.WAIT
    signal_score: int = 0
    
    # 关键点位
    current_price: float = 0.0
    stop_loss_price: float = 0.0  # 动态止损位 (基于ATR)
    pressure_price: float = 0.0   # 上方压力位
    
    # 详细分析
    signal_reasons: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    
    # 技术指标数据 (用于传递给 AI 解读)
    ma_alignment: str = "无"
    macd_status: str = "无"
    rsi_val: float = 0.0
    vol_status: str = "无"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'trend_status': self.trend_status.value,
            'buy_signal': self.buy_signal.value,
            'signal_score': self.signal_score,
            'current_price': self.current_price,
            'stop_loss_price': self.stop_loss_price,
            'signal_reasons': self.signal_reasons,
            'risk_factors': self.risk_factors,
            'indicators': {
                'rsi': self.rsi_val,
                'macd': self.macd_status,
                'ma': self.ma_alignment,
                'vol': self.vol_status
            }
        }

class StockTrendAnalyzer:
    """
    专业版趋势分析器
    """
    
    def analyze(self, df: pd.DataFrame, code: str) -> TrendAnalysisResult:
        result = TrendAnalysisResult(code=code)
        
        # 1. 数据校验 (至少需要60天数据计算 MA60 和 MACD)
        if df is None or df.empty or len(df) < 60:
            logger.warning(f"[{code}] 历史数据不足 (<60)，无法计算高级指标")
            result.risk_factors.append("数据不足，分析精度受限")
            return result
        
        # 按日期正序排列
        df = df.sort_values('date').reset_index(drop=True)
        
        # 2. 计算所有技术指标
        df = self._calc_ma(df)
        df = self._calc_macd(df)
        df = self._calc_rsi(df)
        df = self._calc_atr(df)
        
        # 获取最新一行数据
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        result.current_price = latest['close']
        result.rsi_val = round(latest['RSI'], 1)
        
        # 3. 综合逻辑判断
        score = 0
        reasons = []
        risks = []
        
        # --- A. 趋势判断 (MA) ---
        # 权重：40分
        ma5, ma10, ma20, ma60 = latest['MA5'], latest['MA10'], latest['MA20'], latest['MA60']
        
        if ma20 > ma60 and latest['close'] > ma20:
            if ma5 > ma10 > ma20:
                result.trend_status = TrendStatus.STRONG_BULL
                result.ma_alignment = "完美多头排列"
                score += 40
                reasons.append("✅ 均线呈完美多头排列，中期趋势向上")
            else:
                result.trend_status = TrendStatus.BULL
                result.ma_alignment = "震荡向上"
                score += 30
                reasons.append("✅ 位于20日线上方，趋势偏多")
        elif ma20 < ma60:
            result.trend_status = TrendStatus.BEAR
            result.ma_alignment = "空头趋势"
            risks.append("⚠️ 中期均线(MA20)死叉长期均线(MA60)")
        
        # --- B. 动能判断 (MACD) ---
        # 权重：30分
        dif, dea, macd = latest['DIF'], latest['DEA'], latest['MACD']
        prev_macd = prev['MACD']
        
        if dif > 0 and dea > 0:
            result.macd_status = "水上金叉区"
            if macd > 0 and macd > prev_macd:
                score += 30
                reasons.append("✅ MACD水上红柱放大，加速上涨迹象")
            elif macd > 0 and macd < prev_macd:
                score += 20
                risks.append("⚠️ MACD红柱缩短，上涨动能减弱")
            elif macd < 0: # 水上死叉回调
                score += 10
                reasons.append("✅ MACD水上回调，关注止跌机会")
        elif dif < 0:
            result.macd_status = "水下弱势区"
            if macd > 0:
                reasons.append("⚡ MACD水下金叉，存在反弹可能")
                score += 10
        
        # --- C. 情绪与买点 (RSI) ---
        # 权重：20分
        rsi = latest['RSI']
        
        if rsi > 80:
            risks.append(f"⚠️ RSI高达{rsi:.1f}，严重超买，随时可能回调")
            score -= 20 # 倒扣分
        elif rsi > 70:
            risks.append(f"⚠️ RSI进入超买区({rsi:.1f})，不宜追高")
        elif 40 < rsi < 60:
            score += 20
            reasons.append("✅ RSI处于50左右健康区间，上涨空间充足")
        elif rsi < 30:
            reasons.append("⚡ RSI超卖，存在短线反弹机会")
            score += 10
            
        # --- D. 止损位计算 (ATR) ---
        # 只有在看多时才计算止损
        atr = latest['ATR']
        # 策略：吊灯止损法 (Chandelier Exit) - 最高价回撤 2 倍 ATR
        recent_high = df['high'].iloc[-10:].max()
        stop_loss = recent_high - (2.5 * atr)
        
        # 如果当前价格已经低于止损位，说明趋势坏了
        if latest['close'] < stop_loss:
            score = 0
            risks.append(f"❌ 已跌破ATR动态止损位 ({stop_loss:.2f})")
        else:
            result.stop_loss_price = round(stop_loss, 2)
            # 如果是买入建议，加上止损提示
            if score > 60:
                reasons.append(f"🛡️ 建议止损位：{result.stop_loss_price} (基于2.5倍ATR)")

        # --- E. 最终信号生成 ---
        result.signal_score = score
        result.signal_reasons = reasons
        result.risk_factors = risks
        
        if score >= 85:
            result.buy_signal = BuySignal.STRONG_BUY
        elif score >= 65:
            result.buy_signal = BuySignal.BUY
        elif score >= 45:
            result.buy_signal = BuySignal.HOLD
        elif score < 30:
            result.buy_signal = BuySignal.SELL
        else:
            result.buy_signal = BuySignal.WAIT
            
        return result

    # --- 技术指标计算方法 (使用 pandas 实现，无需 TA-Lib) ---
    
    def _calc_ma(self, df):
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA60'] = df['close'].rolling(window=60).mean()
        return df

    def _calc_macd(self, df, fast=12, slow=26, signal=9):
        # 1. Calculate EMA (Exponential Moving Average)
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        # 2. Calculate DIF
        df['DIF'] = exp1 - exp2
        # 3. Calculate DEA
        df['DEA'] = df['DIF'].ewm(span=signal, adjust=False).mean()
        # 4. Calculate MACD Histogram
        df['MACD'] = 2 * (df['DIF'] - df['DEA'])
        return df

    def _calc_rsi(self, df, periods=14):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        # 填充 NaN
        df['RSI'] = df['RSI'].fillna(50)
        return df

    def _calc_atr(self, df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        df['ATR'] = true_range.rolling(window=period).mean()
        return df
