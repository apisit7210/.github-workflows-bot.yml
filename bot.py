"""
╔══════════════════════════════════════════════════════════╗
║        AI TRADING BOT ULTRA — GitHub Actions Edition     ║
║  XM Markets | MTF | ML Ensemble | Telegram | W/L Track  ║
╚══════════════════════════════════════════════════════════╝
Deploy: GitHub Actions (free, runs every 30 min)
"""

import os, json, time, warnings, requests
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from io import BytesIO
from scipy.stats import linregress
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import accuracy_score
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

TG_TOKEN = os.environ["TG_TOKEN"]
TG_CHAT  = os.environ["TG_CHAT_ID"]
TG_API   = f"https://api.telegram.org/bot{TG_TOKEN}"

SYMBOLS = {
    "EURUSD": "EURUSD=X",   "USDJPY": "USDJPY=X",
    "GBPUSD": "GBPUSD=X",   "GBPJPY": "GBPJPY=X",
    "AUDUSD": "AUDUSD=X",   "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "BTCUSD": "BTC-USD",    "ETHUSD": "ETH-USD",    "SOLUSD": "SOL-USD",
    "GOLD":   "GC=F",       "SILVER": "SI=F",        "OIL":    "CL=F",
    "US100":  "NQ=F",       "US30":   "YM=F",
    "JP225":  "^N225",      "GER40":  "^GDAXI",
}

MTF_CONFIG = {
    "fast":   {"interval": "15m", "period": "7d",  "weight": 0.25},
    "medium": {"interval": "1h",  "period": "30d", "weight": 0.45},
    "slow":   {"interval": "4h",  "period": "60d", "weight": 0.30},
}

ATR_SL_MULT = 1.8
ATR_TP_MULT = 3.2
MIN_CONF    = 0.62
MIN_ADX     = 20
MIN_RR      = 1.5
TOP_N       = 3

# GitHub Actions จะรัน script นี้ทุก 30 นาที
# เราต้องจำว่า cycle ปัจจุบันเป็นคี่หรือคู่
# โดยใช้ไฟล์ cycle.json เก็บ state

CYCLE_FILE  = "cycle.json"
TRADE_FILE  = "trades.json"
STATS_FILE  = "stats.json"

# ═══════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════

def tg_send(text):
    try:
        requests.post(f"{TG_API}/sendMessage", json={
            "chat_id": TG_CHAT,
            "text": text[:4096],
            "parse_mode": "HTML",
        }, timeout=15)
    except Exception as e:
        print(f"[TG] send error: {e}")


def tg_photo(buf, caption=""):
    try:
        buf.seek(0)
        requests.post(f"{TG_API}/sendPhoto", data={
            "chat_id": TG_CHAT,
            "caption": caption[:1024],
            "parse_mode": "HTML",
        }, files={"photo": ("chart.png", buf, "image/png")}, timeout=30)
    except Exception as e:
        print(f"[TG] photo error: {e}")


# ═══════════════════════════════════════════════════════════
# STATE / CYCLE
# ═══════════════════════════════════════════════════════════

def load_cycle():
    if os.path.exists(CYCLE_FILE):
        with open(CYCLE_FILE) as f:
            return json.load(f)
    return {"count": 0}


def save_cycle(data):
    with open(CYCLE_FILE, "w") as f:
        json.dump(data, f)


# ═══════════════════════════════════════════════════════════
# TRADE DATABASE
# ═══════════════════════════════════════════════════════════

def load_trades():
    if os.path.exists(TRADE_FILE):
        with open(TRADE_FILE) as f:
            return json.load(f)
    return []


def save_trades(trades):
    with open(TRADE_FILE, "w") as f:
        json.dump(trades[-300:], f, indent=2)


def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE) as f:
            return json.load(f)
    return {
        "total": 0, "wins": 0, "losses": 0,
        "total_pnl_pct": 0.0, "by_symbol": {},
        "streak": 0, "max_streak": 0,
    }


def save_stats(s):
    with open(STATS_FILE, "w") as f:
        json.dump(s, f, indent=2)


def add_trade(name, signal, conf, price, SL, TP, rr):
    trades = load_trades()
    tid = f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    trades.append({
        "id": tid, "name": name, "signal": signal,
        "conf": round(conf, 4), "entry": round(price, 6),
        "SL": round(SL, 6), "TP": round(TP, 6),
        "rr": round(rr, 2),
        "open_ts": datetime.utcnow().isoformat(),
        "close_ts": None, "outcome": None,
        "close_px": None, "pnl_pct": None, "updates": 0,
    })
    save_trades(trades)
    print(f"[DB] Trade logged: {tid}")
    return tid


def close_trade(tid, outcome, close_px):
    trades = load_trades()
    stats  = load_stats()
    for t in trades:
        if t["id"] == tid and t["outcome"] is None:
            entry = t["entry"]
            pnl   = (close_px - entry) / entry * 100
            if t["signal"] == "SELL":
                pnl = -pnl
            t.update({
                "outcome": outcome,
                "close_px": round(close_px, 6),
                "pnl_pct":  round(pnl, 4),
                "close_ts": datetime.utcnow().isoformat(),
            })
            stats["total"] += 1
            stats["total_pnl_pct"] = round(stats["total_pnl_pct"] + pnl, 4)
            sym = t["name"]
            if sym not in stats["by_symbol"]:
                stats["by_symbol"][sym] = {"total":0,"wins":0,"losses":0,"pnl_pct":0.0}
            stats["by_symbol"][sym]["total"] += 1
            stats["by_symbol"][sym]["pnl_pct"] = round(
                stats["by_symbol"][sym]["pnl_pct"] + pnl, 4)
            if outcome == "WIN":
                stats["wins"] += 1
                stats["by_symbol"][sym]["wins"] += 1
                stats["streak"] = max(0, stats.get("streak", 0)) + 1
            else:
                stats["losses"] += 1
                stats["by_symbol"][sym]["losses"] += 1
                stats["streak"] = min(0, stats.get("streak", 0)) - 1
            stats["max_streak"] = max(stats.get("max_streak", 0), stats["streak"])
            break
    save_trades(trades)
    save_stats(stats)


def get_open_trades():
    return [t for t in load_trades() if t["outcome"] is None]


def get_current_price(yf_symbol):
    try:
        df = yf.download(yf_symbol, period="1d", interval="1m",
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        return float(df["Close"].squeeze().iloc[-1])
    except:
        return None


def progress_bar(pct, width=12):
    filled = int(width * max(0, min(1, pct)))
    return "█" * filled + "░" * (width - filled)


# ═══════════════════════════════════════════════════════════
# TRADE UPDATE
# ═══════════════════════════════════════════════════════════

def check_and_update_trades():
    open_trades = get_open_trades()
    if not open_trades:
        print("[UPDATE] No open trades")
        return

    closed, still_open = [], []

    for t in open_trades:
        name   = t["name"]
        yf_sym = SYMBOLS.get(name)
        if not yf_sym:
            continue
        cur = get_current_price(yf_sym)
        if cur is None:
            print(f"[UPDATE] Cannot get price for {name}")
            continue

        entry, SL, TP, signal, tid = (
            t["entry"], t["SL"], t["TP"], t["signal"], t["id"])

        hit_tp = (signal=="BUY" and cur>=TP) or (signal=="SELL" and cur<=TP)
        hit_sl = (signal=="BUY" and cur<=SL) or (signal=="SELL" and cur>=SL)

        if hit_tp:
            close_trade(tid, "WIN", cur)
            closed.append(("WIN", t, cur))
            continue
        if hit_sl:
            close_trade(tid, "LOSS", cur)
            closed.append(("LOSS", t, cur))
            continue

        total_range = abs(TP - entry)
        moved       = (cur - entry) if signal=="BUY" else (entry - cur)
        to_tp_pts   = abs(TP - cur)
        to_tp_pct   = to_tp_pts / cur * 100
        pnl_now     = moved / entry * 100
        progress    = max(0, min(1, moved / total_range if total_range > 0 else 0))

        open_dt  = datetime.fromisoformat(t["open_ts"])
        hold_min = int((datetime.utcnow() - open_dt).total_seconds() / 60)
        hold_str = f"{hold_min//60}h {hold_min%60}m" if hold_min >= 60 else f"{hold_min}m"

        t["updates"] = t.get("updates", 0) + 1
        still_open.append({
            "t": t, "cur": cur,
            "to_tp_pct": to_tp_pct, "to_tp_pts": to_tp_pts,
            "pnl": pnl_now, "progress": progress, "hold": hold_str,
        })

    # ── WIN/LOSS notifications ───────────────────────────────────────
    for outcome, t, cpx in closed:
        pnl   = (cpx - t["entry"]) / t["entry"] * 100
        if t["signal"] == "SELL":
            pnl = -pnl
        stats = load_stats()
        wr    = stats["wins"] / stats["total"] * 100 if stats["total"] > 0 else 0

        if outcome == "WIN":
            msg = (
                f"🏆 <b>TRADE WIN!</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>{t['name']}</b>  {t['signal']}\n"
                f"✅ TP Hit!\n"
                f"💰 Entry  : <code>{t['entry']:.5f}</code>\n"
                f"🎯 Close  : <code>{cpx:.5f}</code>\n"
                f"📈 P&amp;L : <b>+{abs(pnl):.2f}%</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🏆 WR: <b>{wr:.0f}%</b>  ({stats['wins']}W/{stats['losses']}L)\n"
                f"💼 Total P&amp;L: <b>{stats['total_pnl_pct']:+.2f}%</b>\n"
                f"🔥 Streak: {stats['streak']} wins"
            )
        else:
            msg = (
                f"💀 <b>TRADE LOSS</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>{t['name']}</b>  {t['signal']}\n"
                f"❌ SL Hit\n"
                f"💰 Entry  : <code>{t['entry']:.5f}</code>\n"
                f"🛑 Close  : <code>{cpx:.5f}</code>\n"
                f"📉 P&amp;L : <b>{pnl:.2f}%</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🏆 WR: <b>{wr:.0f}%</b>  ({stats['wins']}W/{stats['losses']}L)\n"
                f"💼 Total P&amp;L: <b>{stats['total_pnl_pct']:+.2f}%</b>\n"
                f"📉 Streak: {abs(stats['streak'])} losses"
            )
        tg_send(msg)
        time.sleep(0.5)

    # ── Update สำหรับ trade ที่ยังเปิดอยู่ ──────────────────────────
    if still_open:
        ts  = datetime.utcnow().strftime("%H:%M UTC")
        lines = [f"📡 <b>Trade Update</b>  <code>{ts}</code>\n"]
        for item in still_open:
            t   = item["t"]
            bar = progress_bar(item["progress"])
            pnl_icon = "🟢" if item["pnl"] >= 0 else "🔴"
            sig_icon = "🟢" if t["signal"] == "BUY" else "🔴"
            lines.append(
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{sig_icon} <b>{t['name']}</b>  {t['signal']}\n"
                f"💰 Entry   : <code>{t['entry']:.5f}</code>\n"
                f"📍 Now     : <code>{item['cur']:.5f}</code>\n"
                f"🎯 TP      : <code>{t['TP']:.5f}</code>\n"
                f"   เหลือ   : <b>{item['to_tp_pct']:.2f}%</b>"
                f" ({item['to_tp_pts']:.5f} pts)\n"
                f"🛑 SL      : <code>{t['SL']:.5f}</code>\n"
                f"{pnl_icon} P&amp;L  : <b>{item['pnl']:+.2f}%</b>\n"
                f"📊 Progress: [{bar}] {item['progress']*100:.0f}%\n"
                f"⏱ Hold    : {item['hold']}  (#{t['updates']} updates)"
            )
        tg_send("\n".join(lines))

        # save updated trade counts
        trades = load_trades()
        umap   = {item["t"]["id"]: item["t"] for item in still_open}
        for i, t in enumerate(trades):
            if t["id"] in umap:
                trades[i] = umap[t["id"]]
        save_trades(trades)


# ═══════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════

def fetch_data(yf_symbol, interval="1h", period="30d"):
    df = yf.download(yf_symbol, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data: {yf_symbol}")
    df.columns = df.columns.get_level_values(0)
    for col in ["Open","High","Low","Close","Volume"]:
        if col in df.columns:
            df[col] = df[col].squeeze()
    df.dropna(inplace=True)
    return df


def compute_indicators(df):
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df.get("Volume", pd.Series(np.ones(len(df)), index=df.index))
    if hasattr(v, "squeeze"):
        v = v.squeeze()

    for w in [9, 20, 50, 100, 200]:
        df[f"EMA{w}"] = ta.trend.ema_indicator(c, window=w)

    adx = ta.trend.ADXIndicator(h, l, c, window=14)
    df["ADX"]     = adx.adx()
    df["ADX_pos"] = adx.adx_pos()
    df["ADX_neg"] = adx.adx_neg()

    macd = ta.trend.MACD(c)
    df["MACD"]        = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"]   = macd.macd_diff()

    ichi = ta.trend.IchimokuIndicator(h, l)
    df["ICHI_conv"]   = ichi.ichimoku_conversion_line()
    df["ICHI_base"]   = ichi.ichimoku_base_line()
    df["ICHI_span_a"] = ichi.ichimoku_a()
    df["ICHI_span_b"] = ichi.ichimoku_b()

    psar = ta.trend.PSARIndicator(h, l, c)
    df["PSAR_bull"] = psar.psar_up_indicator()
    df["PSAR_bear"] = psar.psar_down_indicator()

    df["RSI"]   = ta.momentum.rsi(c, window=14)
    df["RSI7"]  = ta.momentum.rsi(c, window=7)
    df["RSI21"] = ta.momentum.rsi(c, window=21)

    stoch = ta.momentum.StochasticOscillator(h, l, c)
    df["STOCH_k"] = stoch.stoch()
    df["STOCH_d"] = stoch.stoch_signal()

    df["ROC10"] = ta.momentum.roc(c, window=10)
    df["WILLR"] = ta.momentum.williams_r(h, l, c)
    df["CCI"]   = ta.trend.CCIIndicator(h, l, c).cci()
    df["TSI"]   = ta.momentum.TSIIndicator(c).tsi()

    bb = ta.volatility.BollingerBands(c)
    df["BB_high"]  = bb.bollinger_hband()
    df["BB_low"]   = bb.bollinger_lband()
    df["BB_mid"]   = bb.bollinger_mavg()
    df["BB_width"] = (df["BB_high"] - df["BB_low"]) / df["BB_mid"]
    df["BB_pct"]   = bb.bollinger_pband()

    atr_obj = ta.volatility.AverageTrueRange(h, l, c)
    df["ATR"]     = atr_obj.average_true_range()
    df["ATR_pct"] = df["ATR"] / c * 100

    kc = ta.volatility.KeltnerChannel(h, l, c)
    df["KC_high"] = kc.keltner_channel_hband()
    df["KC_low"]  = kc.keltner_channel_lband()
    df["SQUEEZE"] = (df["BB_high"] < df["KC_high"]) & (df["BB_low"] > df["KC_low"])

    df["DC_high"] = h.rolling(20).max()
    df["DC_low"]  = l.rolling(20).min()

    df["OBV"]  = ta.volume.on_balance_volume(c, v)
    df["CMF"]  = ta.volume.chaikin_money_flow(h, l, c, v)
    df["MFI"]  = ta.volume.money_flow_index(h, l, c, v)
    df["VWAP"] = (((h+l+c)/3) * v).cumsum() / v.cumsum()
    df["VOL_MA"]    = v.rolling(20).mean()
    df["VOL_RATIO"] = v / df["VOL_MA"].replace(0, np.nan)

    n = min(55, len(df))
    hi, lo = df["High"].rolling(n).max(), df["Low"].rolling(n).min()
    rng = hi - lo
    for lvl, pct in [("382",0.382),("500",0.500),("618",0.618)]:
        df[f"FIB_{lvl}"] = hi - pct * rng

    def hurst(ts, lag=20):
        if len(ts) < lag*2: return 0.5
        lags = range(2, lag)
        tau  = [np.sqrt(np.std(ts[l:]-ts[:-l])) for l in lags]
        try:
            s, *_ = linregress(np.log(list(lags)), np.log(tau))
            return s * 2.0
        except: return 0.5

    df["HURST"] = c.rolling(40).apply(lambda x: hurst(x.values), raw=False)

    def lr_slope(series, n=20):
        out = [np.nan]*n
        for i in range(n, len(series)):
            y = series.iloc[i-n:i].values
            s, *_ = linregress(np.arange(n), y)
            out.append(s / (y.mean()+1e-10) * 100)
        return pd.Series(out, index=series.index)

    df["LR_SLOPE"] = lr_slope(c)
    atr_ma = df["ATR"].rolling(50).mean()
    df["VOL_REGIME"] = np.where(df["ATR"]>atr_ma*1.3, 2,
                        np.where(df["ATR"]<atr_ma*0.7, 0, 1))
    df.dropna(inplace=True)
    return df


def tf_bias(df):
    try:
        df = compute_indicators(df)
        last, prev = df.iloc[-1], df.iloc[-2]
        b = s = 0
        if last["EMA9"]>last["EMA20"]>last["EMA50"]:   b += 2
        elif last["EMA9"]<last["EMA20"]<last["EMA50"]: s += 2
        if last["Close"]>last["EMA200"]: b += 1
        else:                             s += 1
        if last["ADX"]>MIN_ADX:
            if last["ADX_pos"]>last["ADX_neg"]: b += 1
            else:                                s += 1
        if last["MACD"]>last["MACD_signal"]:  b += 1
        elif last["MACD"]<last["MACD_signal"]: s += 1
        ct = max(last["ICHI_span_a"], last["ICHI_span_b"])
        cb = min(last["ICHI_span_a"], last["ICHI_span_b"])
        if last["Close"]>ct:   b += 1
        elif last["Close"]<cb: s += 1
        total = b + s
        if total == 0: return "NEUTRAL", 0.5, df
        return ("BUY" if b>=s else "SELL"), max(b,s)/total, df
    except:
        return "NEUTRAL", 0.5, df


# ═══════════════════════════════════════════════════════════
# ML
# ═══════════════════════════════════════════════════════════

FEAT_COLS = ["RSI","RSI7","RSI21","MACD_hist","STOCH_k","STOCH_d",
             "BB_pct","BB_width","ATR_pct","ADX","ADX_pos","ADX_neg",
             "ROC10","WILLR","CCI","TSI","CMF","MFI","VOL_RATIO",
             "LR_SLOPE","HURST","VOL_REGIME"]


def train_ml(df):
    d = df.copy()
    d["lbl"] = ((d["Close"].shift(-5)/d["Close"]-1) > 0.005).astype(int)
    d.dropna(inplace=True)
    fc = [c for c in FEAT_COLS if c in d.columns]
    X, y = d[fc].values, d["lbl"].values
    if len(X) < 80:
        return None, None, 0.5, fc
    scaler = StandardScaler()
    rf  = RandomForestClassifier(150, max_depth=6, min_samples_leaf=5,
                                  random_state=42, n_jobs=-1)
    gbm = GradientBoostingClassifier(100, max_depth=4, learning_rate=0.05,
                                      random_state=42)
    vc  = VotingClassifier([("rf",rf),("gbm",gbm)], voting="soft")
    tscv = TimeSeriesSplit(4)
    accs = []
    for tr, te in tscv.split(X):
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        vc.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], vc.predict(Xte)))
    vc.fit(scaler.fit_transform(X), y)
    return vc, scaler, float(np.mean(accs)), fc


# ═══════════════════════════════════════════════════════════
# SCORING + RISK + BACKTEST
# ═══════════════════════════════════════════════════════════

def rule_score(df):
    last, prev, prev2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    b = s = 0; rb = []; rs = []
    def bv(p, r): nonlocal b; b+=p; rb.append(r)
    def sv(p, r): nonlocal s; s+=p; rs.append(r)

    if last["EMA9"]>last["EMA20"]>last["EMA50"]>last["EMA100"]: bv(3,"EMA full bull stack")
    elif last["EMA9"]<last["EMA20"]<last["EMA50"]<last["EMA100"]: sv(3,"EMA full bear stack")
    elif last["EMA9"]>last["EMA20"]>last["EMA50"]: bv(2,"EMA bull stack")
    elif last["EMA9"]<last["EMA20"]<last["EMA50"]: sv(2,"EMA bear stack")
    if prev["EMA20"]<=prev["EMA50"] and last["EMA20"]>last["EMA50"]: bv(2,"Golden Cross")
    elif prev["EMA20"]>=prev["EMA50"] and last["EMA20"]<last["EMA50"]: sv(2,"Death Cross")
    adx=last["ADX"]
    if adx>30:
        if last["ADX_pos"]>last["ADX_neg"]: bv(2,f"ADX={adx:.0f} very strong up")
        else:                                sv(2,f"ADX={adx:.0f} very strong dn")
    elif adx>MIN_ADX:
        if last["ADX_pos"]>last["ADX_neg"]: bv(1,f"ADX={adx:.0f} uptrend")
        else:                                sv(1,f"ADX={adx:.0f} downtrend")
    if prev["MACD"]<prev["MACD_signal"] and last["MACD"]>last["MACD_signal"]: bv(2,"MACD cross up")
    elif prev["MACD"]>prev["MACD_signal"] and last["MACD"]<last["MACD_signal"]: sv(2,"MACD cross dn")
    if last["MACD_hist"]>0 and last["MACD_hist"]>prev["MACD_hist"]>prev2["MACD_hist"]: bv(1,"MACD accel up")
    elif last["MACD_hist"]<0 and last["MACD_hist"]<prev["MACD_hist"]<prev2["MACD_hist"]: sv(1,"MACD accel dn")
    rsi=last["RSI"]
    if rsi<25: bv(3,f"RSI deeply OS ({rsi:.0f})")
    elif rsi<35: bv(2,f"RSI OS ({rsi:.0f})")
    elif rsi>75: sv(3,f"RSI deeply OB ({rsi:.0f})")
    elif rsi>65: sv(2,f"RSI OB ({rsi:.0f})")
    if last["RSI7"]>last["RSI21"] and prev["RSI7"]<=prev["RSI21"]: bv(1,"RSI7 cross up")
    elif last["RSI7"]<last["RSI21"] and prev["RSI7"]>=prev["RSI21"]: sv(1,"RSI7 cross dn")
    if last["Close"]<last["BB_low"]:  bv(2,"Below BB lower")
    elif last["Close"]>last["BB_high"]: sv(2,"Above BB upper")
    if last.get("SQUEEZE",False) and not prev.get("SQUEEZE",False):
        if last.get("LR_SLOPE",0)>0: bv(2,"Squeeze breakout bull")
        else:                          sv(2,"Squeeze breakout bear")
    if last["Close"]>=last["DC_high"]:  bv(2,"Donchian breakout")
    elif last["Close"]<=last["DC_low"]: sv(2,"Donchian breakdown")
    if last["STOCH_k"]<20 and last["STOCH_k"]>last["STOCH_d"]: bv(2,f"Stoch cross OS ({last['STOCH_k']:.0f})")
    elif last["STOCH_k"]>80 and last["STOCH_k"]<last["STOCH_d"]: sv(2,f"Stoch cross OB ({last['STOCH_k']:.0f})")
    ct=max(last["ICHI_span_a"],last["ICHI_span_b"]); cb=min(last["ICHI_span_a"],last["ICHI_span_b"])
    if last["Close"]>ct:   bv(2,"Above Ichimoku cloud")
    elif last["Close"]<cb: sv(2,"Below Ichimoku cloud")
    if prev["ICHI_conv"]<=prev["ICHI_base"] and last["ICHI_conv"]>last["ICHI_base"]: bv(2,"TK cross up")
    elif prev["ICHI_conv"]>=prev["ICHI_base"] and last["ICHI_conv"]<last["ICHI_base"]: sv(2,"TK cross dn")
    if last.get("PSAR_bull")==1: bv(1,"PSAR flip bull")
    elif last.get("PSAR_bear")==1: sv(1,"PSAR flip bear")
    if last["Close"]>last["VWAP"]*1.002: bv(1,"Above VWAP")
    elif last["Close"]<last["VWAP"]*0.998: sv(1,"Below VWAP")
    if last.get("VOL_RATIO",1)>1.8:
        if last.get("CMF",0)>0.1:    bv(2,"Vol spike+CMF pos")
        elif last.get("CMF",0)<-0.1: sv(2,"Vol spike+CMF neg")
    if last.get("MFI",50)<20: bv(1,f"MFI OS ({last.get('MFI',50):.0f})")
    elif last.get("MFI",50)>80: sv(1,f"MFI OB ({last.get('MFI',50):.0f})")
    if last.get("TSI",0)>0 and prev.get("TSI",0)<=0: bv(1,"TSI cross 0 up")
    elif last.get("TSI",0)<0 and prev.get("TSI",0)>=0: sv(1,"TSI cross 0 dn")
    if last.get("CCI",0)<-150: bv(1,f"CCI OS")
    elif last.get("CCI",0)>150: sv(1,f"CCI OB")
    if last.get("WILLR",0)<-90: bv(1,f"WR OS")
    elif last.get("WILLR",0)>-10: sv(1,f"WR OB")
    for lvl in ["618","500","382"]:
        key=f"FIB_{lvl}"
        if key in last.index and abs(last["Close"]-last[key])/last["Close"]<0.003:
            if b>s: bv(1,f"Near Fib {lvl}")
            else:   sv(1,f"Near Fib {lvl}")
            break
    regime_ok = last.get("HURST", 0.5) > 0.45
    total = b + s
    if total==0: return "NEUTRAL",0.5,0,0,[],regime_ok
    signal = "BUY" if b>=s else "SELL"
    return signal, max(b,s)/total, b, s, (rb if signal=="BUY" else rs), regime_ok


def compute_risk(df, signal, conf):
    last  = df.iloc[-1]
    price = float(last["Close"])
    atr   = float(last["ATR"])
    vr    = float(last.get("VOL_REGIME", 1))
    sl_m  = ATR_SL_MULT * (1+vr*0.1)
    tp_m  = ATR_TP_MULT * (1+vr*0.05)
    SL = price - sl_m*atr if signal=="BUY" else price + sl_m*atr
    TP = price + tp_m*atr if signal=="BUY" else price - tp_m*atr
    risk = abs(price-SL); rw = abs(price-TP)
    rr   = rw/risk if risk>0 else 0
    kelly = max(0, min((rr*conf-(1-conf))/rr if rr>0 else 0, 0.25))*100
    est_h = rw/atr
    return price, SL, TP, rr, kelly, atr, est_h


def backtest(df, n=100):
    d = df.copy().tail(n)
    d["sig"] = np.where(d["EMA9"]>d["EMA20"],1,-1)
    d["ret"] = d["Close"].pct_change()
    d["sr"]  = d["sig"].shift(1)*d["ret"]
    d["cum"] = (1+d["sr"]).cumprod()
    pf_n = d["sr"][d["sr"]>0].sum()
    pf_d = abs(d["sr"][d["sr"]<0].sum())
    return {
        "win_rate":     (d["sr"]>0).sum()/len(d)*100,
        "total_return": (d["cum"].iloc[-1]-1)*100,
        "sharpe":       d["sr"].mean()/d["sr"].std()*np.sqrt(252*24) if d["sr"].std()>0 else 0,
        "max_dd":       (d["cum"]/d["cum"].cummax()-1).min()*100,
        "profit_factor": pf_n/pf_d if pf_d>0 else 99,
    }


# ═══════════════════════════════════════════════════════════
# CHART
# ═══════════════════════════════════════════════════════════

def make_chart(df, name, signal, SL, TP, price, conf):
    df_p = df.tail(120).copy()
    idx  = df_p.index
    clr  = "#00ff88" if signal=="BUY" else "#ff4466"

    fig = plt.figure(figsize=(14,10))
    fig.patch.set_facecolor("#090e1a")
    gs  = gridspec.GridSpec(4,1, height_ratios=[5,1.5,1.5,1], hspace=0.06)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax in axes:
        ax.set_facecolor("#090e1a")
        ax.tick_params(colors="#556b8b", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#1e2d45")

    ax1 = axes[0]
    ax1.fill_between(idx, df_p["ICHI_span_a"], df_p["ICHI_span_b"],
        where=df_p["ICHI_span_a"]>=df_p["ICHI_span_b"], alpha=0.1, color="#00ff88")
    ax1.fill_between(idx, df_p["ICHI_span_a"], df_p["ICHI_span_b"],
        where=df_p["ICHI_span_a"]<df_p["ICHI_span_b"],  alpha=0.1, color="#ff4466")
    ax1.fill_between(idx, df_p["BB_high"], df_p["BB_low"], alpha=0.05, color="#58a6ff")
    for ema,col,lw in [("EMA9","#ffd700",1.0),("EMA20","#58a6ff",0.9),
                        ("EMA50","#ff8c00",0.8),("EMA200","#cc44ff",0.7)]:
        if ema in df_p.columns:
            ax1.plot(idx, df_p[ema], color=col, lw=lw, label=ema, alpha=0.85)
    ax1.plot(idx, df_p["VWAP"], color="#cc44ff", lw=0.7, ls=":", label="VWAP", alpha=0.7)
    ax1.plot(idx, df_p["Close"], color="#e0e8ff", lw=1.4, label="Price")
    ax1.axhline(SL, color="#ff4466", lw=0.9, ls="--")
    ax1.axhline(TP, color="#00ff88", lw=0.9, ls="--")
    ax1.scatter(idx[-1], price, color=clr, s=150, zorder=10)
    ax1.text(idx[-1], SL, f"  SL {SL:.5f}", color="#ff4466", fontsize=7, va="center")
    ax1.text(idx[-1], TP, f"  TP {TP:.5f}", color="#00ff88", fontsize=7, va="center")
    ax1.set_title(f"{name}  {'🟢 BUY' if signal=='BUY' else '🔴 SELL'}  Conf:{conf*100:.1f}%",
                  color=clr, fontsize=11, fontweight="bold", pad=6)
    ax1.legend(loc="upper left", fontsize=7, facecolor="#0d1a2e",
               edgecolor="#1e2d45", labelcolor="#8fa0b8", ncol=5)
    ax1.set_xlim(idx[0], idx[-1])

    ax2 = axes[1]
    ax2.plot(idx, df_p["RSI"],  color="#58a6ff", lw=1)
    ax2.plot(idx, df_p["RSI7"], color="#ffd700", lw=0.7, ls="--", alpha=0.7)
    ax2.axhline(70, color="#ff4466", lw=0.5, ls="--")
    ax2.axhline(30, color="#00ff88", lw=0.5, ls="--")
    ax2.fill_between(idx,df_p["RSI"],70,where=df_p["RSI"]>=70,alpha=0.12,color="#ff4466")
    ax2.fill_between(idx,df_p["RSI"],30,where=df_p["RSI"]<=30,alpha=0.12,color="#00ff88")
    ax2.set_ylabel("RSI", color="#556b8b", fontsize=8)
    ax2.set_ylim(0,100); ax2.set_xlim(idx[0], idx[-1])

    ax3 = axes[2]
    ax3.plot(idx, df_p["MACD"],        color="#58a6ff", lw=0.9)
    ax3.plot(idx, df_p["MACD_signal"], color="#ff8c00", lw=0.9)
    hist = df_p["MACD_hist"]
    ax3.bar(idx, hist, color=np.where(hist>=0,"#00ff88","#ff4466"), alpha=0.6, width=0.02)
    ax3.axhline(0, color="#1e2d45", lw=0.5)
    ax3.set_ylabel("MACD", color="#556b8b", fontsize=8)
    ax3.set_xlim(idx[0], idx[-1])

    ax4 = axes[3]
    if "Volume" in df_p.columns and "Open" in df_p.columns:
        vc = ["#00ff88" if df_p["Close"].iloc[i]>=df_p["Open"].iloc[i]
              else "#ff4466" for i in range(len(df_p))]
        ax4.bar(idx, df_p["Volume"], color=vc, alpha=0.7, width=0.02)
        if "VOL_MA" in df_p.columns:
            ax4.plot(idx, df_p["VOL_MA"], color="#ffd700", lw=0.8)
    ax4.set_ylabel("Vol", color="#556b8b", fontsize=8)
    ax4.set_xlim(idx[0], idx[-1])

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130,
                bbox_inches="tight", facecolor="#090e1a")
    plt.close()
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════
# ANALYZE
# ═══════════════════════════════════════════════════════════

def analyze(name, yf_symbol):
    print(f"  [SCAN] {name}...", end=" ", flush=True)
    try:
        mtf_biases = {}
        mtf_dfs    = {}
        for tf, cfg in MTF_CONFIG.items():
            try:
                raw = fetch_data(yf_symbol, cfg["interval"], cfg["period"])
                bias, strength, df_tf = tf_bias(raw)
                mtf_biases[tf] = (bias, strength)
                mtf_dfs[tf]    = df_tf
            except Exception as e:
                print(f"[MTF {tf}] {e}")
                mtf_biases[tf] = ("NEUTRAL", 0.5)

        wb = ws = 0
        for tf, cfg in MTF_CONFIG.items():
            b, st = mtf_biases.get(tf, ("NEUTRAL",0.5))
            if b=="BUY":   wb += cfg["weight"]*st
            elif b=="SELL": ws += cfg["weight"]*st
        mtf_signal = "BUY" if wb>=ws else "SELL"
        tot = wb+ws
        mtf_conf  = max(wb,ws)/tot if tot>0 else 0.5
        alignment = sum(v[0]==mtf_signal for v in mtf_biases.values())

        df_med = mtf_dfs.get("medium")
        if df_med is None or len(df_med) < 100:
            print("skip (data)")
            return None

        adx_val = float(df_med["ADX"].iloc[-1])
        if adx_val < MIN_ADX:
            print(f"skip ADX={adx_val:.0f}")
            return None

        model, scaler, ml_acc, feat_cols = train_ml(df_med)
        if model is not None:
            x  = scaler.transform(df_med[feat_cols].iloc[-1:].values)
            ml_p = float(model.predict_proba(x)[0][1])
        else:
            ml_p = 0.5

        rule_sig, rule_ratio, buy_sc, sell_sc, reasons, regime_ok = rule_score(df_med)
        ml_dir  = ml_p if rule_sig=="BUY" else (1-ml_p)
        mtf_dir = mtf_conf if mtf_signal==rule_sig else (1-mtf_conf)
        conf = min(0.97,
            (0.40*rule_ratio + 0.30*ml_dir + 0.30*mtf_dir) *
            (1.0 if regime_ok else 0.85)
        )
        votes  = [rule_sig, mtf_signal, ("BUY" if ml_p>0.5 else "SELL")]
        signal = max(set(votes), key=votes.count)

        if conf < MIN_CONF:
            print(f"skip conf={conf:.2f}")
            return None

        price, SL, TP, rr, kelly, atr, est_h = compute_risk(df_med, signal, conf)
        if rr < MIN_RR:
            print(f"skip RR={rr:.1f}")
            return None

        bt = backtest(df_med)
        print(f"✅ {signal} conf={conf:.2f} RR={rr:.1f}")
        return {
            "name": name, "yf_sym": yf_symbol,
            "signal": signal, "conf": conf,
            "price": price, "SL": SL, "TP": TP,
            "rr": rr, "kelly": kelly, "est_h": est_h,
            "buy_sc": buy_sc, "sell_sc": sell_sc,
            "adx": adx_val, "ml_p": ml_p, "ml_acc": ml_acc,
            "mtf_conf": mtf_conf, "alignment": alignment,
            "mtf_biases": mtf_biases, "reasons": reasons,
            "bt": bt, "df_med": df_med,
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"ERROR: {e}")
        return None


def send_signal(r):
    name, signal, conf = r["name"], r["signal"], r["conf"]
    price, SL, TP, rr  = r["price"], r["SL"], r["TP"], r["rr"]
    bt, mt = r["bt"], r["mtf_biases"]
    emoji  = "🟢" if signal=="BUY" else "🔴"
    conf_lbl = (
        "🔥🔥 VERY HIGH" if conf>0.82 else
        "🔥 HIGH"        if conf>0.72 else "⚠️ MEDIUM"
    )
    mtf_str = "  ".join(
        f"{k.upper()}:{'🟢' if v[0]=='BUY' else '🔴'}"
        for k,v in mt.items()
    )
    reasons_str = "\n".join(f"  • {x}" for x in r["reasons"][:5])
    stats   = load_stats()
    wr_all  = stats["wins"]/stats["total"]*100 if stats["total"]>0 else 0

    msg = (
        f"{emoji} <b>NEW SIGNAL: {name}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 Direction : <b>{signal}</b>  {conf_lbl}\n"
        f"💰 Entry     : <code>{price:.5f}</code>\n"
        f"🛑 SL        : <code>{SL:.5f}</code>\n"
        f"🎯 TP        : <code>{TP:.5f}</code>\n"
        f"📐 R:R       : 1:{rr:.2f}\n"
        f"💼 Kelly     : {r['kelly']:.1f}% of capital\n"
        f"⏱ Est. hold : ~{r['est_h']:.0f}h\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 Confidence: <b>{conf*100:.1f}%</b>\n"
        f"   ML:{r['ml_p']*100:.0f}%  MTF:{r['mtf_conf']*100:.0f}%  (align {r['alignment']}/3)\n"
        f"📊 Score: BUY={r['buy_sc']:.0f}  SELL={r['sell_sc']:.0f}  ADX={r['adx']:.0f}\n"
        f"🌐 MTF: {mtf_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ Reasons:\n{reasons_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📉 Backtest (100 bars):\n"
        f"   WR:{bt['win_rate']:.0f}%  PF:{bt['profit_factor']:.2f}"
        f"  Sharpe:{bt['sharpe']:.2f}\n"
        f"   Return:{bt['total_return']:.1f}%  MaxDD:{bt['max_dd']:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏆 Bot WR: <b>{wr_all:.0f}%</b>  P&amp;L:{stats['total_pnl_pct']:+.2f}%"
    )

    buf = make_chart(r["df_med"], name, signal, SL, TP, price, conf)
    tg_photo(buf, caption=f"{emoji} {name} {signal} | Conf:{conf*100:.0f}% | TP:{TP:.5f}")
    time.sleep(0.5)
    tg_send(msg)
    add_trade(name, signal, conf, price, SL, TP, rr)


def send_stats():
    stats = load_stats()
    if stats["total"] == 0:
        tg_send("📊 ยังไม่มี trade ที่ปิดแล้ว")
        return
    wr  = stats["wins"]/stats["total"]*100
    avg = stats["total_pnl_pct"]/stats["total"]
    by_sym = stats.get("by_symbol", {})
    sym_lines = []
    for sym, d in sorted(by_sym.items(),
                         key=lambda x: x[1]["pnl_pct"], reverse=True)[:5]:
        sw = d["wins"]/d["total"]*100 if d["total"]>0 else 0
        sym_lines.append(f"  {sym}: {d['wins']}W/{d['losses']}L  WR:{sw:.0f}%  P&L:{d['pnl_pct']:+.2f}%")
    bar = progress_bar(wr/100, width=15)
    tg_send(
        f"📈 <b>Performance Report</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Total : <b>{stats['total']}</b>  ({stats['wins']}✅  {stats['losses']}❌)\n"
        f"🏆 WR    : <b>{wr:.1f}%</b>\n"
        f"   [{bar}]\n"
        f"💰 Total P&amp;L: <b>{stats['total_pnl_pct']:+.2f}%</b>\n"
        f"📐 Avg/trade   : <b>{avg:+.2f}%</b>\n"
        f"🔥 Best streak : {stats['max_streak']} wins\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏅 Top Symbols:\n" + "\n".join(sym_lines)
    )


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    cycle_data  = load_cycle()
    cycle_count = cycle_data.get("count", 0) + 1
    save_cycle({"count": cycle_count})

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*55}")
    print(f"[BOT] Cycle #{cycle_count}  |  {ts}")
    print(f"{'='*55}")

    # ── Step 1: Update open trades (every cycle) ─────────────────
    print("[UPDATE] Checking open trades...")
    check_and_update_trades()

    # ── Step 2: Scan new signals (every 2 cycles = every 60 min) ─
    if cycle_count % 2 == 1:
        print("[SCAN] Scanning for new signals...")
        results = []
        for name, yf_sym in SYMBOLS.items():
            r = analyze(name, yf_sym)
            if r:
                results.append(r)
            time.sleep(1.0)

        if results:
            results.sort(key=lambda x: x["conf"]*x["rr"], reverse=True)
            for r in results[:TOP_N]:
                send_signal(r)
                time.sleep(1.0)
        else:
            tg_send(f"🔍 Cycle #{cycle_count}: ไม่พบสัญญาณที่ผ่านเกณฑ์")

    # ── Step 3: Stats report every 8 cycles (~4h) ─────────────────
    if cycle_count % 8 == 0:
        send_stats()

    print(f"[BOT] Cycle #{cycle_count} complete")


if __name__ == "__main__":
    main()
