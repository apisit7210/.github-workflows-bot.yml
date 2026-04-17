# 🤖 AI Trading Bot ULTRA

> XM Markets | Multi-Timeframe | ML Ensemble | Telegram | GitHub Actions (Free 24/7)

---

## 📁 โครงสร้างไฟล์

```
trading_bot/
├── bot.py                          # โค้ดหลัก
├── requirements.txt                # Python packages
├── .github/
│   └── workflows/
│       └── trading_bot.yml         # GitHub Actions schedule
├── trades.json                     # (auto-created) trade log
├── stats.json                      # (auto-created) W/L stats
└── cycle.json                      # (auto-created) cycle counter
```

---

## 🚀 วิธี Deploy (ทำครั้งเดียว ~10 นาที)

### Step 1 — สร้าง Telegram Bot

```
1. เปิด Telegram → ค้นหา @BotFather
2. พิมพ์ /newbot
3. ตั้งชื่อ เช่น: XM Trading Signal Bot
4. รับ Token เช่น: 7123456789:AAEabcDEFghiJKLmnoPQRstUVwxYZ
5. เปิด chat กับ bot แล้วกด Start
6. เปิด URL นี้ใน browser:
   https://api.telegram.org/bot<TOKEN>/getUpdates
7. หา "id" ใน "chat" → นั่นคือ Chat ID
```

---

### Step 2 — สร้าง GitHub Repository

```
1. ไปที่ github.com → New repository
2. ตั้งชื่อ เช่น: ai-trading-bot
3. เลือก Private (แนะนำ)
4. กด Create repository
```

---

### Step 3 — อัปโหลดไฟล์

**วิธีที่ 1: ผ่านเว็บ GitHub (ง่ายที่สุด)**
```
1. กด "uploading an existing file" ใน repo
2. ลาก bot.py และ requirements.txt ขึ้นไป
3. สร้างโฟลเดอร์ .github/workflows/ แล้วอัปโหลด trading_bot.yml
```

**วิธีที่ 2: ผ่าน Git CLI**
```bash
git clone https://github.com/YOUR_USERNAME/ai-trading-bot.git
cd ai-trading-bot

# copy ไฟล์ทั้งหมดมาที่นี่
git add .
git commit -m "🤖 Initial trading bot"
git push origin main
```

---

### Step 4 — ตั้ง Secrets (สำคัญมาก!)

```
1. ไปที่ repo → Settings → Secrets and variables → Actions
2. กด "New repository secret"
3. เพิ่ม 2 secrets:

   Name: TG_TOKEN
   Value: 7123456789:AAEabcDEFghiJKLmnoPQRstUVwxYZ

   Name: TG_CHAT_ID
   Value: 123456789
```

---

### Step 5 — เปิดใช้งาน Actions

```
1. ไปที่ tab "Actions" ใน repo
2. ถ้ามีปุ่ม "Enable workflows" → กดเลย
3. คลิก "🤖 AI Trading Bot" → "Run workflow" → เพื่อทดสอบครั้งแรก
4. ดู log ว่า bot ทำงานถูกต้อง
```

---

## ⏱️ ตาราง Schedule

| ทุก | กิจกรรม |
|-----|---------|
| **30 นาที** | อัปเดต trade เปิดอยู่ + % เหลือถึง TP |
| **60 นาที** | Scan หา signal ใหม่ทุก 17 symbols |
| **4 ชั่วโมง** | ส่ง Performance Report |
| **ทันที** | แจ้ง WIN / LOSS ทุกครั้ง |

---

## 📊 Symbols ที่ติดตาม

| กลุ่ม | Symbols |
|-------|---------|
| **Forex** | EURUSD, USDJPY, GBPUSD, GBPJPY, AUDUSD, USDCHF, USDCAD |
| **Crypto** | BTCUSD, ETHUSD, SOLUSD |
| **Commodities** | GOLD, SILVER, OIL |
| **Indices** | US100, US30, JP225, GER40 |

---

## 🆓 GitHub Actions Free Tier

```
✅ Free: 2,000 นาที/เดือน
✅ Bot รัน ~3 นาที/ครั้ง × 48 ครั้ง/วัน = 144 นาที/วัน
✅ = ~4,320 นาที/เดือน (เกิน free tier เล็กน้อย)

💡 แก้ได้: เปลี่ยน cron เป็นทุกชั่วโมง
   - cron: "0 * * * *"   # ทุกชั่วโมง = 72 นาที/วัน ✅
```

---

## 🔧 ปรับค่าใน bot.py

```python
MIN_CONF    = 0.62   # ความมั่นใจขั้นต่ำ (สูง = น้อย signal แต่แม่นขึ้น)
MIN_ADX     = 20     # กรอง sideways (สูง = เทรดแต่ trend แรง)
MIN_RR      = 1.5    # R:R ขั้นต่ำ
TOP_N       = 3      # ส่ง top N symbols ต่อรอบ
ATR_SL_MULT = 1.8    # SL ห่างจาก entry (ATR × ค่านี้)
ATR_TP_MULT = 3.2    # TP ห่างจาก entry (ATR × ค่านี้)
```

---

## 📱 ตัวอย่างข้อความที่จะได้รับ

```
🟢 NEW SIGNAL: GOLD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 Direction : BUY  🔥 HIGH
💰 Entry     : 4810.50000
🛑 SL        : 4795.20000
🎯 TP        : 4845.80000
📐 R:R       : 1:2.31
💼 Kelly     : 12.4% of capital
⏱ Est. hold : ~6h
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 Confidence: 74.2%
   ML:71%  MTF:76%  (align 3/3)
📊 Score: BUY=14.5  SELL=4.0  ADX=28
🌐 MTF: FAST:🟢  MEDIUM:🟢  SLOW:🟢
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Reasons:
  • EMA full bull stack
  • ADX=28 very strong up
  • MACD cross up
  • Above Ichimoku cloud
  • RSI7 cross up
```

---

## ⚠️ คำเตือน

> **นี่คือ Educational Tool เท่านั้น**
> ไม่ใช่คำแนะนำการลงทุน การเทรด Forex/Crypto มีความเสี่ยงสูง
> อาจสูญเสียเงินทั้งหมดได้ ควรศึกษาก่อนตัดสินใจ
