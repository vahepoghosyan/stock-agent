```js 
cd /Users/vahe/stock-agent
```

Edit .env — add ANTHROPIC_API_KEY, NEWS_API_KEY, Gmail credentials

## 2. Test immediately
```js python3 main.py --run-now
python3 main.py --run-now
```

## 3. Run as a daily daemon (defaults to 09:00 local time)
```js
python3 main.py
```