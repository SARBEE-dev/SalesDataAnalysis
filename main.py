import yfinance as yf
import pandas as pd
import streamlit as st
# how to run stream lit app on chrome  streamlit run main.py

st.write("""Colons can be used to align columns.

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the 
raw Markdown line up prettily. You can also use inline Markdown.

Markdown | Less | Pretty
--- | --- | ---
*Still* | `renders` | **nicely**
1 | 2 | 3""")
st.write(""" # Data analysis using streamlit beginner level""")
#ticker symbol: AAPL for Apple GOOGL for google
tickerSymbol = 'AAPL'
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')

st.write("""
### Closing Price
""")
st.line_chart(tickerDf.Close)

st.write("""
## Volume Price
""")
st.line_chart(tickerDf.Volume)

