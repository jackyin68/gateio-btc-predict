import json
import urllib.request
import csv

url = 'https://api.gateapi.io/api2/1/candlestick2/btc_usdt?group_sec=3600&range_hour=2400'
req = urllib.request.urlopen(url)
status = req.getcode()
if status == 200:
    rst = req.read().decode()
    data_dict = json.loads(rst)
    data_list = data_dict['data']
    with open('btc_usdt.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time', 'volume', 'close', 'high', 'low', 'open'])
        for data in data_list:
            writer.writerow(data)


def data_usage():
    print("time: 时间戳, volume: 交易量, close: 收盘价, high: 最高价, low: 最低价, open: 开盘价")
