import yfinance as yf

class GetData():
    @staticmethod
    def get_stocks_data(company='GOOG', rename=False):
        data = yf.download(tickers=company)

        if rename:
            data = data.rename(columns={col: col.lower() \
                               for col in data.columns})

        return data