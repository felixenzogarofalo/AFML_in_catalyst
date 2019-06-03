import pandas as pd
import numpy as np

class VolumeBarSeries(object):
    def __init__(self, df, vol_frequency):
        self.df = df
        self.vol_frequency = vol_frequency

    def process_data(self):
        result = []
        buffer = []
        volume_buffer = 0.0

        for i in range(len(self.df.volume)):
            p_i = self.df.close.iloc[i]
            v_i = self.df.volume.iloc[i]
            d_i = self.df.index.values[i]

            buffer.append(p_i)
            volume_buffer += v_i

            if volume_buffer >= self.vol_frequency:
                o = buffer[0]
                h = np.max(buffer)
                l = np.min(buffer)
                c = buffer[-1]

                result.append({
                    "datetime": d_i,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": volume_buffer
                })

                buffer, volume_buffer = [], 0.0

        result = pd.DataFrame(result)
        result.datetime = pd.to_datetime(result["datetime"])
        result = result.set_index("datetime")
        return result


def getTEvents(gRaw, h):
    tEvents, sPos, sNeg = [],0,0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos = max(0, sPos + diff.loc[i])
        sNeg = min(0, sNeg + diff.loc[i])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


def getTEventsVol(gRaw, vol):
    tEvents, sPos, sNeg = [],0,0
    returns = gRaw.pct_change().fillna(method="bfill")
    for i in vol.index[1:]:
        sPos = max(0, sPos + returns.loc[i])
        sNeg = min(0, sNeg + returns.loc[i])
        if sNeg < -vol.loc[i]:
            sNeg = 0
            tEvents.append(i)
        elif sPos > vol.loc[i]:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


def getTEventsCross(fast_ema, slow_ema):
    tEvents = []
    diff = pd.DataFrame(data=fast_ema.values - slow_ema.values, index=slow_ema.index, columns=["diff"])
    for i in range(1, diff.shape[0]):
        if diff.iloc[i, 0] > 0.0 and diff.iloc[i - 1, 0] <=0.0:
            tEvents.append(diff.index[i])
        if diff.iloc[i, 0] < 0.0 and diff.iloc[i - 1, 0] >= 0.0:
            tEvents.append(diff.index[i])
    return pd.DatetimeIndex(tEvents)


