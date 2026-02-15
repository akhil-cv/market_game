import asyncio
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DTYPE = np.dtype([
    ("ts", "float64"),
    ("security_id", "U20"),
    ("side", "U3"),
    ("price", "float64"),
    ("qty", "int32")
])


class RingBuffer:
    def __init__(self, size, dtype):
        self.buffer = np.empty(size, dtype=dtype)
        self.size = size
        self.write_idx = 0
        self.full = False

    def append(self, row):
        self.buffer[self.write_idx] = row
        self.write_idx = (self.write_idx + 1) % self.size
        if self.write_idx == 0:
            self.full = True

    def snapshot(self):
        if not self.full:
            return self.buffer[:self.write_idx]
        return np.concatenate((
            self.buffer[self.write_idx:],
            self.buffer[:self.write_idx]
        ))


class MarketRecorder:
    def __init__(self, fd, buffer_size=2_000_000, flush_interval=300):
        self.fd = fd
        self.ring = RingBuffer(buffer_size, DTYPE)
        self.flush_interval = flush_interval

    async def market_data_loop(self):
        await self.fd.connect()

        while True:
            raw = await self.fd.ws.recv()
            remaining = raw

            while remaining:
                update = self.fd.process_data(remaining)
                if not update:
                    break

                remaining = update.pop("remaining_data", None)

                if update["type"] not in ("Bid", "Ask"):
                    continue

                self.ring.append((
                    time.time(),
                    update["security_id"],
                    update["type"],
                    update["price"],
                    update["qty"]
                ))

    async def parquet_flush(self):
        while True:
            await asyncio.sleep(self.flush_interval)

            snapshot = self.ring.snapshot()
            if len(snapshot) == 0:
                continue

            df = pd.DataFrame(snapshot)
            table = pa.Table.from_pandas(df, preserve_index=False)

            fname = time.strftime("nifty_fut_%Y%m%d_%H%M.parquet")
            pq.write_table(table, fname, compression="zstd")

            print(f"Wrote {len(snapshot)} ticks → {fname}")

    async def run(self):
        await asyncio.gather(
            self.market_data_loop(),
            self.parquet_flush()
        )
