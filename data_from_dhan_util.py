import asyncio
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import os

BASE_DIR = "market_data"
INSTRUMENT = "NIFTY_FUT"

today = datetime.now().strftime("%Y-%m-%d")
time_slot = datetime.now().strftime("%H_%M")


class DhanContext:
    def __init__(self, client_id, access_token):
        self._client_id = client_id
        self._access_token = access_token

    def get_client_id(self):
        return self._client_id

    def get_access_token(self):
        return self._access_token

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

                side = update["type"]

                for level in update["depth"]:
                    self.ring.append((
                        time.time(),
                        str(update["security_id"]),
                        side,
                        level["price"],
                        level["quantity"]
                    ))

    # async def parquet_flush(self):
    #     while True:
    #         await asyncio.sleep(self.flush_interval)

    #         snapshot = self.ring.snapshot()
    #         if len(snapshot) == 0:
    #             continue

    #         df = pd.DataFrame(snapshot)
    #         table = pa.Table.from_pandas(df, preserve_index=False)

    #         fname = time.strftime("nifty_fut_%Y%m%d_%H%M.parquet")
    #         pq.write_table(table, fname, compression="zstd")

    #         print(f"Wrote {len(snapshot)} rows → {fname}")

    # async def run(self):
    #     await asyncio.gather(
    #         self.market_data_loop(),
    #         self.parquet_flush()
    #     )
    
    async def parquet_flush(self):
        while True:
            await asyncio.sleep(self.flush_interval)

            snapshot = self.ring.snapshot()
            if len(snapshot) == 0:
                continue

            df = pd.DataFrame(snapshot)
            table = pa.Table.from_pandas(df, preserve_index=False)

            today = datetime.now().strftime("%Y-%m-%d")
            time_slot = datetime.now().strftime("%H_%M")

            dir_path = os.path.join(
                "market_data",
                "NIFTY_FUT",
                today
            )

            os.makedirs(dir_path, exist_ok=True)

            file_path = os.path.join(dir_path, f"{time_slot}.parquet")

            pq.write_table(table, file_path, compression="zstd")

            print(f"Wrote {len(snapshot)} rows → {file_path}")

    async def run(self):
        try:
            await asyncio.gather(
                self.market_data_loop(),
                self.parquet_flush()
            )
        except asyncio.CancelledError:
            print("Stopping recorder, flushing remaining data...")
            snapshot = self.ring.snapshot()
            if len(snapshot):
                df = pd.DataFrame(snapshot)
                table = pa.Table.from_pandas(df, preserve_index=False)
                fname = time.strftime("nifty_fut_%Y%m%d_%H%M_FINAL.parquet")
                pq.write_table(table, fname, compression="zstd")
            raise

