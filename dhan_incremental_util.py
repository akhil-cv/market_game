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

DTYPE = np.dtype([
    ("ts", "int64"),          # nanoseconds since epoch
    ("security_id", "U20"),
    ("side", "U3"),
    ("price", "float64"),
    ("qty", "int32")
])


# -------------------- Ring Buffer --------------------

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

    def linear_view(self):
        if not self.full:
            return self.buffer[:self.write_idx]
        return np.concatenate((
            self.buffer[self.write_idx:],
            self.buffer[:self.write_idx]
        ))


# -------------------- Market Recorder --------------------

class MarketRecorder:
    def __init__(self, fd, buffer_size=2_000_000, flush_interval=300):
        self.fd = fd
        self.ring = RingBuffer(buffer_size, DTYPE)
        self.flush_interval = flush_interval

        self.last_flush_idx = 0
        self.parquet_writer = None
        self.current_file_path = None
        self._tasks = []

    # -------- Market Data Loop --------
    async def market_data_loop(self):
        await self.fd.connect()
        try:
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

                    for level in update["depth"]:
                        self.ring.append((
                            time.time_ns(),
                            str(update["security_id"]),
                            update["type"],
                            level["price"],
                            level["quantity"]
                        ))

        except asyncio.CancelledError:
            print("market_data_loop cancelled")
            raise

    # -------- Incremental Parquet Flush --------
    async def parquet_flush(self):
        try:
            while True:
                await asyncio.sleep(self.flush_interval)

                data = self.ring.linear_view()
                total_rows = len(data)

                if total_rows <= self.last_flush_idx:
                    continue

                new_rows = data[self.last_flush_idx:total_rows]
                self.last_flush_idx = total_rows

                df = pd.DataFrame(new_rows)

                today = datetime.now().strftime("%Y-%m-%d")
                time_slot = datetime.now().strftime("%H_%M")

                dir_path = os.path.join(BASE_DIR, INSTRUMENT, today)
                os.makedirs(dir_path, exist_ok=True)

                file_path = os.path.join(dir_path, f"{time_slot}.parquet")

                table = pa.Table.from_pandas(df, preserve_index=False)

                if self.parquet_writer is None or self.current_file_path != file_path:
                    if self.parquet_writer:
                        self.parquet_writer.close()

                    self.parquet_writer = pq.ParquetWriter(
                        file_path,
                        table.schema,
                        compression="zstd"
                    )
                    self.current_file_path = file_path

                self.parquet_writer.write_table(table)
                print(f"Appended {len(df)} rows → {file_path}")

        except asyncio.CancelledError:
            print("parquet_flush cancelled")
            raise

    # -------- Runner --------
    async def run(self):
        self._tasks = [
            asyncio.create_task(self.market_data_loop()),
            asyncio.create_task(self.parquet_flush())
        ]

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            print("Stopping recorder...")

            for t in self._tasks:
                t.cancel()

            await asyncio.gather(*self._tasks, return_exceptions=True)

            # Final flush
            data = self.ring.linear_view()
            if self.last_flush_idx < len(data):
                df = pd.DataFrame(data[self.last_flush_idx:])
                table = pa.Table.from_pandas(df, preserve_index=False)

                fname = f"{INSTRUMENT}_{datetime.now():%Y%m%d_%H%M}_FINAL.parquet"
                pq.write_table(table, fname, compression="zstd")
                print(f"Final write: {len(df)} rows → {fname}")

            if self.parquet_writer:
                self.parquet_writer.close()

            raise
