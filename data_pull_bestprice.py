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

# -------------------------------------------------
# FULL DEPTH SCHEMA (original)
# -------------------------------------------------

DTYPE = np.dtype([
    ("ts", "int64"),
    ("snap_idx", "int64"),
    ("security_id", "U20"),
    ("side", "U3"),
    ("price", "float64"),
    ("qty", "int32")
])

# -------------------------------------------------
# BEST BID / ASK SCHEMA
# -------------------------------------------------

DTYPE_L1 = np.dtype([
    ("ts", "int64"),
    ("snap_idx", "int64"),
    ("security_id", "U20"),
    ("bid_price", "float64"),
    ("bid_qty", "int32"),
    ("ask_price", "float64"),
    ("ask_qty", "int32")
])

# -------------------------------------------------
# Ring Buffer
# -------------------------------------------------

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

# -------------------------------------------------
# ORIGINAL FULL DEPTH RECORDER
# -------------------------------------------------

class MarketRecorder:
    def __init__(self, fd, buffer_size=2_000_000, flush_interval=300):

        self.fd = fd
        self.ring = RingBuffer(buffer_size, DTYPE)
        self.flush_interval = flush_interval

        self.snapshot_idx = 0
        self.last_flush_idx = 0

        self.parquet_writer = None
        self.current_file_path = None

    async def market_data_loop(self):

        await self.fd.connect()

        try:

            current_snap = None
            sides_seen = set()

            while True:

                raw = await self.fd.ws.recv()
                remaining = raw

                while remaining:

                    update = self.fd.process_data(remaining)

                    if not update:
                        break

                    remaining = update.pop("remaining_data", None)

                    side = update.get("type")

                    if side not in ("Bid", "Ask"):
                        continue

                    if current_snap is None:
                        self.snapshot_idx += 1
                        current_snap = self.snapshot_idx
                        sides_seen.clear()

                    sides_seen.add(side)
                    ts = time.time_ns()

                    for level in update["depth"]:

                        self.ring.append((
                            ts,
                            current_snap,
                            str(update["security_id"]),
                            side,
                            level["price"],
                            level["quantity"]
                        ))

                    if sides_seen == {"Bid", "Ask"}:
                        current_snap = None

        except asyncio.CancelledError:
            print("market_data_loop cancelled")
            raise

# -------------------------------------------------
# NEW BEST BID / ASK RECORDER
# -------------------------------------------------

class BestBidAskRecorder:

    def __init__(self, fd, buffer_size=2_00_000, flush_interval=300):

        self.fd = fd
        self.ring = RingBuffer(buffer_size, DTYPE_L1)

        self.flush_interval = flush_interval
        self.snapshot_idx = 0
        self.last_flush_idx = 0

        self.parquet_writer = None
        self.current_file_path = None

    async def market_data_loop(self):

        await self.fd.connect()

        try:

            current_bid = None
            current_ask = None

            while True:

                raw = await self.fd.ws.recv()
                remaining = raw

                while remaining:

                    update = self.fd.process_data(remaining)

                    if not update:
                        break

                    remaining = update.pop("remaining_data", None)

                    side = update.get("type")

                    if side not in ("Bid", "Ask"):
                        continue

                    best = update["depth"][0]

                    if side == "Bid":
                        current_bid = (best["price"], best["quantity"])

                    elif side == "Ask":
                        current_ask = (best["price"], best["quantity"])

                    if current_bid and current_ask:

                        ts = time.time_ns()
                        self.snapshot_idx += 1

                        self.ring.append((
                            ts,
                            self.snapshot_idx,
                            str(update["security_id"]),
                            current_bid[0],
                            current_bid[1],
                            current_ask[0],
                            current_ask[1]
                        ))

                        current_bid = None
                        current_ask = None

        except asyncio.CancelledError:
            print("market_data_loop cancelled")
            raise

    # -------------------------------------------------
    # PARQUET FLUSH
    # -------------------------------------------------

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

    # -------------------------------------------------
    # RUNNER
    # -------------------------------------------------

    async def run(self):

        tasks = [
            asyncio.create_task(self.market_data_loop()),
            asyncio.create_task(self.parquet_flush())
        ]

        try:

            await asyncio.gather(*tasks)

        except asyncio.CancelledError:

            print("Stopping recorder...")

            for t in tasks:
                t.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)

            if self.parquet_writer:
                self.parquet_writer.close()

            raise