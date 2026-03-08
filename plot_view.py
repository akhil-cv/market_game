import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

class PlotSimple:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._clean()

    def _clean(self):
        self.df["qty"] = pd.to_numeric(self.df["qty"], errors="coerce")
        self.df["price"] = pd.to_numeric(self.df["price"], errors="coerce")
        self.df["ts"] = pd.to_numeric(self.df["ts"], errors="coerce")
        self.df.dropna(subset=["qty", "price", "ts"], inplace=True)

        self.df["t"] = (self.df["ts"] - self.df["ts"].min()) / 1e6

    def plot_3D(self):
        bids = self.df[self.df["side"] == "Bid"]
        asks = self.df[self.df["side"] == "Ask"]

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection="3d")

        dx_b = np.full(len(bids), 0.6)
        dy_b = np.full(len(bids), 0.05)
        dx_a = np.full(len(asks), 0.6)
        dy_a = np.full(len(asks), 0.05)

        ax.bar3d(
            bids["t"].values,
            bids["price"].values,
            np.zeros(len(bids)),
            dx_b,
            dy_b,
            -bids["qty"].values,
            color="blue",
            alpha=0.7
        )

        ax.bar3d(
            asks["t"].values,
            asks["price"].values,
            np.zeros(len(asks)),
            dx_a,
            dy_a,
            asks["qty"].values,
            color="red",
            alpha=0.7
        )

        max_vol = self.df["qty"].max()
        ax.set_zlim(-max_vol, max_vol)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Price")
        ax.set_zlabel("Quantity")
        ax.set_title("3D Market Depth (Bid ↓ / Ask ↑)")
        ax.view_init(elev=28, azim=135)

        plt.tight_layout()
        plt.show()

    def plot_snapshot(self, ts_value):
        snap = self.df[self.df["ts"] == ts_value]

        if snap.empty:
            raise ValueError("No data found for this timestamp")

        bids = snap[snap["side"] == "Bid"].sort_values("price", ascending=False)
        asks = snap[snap["side"] == "Ask"].sort_values("price")

        plt.figure(figsize=(10, 6))

        plt.bar(
            bids["price"],
            -bids["qty"],
            width=0.05,
            color="blue",
            alpha=0.7,
            label="Bids"
        )

        plt.bar(
            asks["price"],
            asks["qty"],
            width=0.05,
            color="red",
            alpha=0.7,
            label="Asks"
        )

        plt.axhline(0, color="black")
        plt.xlabel("Price")
        plt.ylabel("Quantity")
        plt.title(f"Market Depth Snapshot @ ts={ts_value}")
        plt.legend()

        plt.tight_layout()
        plt.show()

class PlotDynamic:
    def plot_3Dyn(self, T=30, DEPTH=20):
        time = np.arange(T)

        mid_price = 100 + np.cumsum(np.random.normal(0, 0.05, T))

        bid_prices = mid_price[:, None] - np.arange(DEPTH) * 0.1
        ask_prices = mid_price[:, None] + np.arange(DEPTH) * 0.1

        bid_volumes = np.random.randint(50, 400, size=(T, DEPTH))
        ask_volumes = np.random.randint(50, 400, size=(T, DEPTH))

        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111, projection="3d")

        for t in range(T):
            dx = np.full(DEPTH, 0.6)
            dy = np.full(DEPTH, 0.08)

            ax.bar3d(
                np.full(DEPTH, time[t]),
                bid_prices[t],
                np.zeros(DEPTH),
                dx,
                dy,
                -bid_volumes[t],
                color="blue",
                alpha=0.75
            )

            ax.bar3d(
                np.full(DEPTH, time[t]),
                ask_prices[t],
                np.zeros(DEPTH),
                dx,
                dy,
                ask_volumes[t],
                color="red",
                alpha=0.75
            )

        max_vol = max(bid_volumes.max(), ask_volumes.max())
        ax.set_zlim(-max_vol, max_vol)

        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.set_zlabel("Volume")
        ax.set_title("Synthetic 3D Market Depth")

        ax.view_init(elev=28, azim=135)
        plt.tight_layout()
        plt.show()