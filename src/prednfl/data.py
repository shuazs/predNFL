from __future__ import annotations

import pandas as pd
import nflreadpy as nfl


class DataLoader:
    """Load core datasets needed for backtesting."""

    def __init__(self, seasons: tuple[int, ...]) -> None:
        self.seasons = seasons

    def load_schedules(self) -> pd.DataFrame:
        schedules = nfl.load_schedules(seasons=list(self.seasons))
        return self._to_pandas(schedules)

    def load_team_stats(self) -> pd.DataFrame:
        team_stats = nfl.load_team_stats(seasons=list(self.seasons))
        return self._to_pandas(team_stats)

    @staticmethod
    def _to_pandas(frame: pd.DataFrame) -> pd.DataFrame:
        if hasattr(frame, "to_pandas"):
            return frame.to_pandas()
        if hasattr(frame, "to_pandas_df"):
            return frame.to_pandas_df()
        return frame
