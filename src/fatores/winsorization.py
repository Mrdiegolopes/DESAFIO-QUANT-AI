import numpy as np
import pandas as pd


def winsorize_series(s: pd.Series, limits: tuple[float, float] = (0.01, 0.09)) -> pd.Series:
    """
    Aplica winsorização em uma série numéricas cortando nos percentis definidos.
    Exemplo: limits=(0.01, 0.99) limita os dados entre o 1º e o 99º percentil.
    """
    s_clean = s.copy()
    valid_data = s_clean.dropna()

    if len(valid_data) == 0:
        return s_clean

    lower_bound = np.percentile(valid_data, limits[0] * 100)
    upper_bound = np.percentile(valid_data, limits[1] * 100)

    return s_clean.clip(lower=lower_bound, upper=upper_bound)


def winsorize_factors_by_quarter(
    df: pd.DataFrame, factor_cols: list[str], date_col: str = "DT_REFER", limits: tuple[float, float] = (0.01, 0.99)
) -> pd.DataFrame:
    """
    Aplica winsorização grupo a grupo (cross-sectional) para cada trimestre.
    """
    df_out = df.copy()

    for date, group in df_out.groupby(date_col):
        for col in factor_cols:
            if col in group.columns:
                df_out.loc[group.index, col] = winsorize_series(group[col], limits=limits)

    return df_out