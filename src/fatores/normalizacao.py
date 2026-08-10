import pandas as pd


def zscore_cross_sectional(
    df: pd.DataFrame, factor_cols: list[str], date_col: str = "DT_REFER"
) -> pd.DataFrame:
    """
    Calcula o Z-score Cross-Sectional (média=0, std=1) para cada fator
    dentro de cada janela trimestral.
    """
    df_out = df.copy()

    for col in factor_cols:
        zscore_col = f"z_{col}"
        df_out[zscore_col] = df_out.groupby(date_col)[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

    return df_out