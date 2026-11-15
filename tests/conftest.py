import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clima_sintetico():
    """20 años diarios con ET0 que crece año a año (para ordenar escenarios sin ambigüedad)."""
    idx = pd.date_range("2001-01-01", "2020-12-31")
    anio = idx.year - 2001
    estacion = np.sin(2 * np.pi * (idx.day_of_year - 30) / 365.25)
    clima = pd.DataFrame({
        "tmax": 32 + 2 * estacion,
        "tmin": 20 + 2 * estacion,
        "et0": 4.5 + 1.0 * estacion + 0.02 * anio,
        "precip": 0.0,
    }, index=idx)
    clima.loc[clima.index.month.isin([2, 3]) & (clima.index.day % 10 == 0), "precip"] = 8.0
    return clima
