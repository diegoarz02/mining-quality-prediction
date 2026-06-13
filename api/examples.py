"""Ejemplos reales para la documentación interactiva de la API.

Generados desde las últimas horas del dataset procesado (9 de septiembre de 2017), de
modo que el botón Try it out de Swagger funcione al primer clic con valores plausibles.
Archivo generado por scripts internos; no editar a mano.
"""

PREDICT_EXAMPLE = {
    "features": {
        "% Silica Concentrate__lag1h": 1.96,
        "Ore Pulp pH__mean": 9.6538,
        "Amina Flow__mean": 460.7699,
        "Starch Flow__mean": 3638.5437
    }
}

SIMULATE_EXAMPLE = {
    "base_features": {
        "% Silica Concentrate__lag1h": 1.96
    },
    "deltas": {
        "Amina Flow__mean": 50.0,
        "Flotation Column 06 Level__mean": -30.0
    }
}

HISTORY_EXAMPLE = {
    "history": [
        {
            "date": "2017-09-09T17:00:00",
            "values": {
                "% Iron Feed": 49.75,
                "% Silica Feed": 23.2,
                "Starch Flow": 2557.5758,
                "Amina Flow": 499.1311,
                "Ore Pulp Flow": 381.0547,
                "Ore Pulp pH": 9.2873,
                "Ore Pulp Density": 1.6524,
                "Flotation Column 01 Air Flow": 299.9668,
                "Flotation Column 02 Air Flow": 298.8384,
                "Flotation Column 03 Air Flow": 299.9502,
                "Flotation Column 04 Air Flow": 300.0311,
                "Flotation Column 05 Air Flow": 299.988,
                "Flotation Column 06 Air Flow": 300.06,
                "Flotation Column 07 Air Flow": 299.9393,
                "Flotation Column 01 Level": 400.1304,
                "Flotation Column 02 Level": 499.9708,
                "Flotation Column 03 Level": 400.0188,
                "Flotation Column 04 Level": 400.2762,
                "Flotation Column 05 Level": 498.1823,
                "Flotation Column 06 Level": 398.9535,
                "Flotation Column 07 Level": 401.583
            },
            "silica": 2.74
        },
        {
            "date": "2017-09-09T18:00:00",
            "values": {
                "% Iron Feed": 49.75,
                "% Silica Feed": 23.2,
                "Starch Flow": 2582.1233,
                "Amina Flow": 489.3462,
                "Ore Pulp Flow": 380.5578,
                "Ore Pulp pH": 9.4339,
                "Ore Pulp Density": 1.6661,
                "Flotation Column 01 Air Flow": 297.8585,
                "Flotation Column 02 Air Flow": 299.4154,
                "Flotation Column 03 Air Flow": 299.8248,
                "Flotation Column 04 Air Flow": 299.9321,
                "Flotation Column 05 Air Flow": 299.9605,
                "Flotation Column 06 Air Flow": 300.0213,
                "Flotation Column 07 Air Flow": 299.9415,
                "Flotation Column 01 Level": 420.1487,
                "Flotation Column 02 Level": 499.9377,
                "Flotation Column 03 Level": 415.447,
                "Flotation Column 04 Level": 405.1795,
                "Flotation Column 05 Level": 499.7667,
                "Flotation Column 06 Level": 406.6765,
                "Flotation Column 07 Level": 402.9491
            },
            "silica": 1.83
        },
        {
            "date": "2017-09-09T19:00:00",
            "values": {
                "% Iron Feed": 49.75,
                "% Silica Feed": 23.2,
                "Starch Flow": 3327.0478,
                "Amina Flow": 497.2112,
                "Ore Pulp Flow": 380.8472,
                "Ore Pulp pH": 9.1762,
                "Ore Pulp Density": 1.6609,
                "Flotation Column 01 Air Flow": 301.5656,
                "Flotation Column 02 Air Flow": 300.1691,
                "Flotation Column 03 Air Flow": 299.9007,
                "Flotation Column 04 Air Flow": 299.9511,
                "Flotation Column 05 Air Flow": 299.9242,
                "Flotation Column 06 Air Flow": 335.6581,
                "Flotation Column 07 Air Flow": 298.7367,
                "Flotation Column 01 Level": 397.7816,
                "Flotation Column 02 Level": 498.7533,
                "Flotation Column 03 Level": 403.859,
                "Flotation Column 04 Level": 398.9305,
                "Flotation Column 05 Level": 502.9717,
                "Flotation Column 06 Level": 399.4689,
                "Flotation Column 07 Level": 400.5595
            },
            "silica": 1.65
        },
        {
            "date": "2017-09-09T20:00:00",
            "values": {
                "% Iron Feed": 49.75,
                "% Silica Feed": 23.2,
                "Starch Flow": 4225.8003,
                "Amina Flow": 508.9639,
                "Ore Pulp Flow": 381.1129,
                "Ore Pulp pH": 9.3875,
                "Ore Pulp Density": 1.692,
                "Flotation Column 01 Air Flow": 300.0502,
                "Flotation Column 02 Air Flow": 299.9678,
                "Flotation Column 03 Air Flow": 299.9065,
                "Flotation Column 04 Air Flow": 299.9635,
                "Flotation Column 05 Air Flow": 299.9599,
                "Flotation Column 06 Air Flow": 348.1728,
                "Flotation Column 07 Air Flow": 303.5993,
                "Flotation Column 01 Level": 399.9286,
                "Flotation Column 02 Level": 499.649,
                "Flotation Column 03 Level": 399.2086,
                "Flotation Column 04 Level": 399.9671,
                "Flotation Column 05 Level": 501.6245,
                "Flotation Column 06 Level": 398.905,
                "Flotation Column 07 Level": 400.4862
            },
            "silica": 1.71
        },
        {
            "date": "2017-09-09T21:00:00",
            "values": {
                "% Iron Feed": 49.75,
                "% Silica Feed": 23.2,
                "Starch Flow": 2808.2147,
                "Amina Flow": 517.7488,
                "Ore Pulp Flow": 381.0644,
                "Ore Pulp pH": 9.7713,
                "Ore Pulp Density": 1.7356,
                "Flotation Column 01 Air Flow": 299.8143,
                "Flotation Column 02 Air Flow": 299.8016,
                "Flotation Column 03 Air Flow": 299.9732,
                "Flotation Column 04 Air Flow": 299.8617,
                "Flotation Column 05 Air Flow": 299.9266,
                "Flotation Column 06 Air Flow": 349.4228,
                "Flotation Column 07 Air Flow": 309.8754,
                "Flotation Column 01 Level": 399.9614,
                "Flotation Column 02 Level": 500.4849,
                "Flotation Column 03 Level": 471.8275,
                "Flotation Column 04 Level": 399.931,
                "Flotation Column 05 Level": 500.2233,
                "Flotation Column 06 Level": 401.8998,
                "Flotation Column 07 Level": 400.5563
            },
            "silica": 1.8
        },
        {
            "date": "2017-09-09T22:00:00",
            "values": {
                "% Iron Feed": 49.75,
                "% Silica Feed": 23.2,
                "Starch Flow": 3191.4977,
                "Amina Flow": 492.5112,
                "Ore Pulp Flow": 380.445,
                "Ore Pulp pH": 9.7821,
                "Ore Pulp Density": 1.7166,
                "Flotation Column 01 Air Flow": 300.1228,
                "Flotation Column 02 Air Flow": 299.6795,
                "Flotation Column 03 Air Flow": 299.927,
                "Flotation Column 04 Air Flow": 299.9442,
                "Flotation Column 05 Air Flow": 299.937,
                "Flotation Column 06 Air Flow": 349.6136,
                "Flotation Column 07 Air Flow": 305.2546,
                "Flotation Column 01 Level": 399.6014,
                "Flotation Column 02 Level": 498.6145,
                "Flotation Column 03 Level": 836.2772,
                "Flotation Column 04 Level": 400.4554,
                "Flotation Column 05 Level": 500.4062,
                "Flotation Column 06 Level": 402.7033,
                "Flotation Column 07 Level": 401.7691
            },
            "silica": 1.96
        },
        {
            "date": "2017-09-09T23:00:00",
            "values": {
                "% Iron Feed": 49.75,
                "% Silica Feed": 23.2,
                "Starch Flow": 3638.5437,
                "Amina Flow": 460.7699,
                "Ore Pulp Flow": 380.8231,
                "Ore Pulp pH": 9.6538,
                "Ore Pulp Density": 1.6678,
                "Flotation Column 01 Air Flow": 299.7789,
                "Flotation Column 02 Air Flow": 300.0954,
                "Flotation Column 03 Air Flow": 299.8408,
                "Flotation Column 04 Air Flow": 299.9333,
                "Flotation Column 05 Air Flow": 299.942,
                "Flotation Column 06 Air Flow": 349.3551,
                "Flotation Column 07 Air Flow": 315.2647,
                "Flotation Column 01 Level": 400.0263,
                "Flotation Column 02 Level": 499.7218,
                "Flotation Column 03 Level": 868.5492,
                "Flotation Column 04 Level": 398.7647,
                "Flotation Column 05 Level": 500.6822,
                "Flotation Column 06 Level": 399.3756,
                "Flotation Column 07 Level": 398.7753
            }
        }
    ]
}
