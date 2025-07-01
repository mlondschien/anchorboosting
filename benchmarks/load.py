# From https://github.com/xhochy/nyc-taxi-fare-prediction-deployment-example
from pathlib import Path

import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

DATA_PATH = Path(__file__).resolve().parent / "data"
NYC_TAXI_DATASET_PATH = DATA_PATH / "nyc_taxi_data.parquet"
NYC_TAXI_DATASET_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-01.parquet"
)
NYC_TAXI_BURROWS_PATH = DATA_PATH / "nyc_taxi_burrows.parquet"
NYC_TAXI_BURROWS_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"


def load_nyc_taxi_raw():
    if NYC_TAXI_DATASET_PATH.exists():
        return pl.read_parquet(NYC_TAXI_DATASET_PATH)
    else:
        df = pl.read_parquet(NYC_TAXI_DATASET_URL)
        df.write_parquet(NYC_TAXI_DATASET_PATH)
        return df


def load_nyc_taxi_burrows():
    if NYC_TAXI_BURROWS_PATH.exists():
        return pl.read_parquet(NYC_TAXI_BURROWS_PATH)
    else:
        df = pl.read_csv(NYC_TAXI_BURROWS_URL)
        df.write_parquet(NYC_TAXI_BURROWS_PATH)
        return df


def load_nyc_taxi(n):
    df = load_nyc_taxi_raw().head(n)

    burrows = load_nyc_taxi_burrows()

    df = (
        df.filter(pl.col("total_amount").is_not_null() & pl.col("total_amount").gt(0))
        .join(
            burrows.with_columns(pl.col("Borough").alias("pickup_borough")),
            left_on="PULocationID",
            right_on="LocationID",
            how="left",
        )
        .join(
            burrows.with_columns(pl.col("Borough").alias("dropoff_borough")),
            left_on="DOLocationID",
            right_on="LocationID",
            how="left",
        )
        .with_columns(
            pl.col("total_amount").log().alias("total_amount_log"),
            pl.col("tpep_pickup_datetime").dt.weekday().alias("pickup_weekday"),
            pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
            pl.col("tpep_pickup_datetime").dt.minute().alias("pickup_minute"),
            pl.col("pickup_borough").fill_null("Unknown").alias("pickup_borough"),
        )
    )

    continuous_columns = [
        "passenger_count",
        "trip_distance",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "congestion_surcharge",
        "pickup_hour",
        "pickup_minute",
    ]
    df = df.with_columns(pl.col(c).cast(pl.Float64) for c in continuous_columns)

    ordinal_columns = [
        "VendorID",
        "RatecodeID",
        "store_and_fwd_flag",
        "payment_type",
        "service_zone",
        "Borough",
        "Zone",
        "pickup_weekday",
        "pickup_borough",
        "dropoff_borough",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", "passthrough", continuous_columns),
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_columns,
            ),
        ]
    ).set_output(transform="polars")

    y = df.select(pl.col("total_amount_log")).to_numpy().ravel()
    y_binary = (y > y.mean()).astype(int)

    X = preprocessor.fit_transform(df)
    Z = X.select("categorical__Zone").to_numpy().astype("int32").ravel()

    return X.drop("categorical__Zone"), Z, y, y_binary
