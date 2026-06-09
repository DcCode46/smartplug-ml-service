from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import IsolationForest


# ==================================================
# IMPORT LIBRARY
# ==================================================

import time
import traceback
from calendar import monthrange
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

import pandas as pd
import numpy as np


# ==================================================
# KONFIGURASI SERVICE
# ==================================================

INTERVAL_MINUTES = 5
# Production:
# INTERVAL_MINUTES = 5

MODEL_VERSION = "1.0"
HISTORY_PATH = "/smartplug/device1/history/hourly"
ML_PATH = "/smartplug/device1/ml"
ML_DEBUG_PATH = "/smartplug/device1/ml_debug"
# Sesuaikan jika tarif listrik yang dipakai aplikasi berubah.
ELECTRICITY_TARIFF_PER_KWH = 1444.70


# ==================================================
# FIREBASE INIT
# ==================================================

import os
import json

firebase_json = json.loads(
    os.environ["FIREBASE_CREDENTIALS"]
)

cred = credentials.Certificate(
    firebase_json
)


# ==================================================
# HELPER
# ==================================================

def check_firebase_ready():

    try:
        db.reference(ML_PATH).get()
        return True
    except Exception as e:
        print("Firebase Ready Check Failed")
        print(e)
        traceback.print_exc()
        return False


def build_anomaly_records(anomaly_df):

    anomaly_records = []

    for _, row in anomaly_df.iterrows():
        anomaly_records.append({
            "datetime": row["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
            "avg_watt": to_firebase_number(row["avg_watt"]),
            "avg_volt": to_firebase_number(row["avg_volt"]),
            "avg_ampere": to_firebase_number(row["avg_ampere"]),
            "kwh_used": to_firebase_number(row["kwh_used"]),
            "anomaly_score": to_firebase_number(row["anomaly_score"]),
        })

    return anomaly_records


def to_firebase_number(value):

    if pd.isna(value) or not np.isfinite(value):
        return None

    return float(value)


def to_firebase_cost(value):

    if pd.isna(value) or not np.isfinite(value):
        return None

    return int(round(float(value)))


def calculate_risk_status(anomaly_label, anomaly_score):

    score = to_firebase_number(anomaly_score)

    if anomaly_label == "NORMAL" or score is None or score >= 0:
        return {
            "risk_level": 0,
            "message": "Aman",
            "anomaly_score": score,
        }

    if score > -0.05:
        return {
            "risk_level": 1,
            "message": "Aktivitas Tidak Biasa",
            "anomaly_score": score,
        }

    return {
        "risk_level": 2,
        "message": "Perlu Perhatian",
        "anomaly_score": score,
    }


def calculate_analytics(df):

    reference_datetime = df["datetime"].max()
    reference_date = reference_datetime.date()

    today_df = df[
        df["datetime"].dt.date == reference_date
    ]

    month_df = df[
        (df["datetime"].dt.year == reference_datetime.year) &
        (df["datetime"].dt.month == reference_datetime.month)
    ]

    actual_today_kwh = to_firebase_number(
        today_df["kwh_used"].sum()
    )

    actual_month_kwh = to_firebase_number(
        month_df["kwh_used"].sum()
    )

    actual_today_cost = to_firebase_cost(
        today_df["kwh_used"].sum() * ELECTRICITY_TARIFF_PER_KWH
    )

    actual_month_cost = to_firebase_cost(
        month_df["kwh_used"].sum() * ELECTRICITY_TARIFF_PER_KWH
    )

    days_elapsed = reference_datetime.day
    days_in_month = monthrange(
        reference_datetime.year,
        reference_datetime.month
    )[1]

    projected_month_kwh = 0.0

    if days_elapsed > 0:
        avg_daily_kwh = month_df["kwh_used"].sum() / days_elapsed
        projected_month_kwh = avg_daily_kwh * days_in_month

    return {
        "reference_datetime": reference_datetime,
        "actual_today_kwh": actual_today_kwh,
        "actual_today_cost": actual_today_cost,
        "actual_month_kwh": actual_month_kwh,
        "actual_month_cost": actual_month_cost,
        "estimated_cost_today": actual_today_cost,
        "estimated_cost_month": to_firebase_cost(
            projected_month_kwh * ELECTRICITY_TARIFF_PER_KWH
        ),
    }


def build_main_ml_payload(
    risk_status,
    next_prediction_value,
    analytics_data
):

    return {
        "status": risk_status,
        "prediction": {
            "next_hour_kwh": (
                round(next_prediction_value, 4)
                if next_prediction_value is not None
                else None
            ),
            "estimated_cost_today": analytics_data["estimated_cost_today"],
            "estimated_cost_month": analytics_data["estimated_cost_month"],
        },
        "analytics": {
            "actual_today_kwh": (
                round(analytics_data["actual_today_kwh"], 4)
                if analytics_data["actual_today_kwh"] is not None
                else None
            ),
            "actual_today_cost": analytics_data["actual_today_cost"],
            "actual_month_kwh": (
                round(analytics_data["actual_month_kwh"], 4)
                if analytics_data["actual_month_kwh"] is not None
                else None
            ),
            "actual_month_cost": analytics_data["actual_month_cost"],
        },
        "service": {
            "status": "online",
            "model_version": MODEL_VERSION,
            "updated_at": int(time.time()),
        },
    }


def build_debug_payload(anomaly_records, mae, r2):

    return {
        "anomaly_records": anomaly_records,
        "model_metrics": {
            "mae": to_firebase_number(mae),
            "r2_score": to_firebase_number(r2),
        },
    }


# ==================================================
# AMBIL DATA + PROCESS ML
# ==================================================

def run_ml():

    print("\n===== ML RUN =====")
    print(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    ref = db.reference(
        HISTORY_PATH
    )

    data = ref.get()

    if not data:
        raise ValueError("Firebase history data is empty.")

    # ==================================================
    # FLATTEN
    # ==================================================

    rows = []

    for year in data:

        for month in data[year]:

            for day in data[year][month]:

                for hour in data[year][month][day]:

                    item = data[year][month][day][hour]

                    rows.append({

                        "year": int(year),
                        "month": int(month),
                        "day": int(day),
                        "hour": int(hour),

                        "avg_watt":
                        item.get("avg_watt", 0),

                        "avg_volt":
                        item.get("avg_volt", 0),

                        "avg_ampere":
                        item.get("avg_ampere", 0),

                        "kwh_used":
                        item.get("kwh_used", 0),

                        "timestamp":
                        item.get("timestamp", 0),
                    })

    if not rows:
        raise ValueError("No rows available from Firebase history data.")

    # ==================================================
    # DATAFRAME
    # ==================================================

    df = pd.DataFrame(rows)

    # ==================================================
    # SORT
    # ==================================================

    df = df.sort_values(
        by=["year", "month", "day", "hour"]
    )

    # ==================================================
    # CLEANING
    # ==================================================

    print("\n===== SEBELUM CLEANING =====\n")

    print(df)

    # hapus NaN
    df = df.dropna()

    print(df.describe())

    # ==================================================
    # REMOVE OUTLIER
    # ==================================================

    df = df[
        df["kwh_used"] < 0.2
    ]

    df = df[
        df["avg_watt"] < 100
    ]

    # ==================================================
    # FILTER DATA TIDAK MASUK AKAL
    # ==================================================

    df = df[
        (df["avg_volt"] > 150) &
        (df["avg_volt"] < 260)
    ]

    df = df[
        (df["avg_ampere"] >= 0) &
        (df["avg_ampere"] < 30)
    ]

    df = df[
        (df["avg_watt"] >= 0) &
        (df["avg_watt"] < 7000)
    ]

    df = df[
        (df["kwh_used"] >= 0) &
        (df["kwh_used"] < 20)
    ]

    # ==================================================
    # RESET INDEX
    # ==================================================

    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No data remains after cleaning.")

    # ==================================================
    # TAMBAH FITUR WAKTU
    # ==================================================

    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]]
    )

    df["day_of_week"] = df["datetime"].dt.dayofweek

    # ==================================================
    # HASIL
    # ==================================================

    print("\n===== SETELAH CLEANING =====\n")

    print(df)

    print("\n===== INFO =====\n")

    print(df.info())

    print("\n===== DESKRIPSI =====\n")

    print(df.describe())

    if len(df) < 2:
        raise ValueError("Not enough cleaned data for train_test_split.")

    # ==================================================
    # MACHINE LEARNING - PREDICTION
    # ==================================================

    print("\n===== MACHINE LEARNING =====\n")

    # ==================================================
    # FEATURE
    # ==================================================

    X = df[[
        "hour",
        "day_of_week",
        "avg_watt",
        "avg_volt",
        "avg_ampere"
    ]]

    # ==================================================
    # TARGET
    # ==================================================

    y = df["kwh_used"]

    try:
        # ==================================================
        # TRAIN TEST SPLIT
        # ==================================================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # ==================================================
        # MODEL
        # ==================================================

        model = LinearRegression()

        # ==================================================
        # TRAINING
        # ==================================================

        model.fit(X_train, y_train)

        # ==================================================
        # PREDICTION
        # ==================================================

        y_pred = model.predict(X_test)

        # ==================================================
        # EVALUASI
        # ==================================================

        mae = mean_absolute_error(y_test, y_pred)

        r2 = r2_score(y_test, y_pred)

        print("MAE :", mae)

        print("R2 Score :", r2)

        # ==================================================
        # HASIL PREDIKSI
        # ==================================================

        result_df = pd.DataFrame({

            "Actual": y_test.values,

            "Predicted": y_pred

        })

        print("\n===== HASIL PREDIKSI =====\n")

        print(result_df.head(20))

        # ==================================================
        # PREDIKSI DATA TERBARU
        # ==================================================

        latest_data = X.tail(1)

        next_prediction = model.predict(latest_data)

        print("\n===== PREDIKSI JAM BERIKUTNYA =====\n")

        print(
            "Prediksi kWh berikutnya:",
            round(next_prediction[0], 4),
            "kWh"
        )

        print("Prediction Success")
    except Exception:
        print("Prediction Failed")
        raise

    try:
        # ==================================================
        # ANOMALY DETECTION
        # ==================================================

        print("\n===== ANOMALY DETECTION =====\n")

        # ==================================================
        # FEATURE ANOMALY
        # ==================================================

        anomaly_features = df[[
            "avg_watt",
            "avg_volt",
            "avg_ampere",
            "kwh_used"
        ]]

        # ==================================================
        # MODEL ISOLATION FOREST
        # ==================================================

        anomaly_model = IsolationForest(

            contamination=0.05,

            random_state=42
        )

        # ==================================================
        # TRAIN MODEL
        # ==================================================

        anomaly_model.fit(anomaly_features)

        # ==================================================
        # PREDICT
        # ==================================================

        df["anomaly"] = anomaly_model.predict(
            anomaly_features
        )

        # ==================================================
        # CONVERT LABEL
        # ==================================================

        # -1 = anomaly
        #  1 = normal

        df["anomaly_label"] = df["anomaly"].apply(

            lambda x:
            "ANOMALY"
            if x == -1
            else "NORMAL"
        )

        # ==================================================
        # ANOMALY SCORE
        # ==================================================

        df["anomaly_score"] = anomaly_model.decision_function(
            anomaly_features
        )

        # ==================================================
        # HASIL
        # ==================================================

        print(df[[
            "datetime",
            "avg_watt",
            "kwh_used",
            "anomaly_label",
            "anomaly_score"
        ]])

        # ==================================================
        # FILTER ANOMALY
        # ==================================================

        anomaly_df = df[
            df["anomaly_label"] == "ANOMALY"
        ]

        print("\n===== DATA ANOMALY =====\n")

        print(anomaly_df[[
            "datetime",
            "avg_watt",
            "avg_volt",
            "avg_ampere",
            "kwh_used",
            "anomaly_score"
        ]])

        print("Anomaly Success")
    except Exception:
        print("Anomaly Failed")
        raise

    if not check_firebase_ready():
        print("Firebase Upload Failed")
        print("Retry Next Loop")
        return

    try:
        anomaly_records = build_anomaly_records(
            anomaly_df
        )

        latest_status_row = df.tail(1).iloc[0]
        next_prediction_value = to_firebase_number(
            next_prediction[0]
        )
        risk_status = calculate_risk_status(
            latest_status_row["anomaly_label"],
            latest_status_row["anomaly_score"]
        )
        analytics_data = calculate_analytics(df)

        ml_payload = build_main_ml_payload(
            risk_status,
            next_prediction_value,
            analytics_data
        )

        debug_payload = build_debug_payload(
            anomaly_records,
            mae,
            r2
        )

        db.reference(ML_PATH).set(
            ml_payload
        )

        db.reference(
            f"{ML_DEBUG_PATH}/anomaly_records"
        ).set(
            debug_payload["anomaly_records"]
        )

        db.reference(
            f"{ML_DEBUG_PATH}/model_metrics"
        ).set(
            debug_payload["model_metrics"]
        )

        print("Firebase Upload Success")
    except Exception:
        print("Firebase Upload Failed")
        raise


if __name__ == "__main__":

    print("SMART PLUG ML SERVICE STARTED")

    while True:

        try:
            run_ml()
        except Exception as e:
            print("ERROR")
            print(e)
            traceback.print_exc()
            print("Retry Next Loop")

        time.sleep(
            INTERVAL_MINUTES * 60
        )
