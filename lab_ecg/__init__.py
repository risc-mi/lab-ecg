import catabra_pandas
import numpy as np
import pandas as pd
from catabra_lib import metrics
from tqdm import tqdm

MIMIC_LAB_MAPPING = {  # map parameter names to MIMIC itemid (both MIMIC-III and MIMIC-IV)
    "ALT": [50861],
    "AST": [50878],
    "albumin": [50862],
    "BUN": [51006],
    "free_calcium": [50808],
    "LDH": [50954],
    "pO2": [50821],
    "base_excess": [50802],
    "pCO2": [50818],
    "pH": [50820],
    "troponin_t": [51003],
    "creatine_kinase": [50910],
    "lactate": [50813],
    "bilirubin": [50885],
    "total_calcium": [50893],
    "magnesium": [50960],
    "PTT": [51275],
    "INR": [51237],
    "glucose": [50931, 50809],
    "leukocytes": [51301],
    "platelets": [51265],
    "bicarbonate": [50803, 50882],
    "hemoglobin": [51222, 50811],
    "hematocrit": [51221, 50810],
    "creatinine": [50912],
    "chloride": [50902, 50806],
    "sodium": [50983, 50824],
    "potassium": [50971, 50822],
    "phosphate": [50970],
}


RANGES = {
    "free_calcium": {"lower": 0.5, "upper": 2.0},
    "LDH": {"lower": 75, "upper": 9000},
    "pO2": {"lower": 20, "upper": 500},
    "base_excess": {"lower": -20, "upper": 20},
    "pCO2": {"lower": 15, "upper": 120},
    "pH": {"lower": 6.3, "upper": 8.4},
    "troponin_t": {"lower": 0, "upper": 20.85},
    "creatine_kinase": {"lower": 1, "upper": 110000},
    "lactate": {"lower": 0.4, "upper": 30},
    "bilirubin": {"lower": 0, "upper": 45},
    "AST": {"lower": 0, "upper": 5000},
    "ALT": {"lower": 0, "upper": 5000},
    "total_calcium": {"lower": 5, "upper": 13},
    "hemoglobin": {"lower": 1.3, "upper": 25},
    "magnesium": {"lower": 0.1, "upper": 10},
    "PTT": {"lower": 18.8, "upper": 150},
    "INR": {"lower": 0.8, "upper": 12.0},
    "glucose": {"lower": 33, "upper": 2000},
    "leukocytes": {"lower": 0, "upper": 1000},
    "platelets": {"lower": 0, "upper": 2000},
    "bicarbonate": {"lower": 0, "upper": 60},
    "hematocrit": {"lower": 0, "upper": 75},
    "BUN": {"lower": 0, "upper": 250},
    "creatinine": {"lower": 0.1, "upper": 60},
    "chloride": {"lower": 50, "upper": 175},
    "sodium": {"lower": 50, "upper": 225},
    "potassium": {"lower": 0.1, "upper": 12},
    "phosphate": {"lower": 0.5, "upper": 20},
    "albumin": {"lower": 0.6, "upper": 6},
}


MAX_DIFF = {  # maximum allowed difference of two measurements assigned to same waveform recording
    "base_excess": 1,
    "free_calcium": 0,
    "bicarbonate": 0,
    "lactate": 0,
    "pCO2": 1,
    "pO2": 1,
    "PTT": 0,
    "phosphate": 0,
    "albumin": 0,
    "ALT": 0,
    "AST": 0,
    "BUN": 0,
    "total_calcium": 0,
    "chloride": 0,
    "bilirubin": 0,
    "creatine_kinase": 0,
    "glucose": 5,
    "hematocrit": 1,
    "hemoglobin": 0.3,
    "troponin_t": 0,
    "INR": 0,
    "potassium": 0.3,
    "creatinine": 0,
    "LDH": 0,
    "leukocytes": 0,
    "magnesium": 0,
    "sodium": 2,
    "pH": 0.01,
    "platelets": 0,
}


TARGETS = [
    ("hemoglobin", "<10", "Hb < 10 g/dL"),
    ("hemoglobin", "<12", "Hb < 12 g/dL"),
    ("hemoglobin", "<8", "Hb < 8 g/dL"),
    ("hematocrit", ">50", "Hematocrit > 50 %"),
    ("hematocrit", "<38", "Hematocrit < 38 %"),
    ("potassium", "<3.3", "K+ < 3.3 mEq/L"),
    ("potassium", "<3.5", "K+ < 3.5 mEq/L"),
    ("potassium", ">5.1", "K+ > 5.1 mEq/L"),
    ("albumin", "<3.4", "Albumin < 3.4 g/dL"),
    ("creatinine", ">1.2", "Creatinine > 1.2 mg/dL"),
    ("BUN", "<6", "BUN < 6 mg/dL"),
    ("BUN", ">20", "BUN > 20 mg/dL"),
    ("INR", ">1.5", "INR > 1.5"),
    ("total_calcium", "<8", "Total Ca < 8 mg/dL"),
    ("total_calcium", "<8.4", "Total Ca < 8.4 mg/dL"),
    ("bilirubin", ">1.5", "Bilirubin > 1.5 mg/dL"),
    ("phosphate", ">4.5", "Phosphate > 4.5 mg/dL"),
    ("glucose", ">150", "Glucose > 150 mg/dL"),
    ("leukocytes", ">11", "Leukocytes > 11 K/µL"),
    ("pO2", "<85", "pO2 (arterial) < 85 mmHg"),
    ("ALT", ">40", "ALT > 40 IU/L"),
    ("AST", ">40", "AST > 40 IU/L"),
    ("total_calcium", ">10.3", "Total Ca > 10.3 mg/dL"),
    ("total_calcium", ">11", "Total Ca > 11 mg/dL"),
    ("creatine_kinase", ">322", "CK > 322 IU/L"),
    ("creatine_kinase", "<29", "CK < 29 IU/L"),
    ("glucose", "<70", "Glucose < 70 mg/dL"),
    ("troponin_t", ">0.1", "Troponin T > 0.1 ng/mL"),
    ("potassium", ">5.5", "K+ > 5.5 mEq/L"),
    ("leukocytes", "<4", "Leukocytes < 4 K/µL"),
    ("magnesium", ">2.6", "Mg > 2.6 mg/dL"),
    ("magnesium", "<1.6", "Mg < 1.6 mg/dL"),
    ("sodium", ">145", "Na > 145 mEq/L"),
    ("sodium", "<133", "Na < 133 mEq/L"),
    ("sodium", "<130", "Na < 130 mEq/L"),
    ("pH", ">7.45", "pH (arterial) > 7.45"),
    ("pH", "<7.35", "pH (arterial) < 7.35"),
    ("base_excess", ">3", "Base Excess > 3 mEq/L"),
    ("base_excess", "<-3", "Base Excess < -3 mEq/L"),
    ("free_calcium", ">1.32", "Free Ca > 1.32 mmol/L"),
    ("free_calcium", "<1.12", "Free Ca < 1.12 mmol/L"),
    ("bicarbonate", ">32", "Bicarbonate > 32 mEq/L"),
    ("bicarbonate", "<22", "Bicarbonate < 22 mEq/L"),
    ("lactate", ">2", "Lactate > 2 mmol/L"),
    ("pCO2", ">45", "pCO2 (arterial) > 45 mmHg"),
    ("pCO2", "<35", "pCO2 (arterial) < 35 mmHg"),
    ("phosphate", "<2.7", "Phosphate < 2.7 mg/dL"),
    ("platelets", ">440", "Platelets > 440 K/µL"),
    ("platelets", "<150", "Platelets < 150 K/µL"),
    ("PTT", ">36.5", "PTT > 36.5 s"),
    ("PTT", "<25", "PTT < 25 s"),
    ("chloride", ">108", "Cl- > 108 mEq/L"),
    ("chloride", "<96", "Cl- < 96 mEq/L"),
    ("LDH", ">250", "LDH > 250 IU/L"),
    ("pO2", ">150", "pO2 (arterial) > 150 mmHg"),
]


def validate_lab(
    lab: pd.DataFrame,
    param_col: str = "itemid",
    value_col: str = "valuenum",
    keep_unknown_params: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Validate laboratory measurements wrt. `RANGES`.

    Parameters
    ----------
    lab
        Table with laboratory measurements. DataFrame with one row per measurement and columns `param_col` and
        `value_col`.
    param_col
        Name of the column in `lab` containing lab param names / IDs. If the data type of this column is int,
        `MIMIC_LAB_MAPPING` is used to obtain corresponding IDs.
    value_col
        Name of the column in `lab` containing measured values.
    keep_unknown_params
        Keep rows with lab params not appearing in `RANGES`. If False, such rows are deemed invalid and discarded.
    verbose
        Print intermediate output, like the fraction of valid measurements per lab param.

    Returns
    -------
    Subset of `lab`, restricted to rows with valid measurements.
    """
    if lab[param_col].dtype.kind in "ui":
        mapping = MIMIC_LAB_MAPPING
    else:
        mapping = {}
    global_mask = pd.Series(index=lab.index, data=keep_unknown_params)
    for k, rng in RANGES.items():
        param_mask = lab[param_col].isin(mapping.get(k, [k]))
        if param_mask.any():
            mask = param_mask & (lab[value_col] >= rng.get("lower", -1e7)) & (lab[value_col] <= rng.get("upper", 1e7))
            if verbose:
                print(f"{k}: {mask.sum() / param_mask.sum():.3f}")
            if keep_unknown_params:
                global_mask &= ~param_mask | mask
            else:
                global_mask |= param_mask & mask
    return lab if global_mask.all() else lab[global_mask].copy()


def assign_labs_to_records(
    lab: pd.DataFrame,
    records: pd.DataFrame,
    specimen_type: pd.DataFrame | None = None,
    temporal_dist_minutes: int = 30,
    param_col: str = "itemid",
    value_col: str = "valuenum",
    lab_time_col: str = "charttime",
    records_time_col: str = "timestamp",
    lab_id_col: str = "subject_id",
    records_id_col: str = "subject_id",
    verbose: bool = True,
):
    """Assign lab measurements to records, based on subject- or similar IDs and timestamps.

    Parameters
    ----------
    lab
        Table with lab measurements. DataFrame with one row per measurement and columns `param_col`, `value_col`,
        `lab_time_col` and `lab_id_col`.
    records
        Table with records. DataFrame with one row per record and columns `records_time_col` and `records_id_col`.
        Must have a unique row index.
    specimen_type
        Table with specimen types, optional. If given, must be a DataFrame with column "value" containing specimen
        types (like "ART.", "VEN.", etc.) and a row index that links to column "specimen_id" in `lab`.
        Specimen types are currently only used to restrict lab params pH, pCO2 and pO2 to arterial blood.
    temporal_dist_minutes
        The maximum temporal distance, in minutes, allowed between a lab measurement and a record in `records`.
    param_col
        Name of the column in `lab` containing lab param names / IDs. If the data type of this column is int,
        `MIMIC_LAB_MAPPING` is used to obtain corresponding IDs.
    value_col
        Name of the column in `lab` containing measured values.
    lab_time_col
        Name of the column in `lab` containing measurement times. If the data type is int or float, times are assumed to
        be given in seconds.
    records_time_col
        Name of the column in `records` containing timestamps. If the data type is int or float, times are assumed to be
        given in seconds. In any case, must be the same data type as in `lab_time_col`.
    lab_id_col
        Name of the column in `lab` containing IDs, like subject- or stay-IDs. Lab measurements are only assigned to
        records with the same ID.
    records_id_col
        Name of the column in `records` containing IDs, like subject- or stay-IDs. Lab measurements are only assigned to
        records with the same ID.
    verbose
        Print intermediate output, like a progress bar.

    Returns
    -------
    DataFrame with a subset of the rows of `records`
    """

    if specimen_type is not None and "specimen_id" not in lab.columns:
        raise ValueError('specimen_type given, but found no "specimen_id" column in lab.')

    if lab[param_col].dtype.kind in "ui":
        mapping = MIMIC_LAB_MAPPING
    else:
        mapping = {k: [k] for k in MIMIC_LAB_MAPPING}

    if temporal_dist_minutes <= 0:
        raise ValueError("The temporal distance must be positive, but got", temporal_dist_minutes)
    if lab[lab_time_col].dtype.kind in "mM":
        temporal_dist_minutes = pd.Timedelta(temporal_dist_minutes, unit="m")
    else:
        temporal_dist_minutes *= 60

    assert records.index.is_unique

    dfs = []
    for k, ids in tqdm(mapping.items()) if verbose else mapping.items():
        df = lab.loc[
            lab[param_col].isin(ids),
            [lab_id_col, lab_time_col, value_col] + ([] if specimen_type is None else ["specimen_id"]),
        ]
        if df.empty:
            continue

        df.set_index(lab_id_col, inplace=True)
        if specimen_type is not None and k in ("pCO2", "pO2", "pH"):
            df = df.join(specimen_type["value"].to_frame("specimen_type"), on="specimen_id")
            df = df[df["specimen_type"] == "ART."].drop("specimen_type", axis=1)

        df = records[[records_id_col, records_time_col]].join(df, on=records_id_col, how="left")
        df[k + "_dist"] = (df[lab_time_col] - df[records_time_col]).abs()
        df = df[df[k + "_dist"] <= temporal_dist_minutes]
        df_dist = df.groupby(level=0)[k + "_dist"].min()
        df_median = df[df[k + "_dist"] == df_dist.reindex(df.index).values].groupby(level=0)[value_col].median()
        df_diff = df.groupby(level=0)[value_col].agg(["min", "max"]).diff(axis=1)["max"]
        df = pd.concat(
            [df_median.to_frame(k), df_dist.to_frame(k + "_dist"), df_diff.to_frame(k + "_diff")],
            axis=1,
            sort=False,
        )

        df.loc[df[k + "_diff"] > MAX_DIFF.get(k, 1e7), k] = np.nan
        df.drop(k + "_diff", axis=1, inplace=True)
        df.dropna(subset=[k], axis=0, inplace=True)

        dfs.append(df)

    return pd.concat(dfs, axis=1, sort=False, join="outer")


def resample_eav(
    samples: pd.DataFrame,
    entity_col: str,
    value_col: str,
    time_col: str,
    window_start,
    window_stop,
    agg=("r-1", "mean"),
) -> pd.DataFrame:
    """
    Light-weight wrapper around function `catabra.util.longitudinal.resample_eav()`, to easily extract/resample values
    from a given table.

    Parameters
    ----------
    samples
        DataFrame with samples. Must have one row per sample. This serves both as the source where values are
        extracted/resampled from, as well as the definition of the target time windows. See arguments `window_start`
        and `window_stop` for details.
    entity_col
        Name of the column in `samples` containing information about individual entities, e.g., subjects.
    value_col
        Name of the column in `samples` containing the values to extract/resample.
    time_col
        Name of the column in `samples` containing time stamps.
    window_start
        Start of each time window, relative to `time_col`. Negative if the windows start in the past, positive if they
        start in the future.
    window_stop
        End of each time window, relative to `time_col`. Negative if the windows end in the past, positive if they end
        in the future.
    agg
        List of aggregations to compute for each window.

    Returns
    -------
    DataFrame with extracted/resampled values. Has the same number of rows and the same row index as `samples`, and one
    column for each of the requested aggregations. Values can be NaN.
    """

    df = catabra_pandas.resample_eav(
        samples,
        catabra_pandas.make_windows(anchor=time_col, start_rel=window_start, stop_rel=window_stop),
        agg={value_col: list(agg)},
        entity_col=entity_col,
        time_col=time_col,
        value_col=value_col,
        include_stop=True,
    )
    assert (df.index == samples.index).all()

    out = pd.DataFrame(index=df.index)
    for a in agg:
        out[value_col + "_" + a] = df[(value_col, a)]

    return out


def evaluate(
    y_true: np.ndarray, y_hat: np.ndarray, y_pred: np.ndarray, n_repetitions: int = 1000
) -> metrics.Bootstrapping:
    def compute_metrics(_y_true, _y_hat, _y_pred) -> dict:
        if len(np.unique(_y_true)) == 2:
            cm = metrics.confusion_matrix(_y_true, _y_pred)
            return {
                "auroc": metrics.roc_auc(_y_true, _y_hat),
                "ap": metrics.average_precision(_y_true, _y_hat),
                "sens": metrics.sensitivity_cm(cm=cm),
                "spec": metrics.specificity_cm(cm=cm),
                "ppv": metrics.positive_predictive_value_cm(cm=cm),
                "npv": metrics.negative_predictive_value_cm(cm=cm),
                "f1": metrics.f1_cm(cm=cm),
            }
        else:
            return {
                "auroc": np.nan,
                "ap": np.nan,
                "sens": np.nan,
                "spec": np.nan,
                "ppv": np.nan,
                "npv": np.nan,
                "f1": np.nan,
            }

    bs = metrics.Bootstrapping(y_true, y_hat, y_pred, fn=compute_metrics)
    if n_repetitions <= 1:
        return bs.run(sample_indices=np.arange(len(y_true))[np.newaxis])
    else:
        return bs.run(n_repetitions=n_repetitions)
