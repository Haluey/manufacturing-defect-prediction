from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="공정 불량 예측",
    page_icon="🏭",
    layout="centered"
)

project_dir = Path(__file__).resolve().parent
model_dir = project_dir / "models"
data_dir = project_dir / "data" / "raw"


@st.cache_resource
def load_models():
    return {
        "첫 번째 모델": joblib.load(
            model_dir / "defect_prediction_baseline.joblib"
        ),
        "두 번째 모델": joblib.load(
            model_dir / "defect_prediction.joblib"
        )
    }


@st.cache_data
def load_data():
    measurements = pd.read_csv(
        data_dir / "secom.data",
        sep=r"\s+",
        header=None
    )

    labels = pd.read_csv(
        data_dir / "secom_labels.data",
        sep=r"\s+",
        header=None,
        names=["result", "timestamp"]
    )

    answers = labels["result"].map({-1: 0, 1: 1})
    return measurements, answers


def predict_record(bundle, record):
    # 해당 모델에 저장된 항목과 순서로 전처리
    selected = record.loc[:, bundle["selected_columns"]]
    processed = bundle["imputer"].transform(selected)

    if bundle["scaler"] is not None:
        processed = bundle["scaler"].transform(processed)

    model = bundle["model"]
    defect_column = list(model.classes_).index(1)
    probability = float(
        model.predict_proba(processed)[0, defect_column]
    )

    prediction = int(probability >= bundle["threshold"])
    return prediction, probability


st.title("공정 불량 예측")
st.caption("SECOM 테스트 기록으로 두 모델의 예측 결과를 비교합니다.")

required_files = [
    model_dir / "defect_prediction_baseline.joblib",
    model_dir / "defect_prediction.joblib",
    data_dir / "secom.data",
    data_dir / "secom_labels.data"
]

if any(not path.exists() for path in required_files):
    st.error("모델 또는 원본 데이터 파일이 없습니다. 폴더를 확인해주세요.")
    st.stop()

bundles = load_models()
X, y = load_data()

test_indices = bundles["두 번째 모델"]["test_indices"]

if bundles["첫 번째 모델"]["test_indices"] != test_indices:
    st.error("두 모델의 테스트 기록 구성이 다릅니다.")
    st.stop()

record_index = st.selectbox(
    "테스트 기록 선택",
    options=sorted(test_indices),
    format_func=lambda index: f"기록 {index}"
)

record = X.loc[[record_index]]
actual_label = "불량" if y.loc[record_index] == 1 else "정상"

st.markdown(f"**실제 검사 결과: {actual_label}**")
st.divider()

panels = st.columns(2)

for panel, (name, bundle) in zip(panels, bundles.items()):
    prediction, probability = predict_record(bundle, record)
    predicted_label = "불량" if prediction == 1 else "정상"

    with panel:
        st.subheader(name)
        st.metric("불량 예측 확률", f"{probability:.2%}")
        st.caption(f"불량 판단 기준: {bundle['threshold']:.0%}")

        if prediction == 1:
            st.warning(f"예측 결과: {predicted_label}")
        else:
            st.info(f"예측 결과: {predicted_label}")

        if predicted_label == actual_label:
            st.write("실제 검사 결과와 일치")
        else:
            st.write("실제 검사 결과와 불일치")

with st.expander("선택한 기록의 원본 측정값"):
    measurement_table = record.T
    measurement_table.columns = ["측정값"]
    measurement_table.index.name = "측정 항목 번호"
    st.dataframe(measurement_table)

st.caption(
    "학습·평가용 예시 화면입니다. "
    "불량 예측 확률은 실제 불량 발생률과 일치하도록 보정된 값이 아닙니다."
)