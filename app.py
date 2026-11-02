from pathlib import Path
import warnings

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Nhóm 16 – Dự đoán hỏng hóc máy",
    page_icon="⚙️",
    layout="centered"
)

@st.cache_resource
def load_model():
    path = Path(__file__).parent / "mo_hinh_demo.joblib"
    return joblib.load(path)

demo = load_model()

st.title("⚙️ Dự đoán hỏng hóc máy")
st.caption("Đồ án học máy – Nhóm 16")

st.write(
    "Nhập các thông số vận hành để phân loại trạng thái máy "
    "bằng mô hình đã huấn luyện."
)

# Hai mẫu minh họa đã lấy từ validation
mau = st.selectbox(
    "Chọn mẫu minh họa hoặc chỉnh thông số bên dưới",
    ["Mẫu không hỏng", "Mẫu có hỏng"]
)

if mau == "Mẫu không hỏng":
    defaults = ("M", 302.5, 312.0, 1487, 46.1, 175)
else:
    defaults = ("L", 304.0, 312.9, 1363, 62.5, 200)

# Tạo lại biểu mẫu khi đổi mẫu minh họa
with st.form(key=f"du_doan_{mau}"):
    col1, col2 = st.columns(2)

    with col1:
        loai = st.selectbox(
            "Loại sản phẩm",
            ["L", "M", "H"],
            index=["L", "M", "H"].index(defaults[0])
        )
        air = st.number_input(
            "Nhiệt độ không khí (K)",
            min_value=0.1,
            value=defaults[1],
            step=0.1
        )
        process = st.number_input(
            "Nhiệt độ quá trình (K)",
            min_value=0.1,
            value=defaults[2],
            step=0.1
        )

    with col2:
        speed = st.number_input(
            "Tốc độ quay (rpm)",
            min_value=1,
            value=defaults[3],
            step=1
        )
        torque = st.number_input(
            "Mô-men xoắn (Nm)",
            min_value=0.1,
            value=defaults[4],
            step=0.1
        )
        wear = st.number_input(
            "Thời gian sử dụng dụng cụ (min)",
            min_value=0,
            value=defaults[5],
            step=1
        )

    submit = st.form_submit_button("Dự đoán")

if submit:
    row = pd.DataFrame([{
        "Type": loai,
        "Air temperature [K]": air,
        "Process temperature [K]": process,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear
    }])[demo["feature_columns"]]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "X does not have valid feature names, "
                "but LGBMClassifier was fitted with feature names"
            ),
            category=UserWarning
        )
        probability = float(
            demo["pipeline"].predict_proba(row)[0, 1]
        )

    threshold = float(demo["threshold"])

    if probability >= threshold:
        st.error("DỰ ĐOÁN: CÓ HỎNG HÓC")
    else:
        st.success("DỰ ĐOÁN: KHÔNG HỎNG HÓC")

    st.metric(
        "Xác suất hỏng hóc ước lượng",
        f"{probability:.2%}"
    )
    st.caption(
        f"Mô hình: {demo['model_name']} | "
        f"Ngưỡng phân loại: {threshold:.2%}"
    )

st.divider()
st.caption(
    "Demo phân loại trên dữ liệu mô phỏng AI4I 2020. "
    "Kết quả không xác định thời điểm máy sẽ hỏng."
)
st.caption(
    "Hai mẫu có sẵn dùng để minh họa thao tác, "
    "không thay thế kết quả đánh giá trên tập kiểm thử."
)
