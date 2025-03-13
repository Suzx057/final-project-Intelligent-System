import streamlit as st
import pandas as pd
import numpy as np
import joblib

# โหลดโมเดลที่ฝึกไว้
linear_model = joblib.load("models/linear_regression_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")  # โหลดรายชื่อ Features ที่ใช้ตอน Train

# ตั้งค่าหน้า UI
st.title("🏡 House Price Prediction Demo")
st.markdown("### 📈 ทดลองทำนายราคาบ้านจากปัจจัยต่าง ๆ")

# ✅ UI Layout แบบ 2 Columns เพื่อให้ดูง่ายขึ้น
col1, col2 = st.columns(2)

# ✅ สร้าง Input Fields ตาม Feature ที่ใช้ Train
input_values = {}
for i, col in enumerate(feature_columns):
    if i % 2 == 0:
        with col1:
            input_values[col] = st.number_input(f"{col}", min_value=0, max_value=10000, value=100)
    else:
        with col2:
            input_values[col] = st.number_input(f"{col}", min_value=0, max_value=10000, value=100)

# ปุ่มทำนายราคา
st.markdown("---")  # เส้นคั่น
if st.button("🚀 **Predict House Price**"):
    # ✅ ตรวจสอบค่าที่ได้รับจาก UI
    input_data = pd.DataFrame([input_values])
    st.write("🔹 **ค่าที่ได้รับจาก UI:**")
    st.dataframe(input_data.style.format("{:.2f}"))

    # ✅ ทำ Standard Scaling
    try:
        input_scaled = scaler.transform(input_data)
        st.write("🔹 **ค่าหลังจาก Scaling:**")
        st.dataframe(pd.DataFrame(input_scaled, columns=feature_columns).style.format("{:.2f}"))

        # ✅ ทำนายราคา
        pred_linear = linear_model.predict(input_scaled)[0]
        pred_rf = rf_model.predict(input_scaled)[0]

        # ✅ แสดงผลลัพธ์
        st.markdown("### 🎯 **Prediction Results**")
        st.info(f"🏡 **Linear Regression Prediction:** ${pred_linear:,.2f}")
        st.success(f"🌲 **Random Forest Prediction:** ${pred_rf:,.2f}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
