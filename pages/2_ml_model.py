import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ✅ โหลดโมเดลและ Scaler
linear_model = joblib.load("models/linear_regression_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")  # โหลด Feature List ที่ถูกต้อง

# ✅ ตั้งค่าหน้า UI
st.title("🏡 House Price Prediction")
st.markdown("### 🔍 Demo การทำงานของ Machine Learning Model")

st.info("""
🔹 **เกี่ยวกับโมเดลนี้:**  
- ใช้ **Linear Regression** และ **Random Forest** ในการทำนายราคาบ้าน  
- ใช้ **Feature Scaling** เพื่อปรับค่าคุณสมบัติให้อยู่ในช่วงที่เหมาะสม  
- ฝึกโมเดลโดยใช้ **Ames Housing Dataset** ซึ่งเป็นชุดข้อมูลอสังหาริมทรัพย์  
""")

st.markdown("---")  # เส้นคั่น

# ✅ อธิบาย Feature ที่ผู้ใช้ต้องกรอก
st.subheader("📌 คำอธิบายของแต่ละ Feature")
feature_descriptions = {
    "OverallQual": "คุณภาพโดยรวมของบ้าน (1 = แย่, 10 = ดีมาก)",
    "GrLivArea": "พื้นที่ใช้สอยบนพื้นดิน (หน่วย: ตารางฟุต)",
    "TotalBsmtSF": "พื้นที่รวมของห้องใต้ดิน (หน่วย: ตารางฟุต)",
    "GarageArea": "พื้นที่โรงจอดรถ (หน่วย: ตารางฟุต)",
    "YearBuilt": "ปีที่บ้านถูกสร้าง",
    "FullBath": "จำนวนห้องน้ำเต็มรูปแบบ (ไม่รวมครึ่งห้องน้ำ)"
}
for key, value in feature_descriptions.items():
    st.markdown(f"**{key}** → {value}")

st.markdown("---")  # เส้นคั่น

# ✅ UI สำหรับป้อนข้อมูล
st.subheader("📝 ป้อนข้อมูลบ้านที่ต้องการทำนาย")
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.number_input("🏠 Overall Quality (1-10)", min_value=1, max_value=10, value=5)
    gr_liv_area = st.number_input("📏 Above Ground Living Area (sq ft)", min_value=300, max_value=5000, value=1500)
    total_bsmt_sf = st.number_input("🏗️ Total Basement Area (sq ft)", min_value=0, max_value=3000, value=800)

with col2:
    garage_area = st.number_input("🚗 Garage Area (sq ft)", min_value=0, max_value=1500, value=500)
    year_built = st.number_input("📅 Year Built", min_value=1800, max_value=2023, value=2000)
    full_bath = st.number_input("🛁 Number of Full Bathrooms", min_value=0, max_value=5, value=2)

# ✅ ปุ่มทำนายราคา
st.markdown("---")  # เส้นคั่น
if st.button("🚀 **Predict Price**"):
    st.markdown("### 🔄 ขั้นตอนการพยากรณ์ราคาบ้าน")
    
    # ✅ สร้าง DataFrame ให้ตรงกับตอน Train
    input_data = pd.DataFrame([[overall_qual, gr_liv_area, total_bsmt_sf, garage_area, year_built, full_bath]],
                              columns=["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageArea", "YearBuilt", "FullBath"])

    # ✅ เติมค่าให้ครบ 245 คอลัมน์
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # ใส่ค่า 0 ให้คอลัมน์ที่ขาดไป

    # ✅ เรียงลำดับคอลัมน์ให้ตรงกับที่โมเดล Train
    input_data = input_data[feature_columns]

    # 🔍 **ตรวจสอบค่าก่อน Scaling**
    st.markdown("#### 📊 ข้อมูลก่อน Scaling")
    st.dataframe(input_data)

    # ✅ ทำ Standard Scaling
    input_scaled = scaler.transform(input_data)

    # 🔍 **ตรวจสอบค่าหลัง Scaling**
    st.markdown("#### 🔢 ข้อมูลหลัง Scaling")
    st.write(pd.DataFrame(input_scaled, columns=feature_columns).head())

    # ✅ ทำนายราคาบ้าน
    pred_linear = linear_model.predict(input_scaled)[0]
    pred_rf = rf_model.predict(input_scaled)[0]

    # ✅ แสดงผลลัพธ์
    st.markdown("### 🎯 **ผลลัพธ์การทำนายราคาบ้าน**")
    st.success(f"📊 **Linear Regression Prediction:** ${pred_linear:,.2f}")
    st.success(f"🌲 **Random Forest Prediction:** ${pred_rf:,.2f}")

st.markdown("---")  # เส้นคั่น

# ✅ อธิบาย Workflow ของโมเดล
st.subheader("🛠️ Workflow ของการทำงานของโมเดล")
st.markdown("""
1️⃣ **รับข้อมูลจากผู้ใช้** → เช่น คุณภาพบ้าน, ขนาด, ปีที่สร้าง  
2️⃣ **ปรับขนาดข้อมูลด้วย Standard Scaling** → ให้ค่าของแต่ละ Feature อยู่ในช่วงที่เหมาะสม  
3️⃣ **ป้อนข้อมูลเข้าสู่ Machine Learning Model** → ทั้ง **Linear Regression** และ **Random Forest**  
4️⃣ **โมเดลจะคืนค่าราคาบ้านที่คาดการณ์** → แสดงผลให้ผู้ใช้  
""")

# ✅ แหล่งอ้างอิง (References)
st.subheader("📚 แหล่งอ้างอิง")
st.markdown("""
- Kaggle Ames Housing Dataset: [🔗 Link](https://www.kaggle.com/datasets/quantbruce/ames-housing-data)  
- StandardScaler ใน Machine Learning: [🔗 Link](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)  
- Random Forest Regression: [🔗 Link](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)  
""")

# ✅ ส่วนของผู้จัดทำ
st.markdown("---")
st.subheader("📌 จัดทำโดย")
st.markdown("""
👨‍🎓 **นาย จิตรภาณุ คุ้มดี**  
🎓 รหัสนักศึกษา: **6404062663037**  
📚 สาขา: **วิทยาการคอมพิวเตอร์**  
🏫 มหาวิทยาลัย: **มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ**  
""")