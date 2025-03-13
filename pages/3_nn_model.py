import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ✅ โหลดโมเดล Neural Network และ Scaler
nn_model = joblib.load("models/neural_network_model.pkl")
scaler = joblib.load("models/scaler_nn.pkl")
feature_columns = joblib.load("models/feature_columns_nn.pkl")  # โหลดรายชื่อ Features ที่ใช้ตอน Train

# ✅ ตั้งค่าหน้า UI
st.title("💳 Credit Card Fraud Detection")
st.markdown("### 🏦 ตรวจสอบว่าธุรกรรมเป็น **Fraud หรือ Safe**")

# ✅ แสดงคำอธิบาย
st.info("""
🔹 **เกี่ยวกับโมเดลนี้:**  
- ใช้ **Neural Network** เพื่อตรวจจับ **ธุรกรรมบัตรเครดิตที่อาจเป็น Fraud**  
- ใช้ **Feature Scaling** และ **Feature Engineering** เพื่อเพิ่มประสิทธิภาพของโมเดล  
- ฝึกโมเดลโดยใช้ **Credit Card Fraud Dataset** จาก Kaggle  
""")

st.markdown("---")  # เส้นคั่น

# ✅ อธิบาย Feature ที่ผู้ใช้ต้องกรอก
st.subheader("📌 คำอธิบายของแต่ละ Feature")
feature_descriptions = {
    "V1 - V28": "ค่าที่ถูกลดมิติโดย **PCA (Principal Component Analysis)** เพื่อลดความซับซ้อนของข้อมูล",
    "Amount": "จำนวนเงินของธุรกรรมที่เกิดขึ้น (หน่วยเป็น USD)",
    "Time": "เวลาที่ธุรกรรมเกิดขึ้น (วัดจากเวลาธุรกรรมแรกใน Dataset, หน่วยเป็นวินาที)"
}
for key, value in feature_descriptions.items():
    st.markdown(f"**{key}** → {value}")

st.markdown("---")  # เส้นคั่น

# ✅ UI Layout แบบ 3 Columns เพื่อให้กรอกค่าได้ง่าย
st.subheader("📝 ป้อนค่าของธุรกรรมที่ต้องการตรวจสอบ")
col1, col2, col3 = st.columns(3)

# ✅ สร้าง Input Fields ตาม Feature ที่ใช้ Train
input_values = {}
for i, col in enumerate(feature_columns):
    if i % 3 == 0:
        with col1:
            input_values[col] = st.number_input(f"{col}", min_value=-50.0, max_value=50.0, value=0.0)
    elif i % 3 == 1:
        with col2:
            input_values[col] = st.number_input(f"{col}", min_value=-50.0, max_value=50.0, value=0.0)
    else:
        with col3:
            input_values[col] = st.number_input(f"{col}", min_value=-50.0, max_value=50.0, value=0.0)

# ✅ ปุ่มตรวจสอบ Fraud
st.markdown("---")  # เส้นคั่น
if st.button("🚀 **ตรวจสอบธุรกรรม**"):
    st.markdown("### 🔄 ขั้นตอนการพยากรณ์ธุรกรรม")
    
    # ✅ ตรวจสอบค่าที่ได้รับจาก UI
    input_data = pd.DataFrame([input_values])
    st.write("🔹 **ค่าที่ได้รับจาก UI:**")
    st.dataframe(input_data.style.format("{:.2f}"))

    # ✅ ทำ Standard Scaling
    try:
        input_scaled = scaler.transform(input_data)
        st.write("🔹 **ค่าหลังจาก Scaling:**")
        st.dataframe(pd.DataFrame(input_scaled, columns=feature_columns).style.format("{:.2f}"))

        # ✅ ทำนายผล
        prediction = nn_model.predict(input_scaled)[0]

        # ✅ แสดงผลลัพธ์
        st.markdown("### 🎯 **Fraud Detection Result**")
        if prediction == 1:
            st.error("🚨 **Alert: This transaction is FRAUD!** ❌")
        else:
            st.success("✅ **This transaction is SAFE.** 🟢")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

st.markdown("---")  # เส้นคั่น

# ✅ อธิบาย Workflow ของโมเดล
st.subheader("🛠️ Workflow ของการทำงานของโมเดล")
st.markdown("""
1️⃣ **รับข้อมูลธุรกรรมจากผู้ใช้** → เช่น จำนวนเงิน, เวลา, และค่าที่ถูกลดมิติโดย PCA  
2️⃣ **ปรับขนาดข้อมูลด้วย Standard Scaling** → เพื่อให้ค่าของแต่ละ Feature อยู่ในช่วงที่เหมาะสม  
3️⃣ **ป้อนข้อมูลเข้าสู่ Neural Network Model** → โมเดลจะพิจารณาค่าที่ได้รับ และทำการทำนาย  
4️⃣ **โมเดลจะคืนค่าระหว่าง 0 - 1** → ค่าที่ใกล้ 1 หมายถึง Fraud, ค่าที่ใกล้ 0 หมายถึง Safe  
5️⃣ **แสดงผลลัพธ์ให้ผู้ใช้** → ถ้าธุรกรรมเป็น Fraud จะมีแจ้งเตือน 🚨  
""")

# ✅ แหล่งอ้างอิง (References)
st.subheader("📚 แหล่งอ้างอิง")
st.markdown("""
- Kaggle Credit Card Fraud Dataset: [🔗 Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- SMOTE for Imbalanced Data: [🔗 Link](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)  
- StandardScaler ใน Machine Learning: [🔗 Link](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)  
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