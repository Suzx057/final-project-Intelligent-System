import streamlit as st

# ✅ ตั้งค่าหน้าหัวข้อ
st.title("📊 About Machine Learning")

# ✅ หัวข้อ 1: ที่มาของ Dataset และ Feature ที่ใช้
st.subheader("📂 ที่มาของ Dataset และ Feature ที่ใช้")
st.markdown("""
🔹 **Dataset: Ames Housing Dataset (ใช้ใน House Price Prediction)**  
- **ที่มา:** [Kaggle - Ames Housing](https://www.kaggle.com/datasets)  
- **คำอธิบาย:**  
  เป็นชุดข้อมูลเกี่ยวกับอสังหาริมทรัพย์ในเมือง Ames รัฐไอโอวา สหรัฐอเมริกา  
  มีมากกว่า **80 Features** และใช้สำหรับทำนายราคาบ้าน 🏡  

- **Feature ที่ใช้ในโมเดล:**  
  - `OverallQual` → คุณภาพโดยรวมของบ้าน (1-10)  
  - `GrLivArea` → พื้นที่ใช้สอยหลักของบ้าน (ตารางฟุต)  
  - `TotalBsmtSF` → พื้นที่ของชั้นใต้ดินทั้งหมด  
  - `GarageArea` → พื้นที่โรงจอดรถ (ตารางฟุต)  
  - `YearBuilt` → ปีที่สร้างบ้าน  
  - `FullBath` → จำนวนห้องน้ำเต็ม  

⚠️ **ความไม่สมบูรณ์ของข้อมูล**  
- **Missing Values** → มีคุณสมบัติบางตัวที่ไม่มีข้อมูล เช่น `TotalBsmtSF`  
- **Outliers** → มีข้อมูลที่ราคาสูงหรือต่ำผิดปกติ ซึ่งต้องตรวจสอบเพิ่มเติม  
""")

# ✅ แนวทางการพัฒนาโมเดล
st.markdown("""
### 🔍 แนวทางการพัฒนาโมเดล Machine Learning ตั้งแต่ต้นจนจบ
Machine Learning คือกระบวนการที่ให้คอมพิวเตอร์เรียนรู้จากข้อมูล **(Training Data)** เพื่อสร้างโมเดลที่สามารถทำนายผลลัพธ์ใหม่ๆ ได้  
""")

# ✅ หัวข้อ 2: การเตรียมข้อมูล (Data Preparation)
st.subheader("📊 การเตรียมข้อมูล (Data Preparation)")
st.markdown("""
✅ **1. การทำความสะอาดข้อมูล (Data Cleaning)**  
- จัดการค่าที่หายไป (Missing Values)  
- กำจัดค่าผิดปกติ (Outliers)  

✅ **2. การเลือก Feature ที่สำคัญ (Feature Selection)**  
- วิเคราะห์ว่าตัวแปรใดมีผลต่อโมเดล  
- ใช้วิธี **Correlation Analysis** หรือ **Feature Importance**  

✅ **3. Feature Engineering**  
- สร้างตัวแปรใหม่จากข้อมูลที่มี เช่น `TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF`  
- แปลงข้อมูลที่เป็นตัวอักษรให้เป็นตัวเลข (`One-Hot Encoding`)  

✅ **4. Normalization / Standardization**  
- ใช้ **MinMaxScaler** หรือ **StandardScaler** ปรับค่าสเกลของข้อมูล  
""")

# ✅ หัวข้อ 3: อัลกอริทึมที่นิยมใช้ใน Machine Learning
st.subheader("📖 อัลกอริทึมที่นิยมใช้ใน Machine Learning")
st.markdown("""
- **Linear Regression** → ใช้สำหรับปัญหาพยากรณ์ตัวเลข เช่น **ทำนายราคาบ้าน** 🏠  
- **Decision Tree / Random Forest** → ใช้สำหรับ **การตัดสินใจ** เช่น **ทำนายว่าผู้ใช้จะซื้อสินค้าหรือไม่**  
- **Gradient Boosting (XGBoost, LightGBM)** → ใช้ปรับปรุงประสิทธิภาพโมเดลที่ต้องการความแม่นยำสูง  
""")

# ✅ หัวข้อ 4: ขั้นตอนการพัฒนาโมเดล (Model Development)
st.subheader("🛠️ ขั้นตอนการพัฒนาโมเดล Machine Learning")
st.markdown("""
1️⃣ **เตรียมข้อมูล (Data Preprocessing)** → ล้างข้อมูล, จัดการ Missing Values, สร้าง Feature ใหม่  
2️⃣ **เลือกอัลกอริทึมที่เหมาะสม (Algorithm Selection)** → เช่น Linear Regression, Random Forest  
3️⃣ **แบ่งข้อมูลเป็น Train/Test (Data Splitting)** → ใช้ **80%** เทรนโมเดล และ **20%** ทดสอบ  
4️⃣ **Train โมเดล** → ใช้ `Scikit-learn` หรือ `XGBoost` ในการเรียนรู้จากข้อมูล  
5️⃣ **Hyperparameter Tuning** → ปรับแต่งค่าโมเดล เช่น Learning Rate, Max Depth  
6️⃣ **Evaluate โมเดล** → ใช้ค่าต่างๆ เช่น **RMSE, Accuracy**  
7️⃣ **Deploy โมเดล** → นำโมเดลไปใช้งานจริง เช่น API หรือ Web App  
""")

# ✅ แหล่งอ้างอิงที่สำคัญ
st.subheader("📚 แหล่งอ้างอิงที่เกี่ยวข้อง")
st.markdown("""
- 📖 [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)  
- 📖 [XGBoost: A Scalable Tree Boosting System](https://xgboost.ai/)  
- 📖 [Machine Learning Crash Course - Google](https://developers.google.com/machine-learning/crash-course)  
- 📖 [Ames Housing Dataset - Kaggle](https://www.kaggle.com/datasets)  
""")

# ✅ สรุป
st.success("🎯 การพัฒนา Machine Learning Model ครอบคลุมตั้งแต่ Data Preparation → Algorithm → Model Training → Deployment!")

# ✅ ส่วนของผู้จัดทำ
st.markdown("---")
st.subheader("📌 จัดทำโดย")
st.markdown("""
👨‍🎓 **นาย จิตรภาณุ คุ้มดี**  
🎓 รหัสนักศึกษา: **6404062663037**  
📚 สาขา: **วิทยาการคอมพิวเตอร์**  
🏫 มหาวิทยาลัย: **มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ**  
""")
