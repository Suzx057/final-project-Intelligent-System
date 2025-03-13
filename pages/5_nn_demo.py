import streamlit as st

# ✅ ตั้งค่าหน้าหัวข้อ
st.title("💳 About Neural Network for Fraud Detection")

# ✅ หัวข้อ 1: ที่มาของ Dataset และ Feature ที่ใช้
st.subheader("📂 ที่มาของ Dataset และ Feature ที่ใช้")
st.markdown("""
🔹 **Dataset: Credit Card Fraud Detection Dataset**  
- **ที่มา:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **คำอธิบาย:**  
  - เป็นชุดข้อมูลธุรกรรมบัตรเครดิตจากสถาบันการเงินในยุโรป ซึ่งมีทั้งธุรกรรมปกติและธุรกรรม Fraud  
  - มีทั้งหมด **284,807 รายการธุรกรรม** โดยมีธุรกรรม Fraud เพียง **0.17%** ของข้อมูลทั้งหมด  
  - มี Feature **30 ตัว** ที่ได้จาก **Principal Component Analysis (PCA)**  
  - ใช้ Label `Class` เป็นตัวบ่งชี้ (0 = ปกติ, 1 = Fraud)  

- **Feature ที่ใช้ในโมเดล:**  
  - `V1 - V28` → ข้อมูลที่ได้จาก PCA ไม่สามารถตีความเชิงธุรกิจได้โดยตรง  
  - `Amount` → จำนวนเงินที่ทำธุรกรรม  
  - `Time` → เวลาที่ธุรกรรมเกิดขึ้น (หน่วยเป็นวินาที หลังจากธุรกรรมแรก)  
  - `Class` → Target Label (0 = ปกติ, 1 = Fraud)  

⚠️ **ความไม่สมบูรณ์ของข้อมูล**  
- **Class Imbalance** → ข้อมูล Fraud มีจำนวนน้อยมากเมื่อเทียบกับข้อมูลปกติ  
- **Feature Scaling** → ข้อมูล `Amount` และ `Time` ต้องถูก Normalized ก่อนใช้ในโมเดล  
- **No Missing Values** → ข้อมูลไม่มีค่า Missing แต่ต้องปรับสมดุลของ Class ด้วย SMOTE  
""")

# ✅ หัวข้อ 2: การเตรียมข้อมูล (Data Preparation)
st.subheader("📊 การเตรียมข้อมูล (Data Preparation)")
st.markdown("""
- **Feature Scaling:** ใช้ **StandardScaler** เพื่อปรับค่าคุณสมบัติให้อยู่ในช่วงที่เหมาะสม
- **Class Imbalance Handling:** เนื่องจากข้อมูล Fraud มีสัดส่วนน้อยมาก เราใช้ **SMOTE (Synthetic Minority Over-sampling Technique)** เพื่อสร้างตัวอย่าง Fraud เพิ่มเติมให้สมดุลกับข้อมูลปกติ
- **Feature Selection:** เลือกใช้เฉพาะ Features ที่สำคัญ โดยใช้ **PCA (Principal Component Analysis)** หรือ **Feature Importance จาก Random Forest**
""")

# ✅ หัวข้อ 3: ทฤษฎีของอัลกอริทึมที่พัฒนา
st.subheader("📖 ทฤษฎีของ Neural Network สำหรับ Fraud Detection")
st.markdown("""
- **Neural Network:** โมเดลที่มีหลายชั้น ใช้โครงสร้างของ **Fully Connected Layers (Dense Layers)**
- **Activation Functions:**
  - **ReLU (Rectified Linear Unit):** ใช้ใน Hidden Layers เพื่อให้โมเดลเรียนรู้ได้เร็วขึ้น
  - **Sigmoid:** ใช้ใน Output Layer เพื่อให้ค่าพยากรณ์อยู่ในช่วง 0-1 สำหรับปัญหาการจำแนกประเภท
- **Loss Function:** ใช้ **Binary Cross-Entropy** เนื่องจากปัญหานี้เป็น **Binary Classification (0 = Safe, 1 = Fraud)**
- **Optimizer:** ใช้ **Adam Optimizer** ซึ่งเป็นตัวปรับค่าที่ได้รับความนิยมสูงใน Deep Learning
""")

# ✅ หัวข้อ 4: ขั้นตอนการพัฒนาโมเดล (Model Development)
st.subheader("🛠️ ขั้นตอนการพัฒนาโมเดล")
st.markdown("""
1️⃣ **เตรียมข้อมูล (Data Preprocessing):**  
   - โหลดข้อมูล, ตรวจสอบ Missing Values, ทำ Feature Engineering  

2️⃣ **จัดการ Class Imbalance:**  
   - ใช้ **SMOTE หรือ Undersampling** เพื่อลดอคติของโมเดล  

3️⃣ **แบ่งข้อมูลเป็น Train/Test:**  
   - ใช้ **80% สำหรับเทรน** และ **20% สำหรับทดสอบ**  

4️⃣ **สร้าง Neural Network Model:**  
   - ใช้โครงสร้าง Fully Connected Layers (Dense Layers)  
   - เพิ่ม Dropout Layers เพื่อลด Overfitting  

5️⃣ **Train โมเดล:**  
   - ใช้ **Binary Cross-Entropy Loss** และ **Adam Optimizer**  
   - ใช้ **Batch Normalization** เพื่อช่วยให้โมเดล Converge เร็วขึ้น  

6️⃣ **Evaluate โมเดล:**  
   - ใช้ **Precision, Recall, F1-score และ AUC-ROC Curve**  

7️⃣ **Deploy โมเดล:**  
   - นำโมเดลที่ Train เสร็จไปใช้ผ่าน **Streamlit Web App หรือ REST API**
""")

# ✅ แหล่งอ้างอิง (References)
st.subheader("📚 แหล่งอ้างอิง")
st.markdown("""
- Kaggle Credit Card Fraud Dataset: [🔗 Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- SMOTE for Imbalanced Data: [🔗 Link](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)  
- Binary Cross-Entropy Loss: [🔗 Link](https://keras.io/api/losses/probabilistic_losses/#binarycrossentropy-class)  
- Adam Optimizer in Neural Networks: [🔗 Link](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)  
""")

# ✅ สรุป
st.success("✅ การพัฒนา Neural Network Model ครอบคลุมตั้งแต่ Data Preparation → Algorithm → Model Deployment!")

# ✅ ส่วนของผู้จัดทำ
st.markdown("---")
st.subheader("📌 จัดทำโดย")
st.markdown("""
👨‍🎓 **นาย จิตรภาณุ คุ้มดี**  
🎓 รหัสนักศึกษา: **6404062663037**  
📚 สาขา: **วิทยาการคอมพิวเตอร์**  
🏫 มหาวิทยาลัย: **มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ**  
""")
