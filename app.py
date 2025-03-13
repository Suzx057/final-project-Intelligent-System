import streamlit as st

# ✅ ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Final Project IS 2567-2",
    layout="wide",
    page_icon="🏡",
    initial_sidebar_state="collapsed"
)

# ✅ ใช้ CSS **ลบ Sidebar + ปรับ UI สวยๆ**
st.markdown("""
    <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
            visibility: hidden !important;
            width: 0px !important;
            height: 0px !important;
        }
        body {
            background-color: #1A1A2E; 
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .title {
            text-align: center; 
            font-size: 44px; 
            font-weight: bold; 
            color: #4CAF50;
            margin-bottom: 5px;
            letter-spacing: 1px;
        }
        .subtitle {
            text-align: center; 
            font-size: 20px; 
            color: #B0BEC5;
            margin-bottom: 20px;
        }
        .menu-container {
            display: flex; 
            justify-content: center; 
            gap: 15px; 
            margin-bottom: 25px;
        }
        .menu-item {
            padding: 12px 24px; 
            border-radius: 12px; 
            background: linear-gradient(135deg, #007AFF, #00D4FF);
            color: white; 
            font-weight: bold; 
            cursor: pointer; 
            transition: 0.3s;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            font-size: 16px;
            text-align: center;
            border: none;
            outline: none;
        }
        .menu-item:hover {
            transform: translateY(-3px);
            background: linear-gradient(135deg, #005BBB, #0099FF);
            box-shadow: 0px 6px 15px rgba(0, 0, 0, 0.3);
        }
        .info-box {
            background: #16213E;
            padding: 20px;
            border-radius: 12px;
            color: white;
            font-size: 16px;
            width: 75%;
            margin: auto;
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3);
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin-bottom: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ ตั้งค่าหน้าเริ่มต้น
if "page" not in st.session_state:
    st.session_state["page"] = "about_ml"

# ✅ แสดงหัวข้อหลัก
st.markdown('<div class="title">🏡 Final Project IS 2567-2</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">✨ Machine Learning & Neural Network ✨</div>', unsafe_allow_html=True)

# ✅ เมนูหลัก (4 ปุ่ม)
menu_cols = st.columns(4)
menu_labels = ["📖 About ML", "📊 ML Model Demo", "📖 About NN", "📊 NN Model Demo"]
menu_pages = ["about_ml", "ml_demo", "about_nn", "nn_demo"]

for i in range(4):
    with menu_cols[i]:
        if st.button(menu_labels[i], key=f"menu_{i}"):
            st.session_state["page"] = menu_pages[i]

# ✅ แสดงหน้าตามที่เลือก
if st.session_state["page"] == "about_ml":
    st.markdown('<div class="info-box"> \
        <h4>📖 About Machine Learning</h4> \
        <p>อธิบายพื้นฐานของ Machine Learning, การเตรียมข้อมูล, อัลกอริทึม และการพัฒนาโมเดล</p> \
        </div>', unsafe_allow_html=True)
    exec(open("pages/1_data_preparation.py", encoding="utf-8").read())

elif st.session_state["page"] == "ml_demo":
    st.markdown('<div class="info-box"> \
        <h4>📊 Machine Learning Model Demo</h4> \
        <p>แสดงการทำงานของโมเดล Machine Learning เช่น House Price Prediction</p> \
        </div>', unsafe_allow_html=True)
    exec(open("pages/2_ml_model.py", encoding="utf-8").read())

elif st.session_state["page"] == "about_nn":
    st.markdown('<div class="info-box"> \
        <h4>📖 About Neural Network</h4> \
        <p>อธิบายพื้นฐานของ Neural Network, การเตรียมข้อมูล, โครงสร้างโมเดล และการพัฒนา</p> \
        </div>', unsafe_allow_html=True)
    exec(open("pages/5_nn_demo.py", encoding="utf-8").read())

elif st.session_state["page"] == "nn_demo":
    st.markdown('<div class="info-box"> \
        <h4>📊 Neural Network Model Demo</h4> \
        <p>แสดงการทำงานของโมเดล Neural Network เช่น Credit Card Fraud Detection</p> \
        </div>', unsafe_allow_html=True)
    exec(open("pages/3_nn_model.py", encoding="utf-8").read())
