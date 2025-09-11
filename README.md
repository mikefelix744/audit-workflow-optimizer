
Audit Workflow Optimizer — Prototype Demo
=======================================

What this prototype includes (saved in this folder):
- engagements_sample.csv : sample historical engagement dataset
- staff_sample.csv : sample staff dataset
- app.py : simple Streamlit demo that:
    * estimates engagement hours using a simple linear regression (from historicals)
    * suggests a baseline timeline
    * matches staff using a heuristic score
    * allows a simple scenario simulation for delays

How to run the demo locally:
1. Ensure you have Python 3.8+ and pip.
2. Install dependencies:
    pip install streamlit pandas numpy
3. Run the app:
    streamlit run app.py
4. The app will open in your browser. Use the sidebar to create a new engagement and click "Estimate & Match Staff".




