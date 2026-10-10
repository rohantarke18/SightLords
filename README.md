# StoreFlow AI - basic prototype

Run order (inside the project folder, with the virtual environment active):

    pip install -r requirements.txt
    python step1_track.py --video your_video.mp4
    python step2_zones.py
    python step3_analyze.py
    python step3b_render_video.py --video your_video.mp4
    streamlit run step4_dashboard.py

No video yet? Test steps 3 and 4 with fake data:

    python make_demo_data.py
    python step3_analyze.py
    streamlit run step4_dashboard.py
