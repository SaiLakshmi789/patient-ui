# generate_token.py
#import jwt, time
#secret = "VERY_LONG_RANDOM_SECRET"
#t = jwt.encode({"sub":"P123","scope":"patient_dashboard","exp":int(time.time())+3600}, secret, algorithm="HS256")
#print("http://localhost:8501/?token="+t)
import time, jwt, urllib.parse
secret="CHANGE_ME_TO_A_LONG_RANDOM_SECRET"
t=jwt.encode({"sub":"44382","scope":"patient_dashboard","exp":int(time.time())+3600}, secret, algorithm="HS256")
print("https://patient-report-dashboard.streamlit.app/?token="+urllib.parse.quote(t))

