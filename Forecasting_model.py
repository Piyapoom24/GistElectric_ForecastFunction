import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

# ฟังก์ชันดึงข้อมูลจาก API และแปลง พ.ศ. เป็น ค.ศ.
def fetch_data_from_api(api_url): # กำหนด input ของฟังก์ชันโดยรับ API url
    response = requests.get(api_url, timeout=10) #ขอชุดข้อมูลผ่าน API url โดยกำหนดเวลารอในการขอข้อมูล 10 วิทนาที
    response.raise_for_status() #เช็ค API status
    data = response.json() # แปลงข้อมูลจาก API เป็น json file

    df = pd.json_normalize(data['features']) # แปลง json file ให้เป็นในรูปแบบ Dataframe(ตาราง)

    if 'record_date' in df.columns: #แปลง column เวลาให้เป็น ค.ศ. (โดยพ.ศ.-543 = ค.ศ.)
        df['record_date'] = df['record_date'].apply(
        lambda x: f"{x.split('/')[0].zfill(2)}/{x.split('/')[1].zfill(2)}/{int(x.split('/')[2]) - 543}"
         if isinstance(x, str) and len(x.split('/')) == 3 else x
    )

    df['record_date'] = pd.to_datetime(df['record_date'], format='%d/%m/%Y', errors='coerce') #จัดการ format และแปลงtype record_date ที่เป็น Str ให้เป็น Datetime 
    df = df.sort_values(by='record_date') #เรียงข้อมูลตาม record_date 
    return df

# ฟังก์ชันส่งข้อมูลไปยัง API
def send_forecast_to_api(endpoint, data):
    response = requests.post(endpoint, json=data, timeout=10)
    response.raise_for_status()
    return response.json()

# คลาสสำหรับรวมอาคาร
class SarimaModel_SumAll:
    def __init__(self, df): #รับ input ที่ผ่านฟังก์ชัน feth_data_from_api 
        self.df = df #กำหนดตัวแปร คือ input
        self.data = None #กำหนดตัวแปรว่างๆไว้เพื่อเก็บค่า
        self.train = None #กำหนดตัวแปรว่างๆไว้เพื่อเก็บค่า
        self.model = None #กำหนดตัวแปรว่างๆไว้เพื่อเก็บค่า
        self.forecasts = None #กำหนดตัวแปรว่างๆไว้เพื่อเก็บค่า
    
    def train_model(self): #ฟังก์ชันในการ train model
        full_data = self.df #กำหนดัตวแปร โดยให้ input เป็นชื่อ full_data (กันความสับสน)

        #ต้องข้อมูลรวม หน่วยการใช้ไฟ(unit) ทั้งหมดเป็นรายเดือน เลยทำการรวมข้อมูล Groub ข้อมูลโดย record_date
        total_data = full_data.groupby(full_data['record_date'].dt.to_period('M'))['unit'].sum().to_timestamp()
        self.data = total_data #เก็บข้อมูลที่ทำการรวมแล้วไว้ใน self.data
        self.train = self.data[:] #แบ่ง train data โดยใช้ข้อมูลทั้งหมดในการ train  เก็บไว้ในตัวแปร self.train

        #train Arima(Seasonal) model โดยใช้ self.train และเก็บ model ไว้ใน self.model
        self.model = auto_arima(
            self.train,
            seasonal=True,
            m=12,
            d=None,
            max_d=1,
            D=1,
            max_p=4,
            max_q=4,
            information_criterion='aic',
            trend='ct',
            suppress_warnings=True,
            error_action='ignore'
        )
        return self.model
    
    def forecast(self): #ฟังก์ชันในการพยากรณ์ 
        self.forecasts = self.model.predict(n_periods=6) #ใช้model ที่เก็บไว้ในตัวแปร self.model พยากรณ์ 6เดือนถัดไป แล้วเก็บค่าไว้ใน self.forecasts
        return self.forecasts
    
    def forecast_to_api(self): #ฟังก์ชันในการส่งค่าพยากรณ์ไปยัง API
        last_date = self.train.index[-1] #ดึงวันสุดท้ายของ train data
        
        #สร้างช่วงเวลา 6 เดือน โดยเริ่มจากวันสุดท้ายของชุด train (freq='MS' คือ ความถี่เป็นเดือน)
        forecast_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=6, freq='MS')
        
        #สร้าง data frame โดยใช้ช่วงเวลา6เดือนที่เก็บไว้ในตัวแปร forecast_dates และ ค่าพยากรณ์ที่เก็บไว้ในตัวแปร self.forecasts
        forecast_df = pd.DataFrame({
            'record_date': forecast_dates.to_series().apply(lambda d: f"{d.day:02d}/{d.month:02d}/{d.year + 543}"), # แปลง ค.ศ. กลับเป็น พ.ศ.
            'unit': self.forecasts.astype(float) 
        })

        forecast_data = forecast_df.to_dict(orient='records') #แปลงข้อมูลให้อยู่ในรูปแบบ dict 
        return send_forecast_to_api('http://localhost:3000/api/forecast/buildingsum', forecast_data) #ส่งข้อมูลไปยัง API 

#คลาสโมเดลของแต่ละอาคาร 90อาคาร
class SarimaModel_90_Building:
    def __init__(self, fdata): #รับ input ที่ผ่านฟังก์ชัน feth_data_from_api 
        self.fdata = fdata #กำหนดตัวแปร คือ input
        self.data = {} #กำหนดตัวแปรว่างๆไว้เพื่อเก็บค่า
        self.models = {} #กำหนดตัวแปรว่างๆไว้เพื่อเก็บค่า
        self.forecasts = {} #กำหนดตัวแปรว่างๆไว้เพื่อเก็บค่า
        self.trains = {} #กำหนดตัวแปรว่างๆไว้เพื่อเก็บค่า
    
    def train_model(self): #ฟังก์ชันในการ train model
        full_data = self.fdata #กำหนดัตวแปร โดยให้ input เป็นชื่อ full_data (กันความสับสน)

        #ส่วนนี้เป็นส่วนในการแยกข้อมูลแต่ละอาคาร โดยใช้ หมายเลข building_id (1-90)
        for i in range(1, 91): #ให้ i แทน หมายเลข building_id ในแต่ละรอบตั้งแต่ 1-90
            df = full_data[full_data['building_id'] == i].copy() #ดึงข้อมูลแต่ละอาคาร โดยที่รอบที่ i แทน building_id
            if df.empty: #ถ้าเจอว่าdataว่าง ให้ข้ามข้อมูลชุดนั้นได้เลย
                print(f"No data for building_id {i}")
                continue
            df.index = df['record_date'] #แปลง index เป็นวันที่
            df = df[['unit', 'building_name']] #filter เอาแค่คอลัมน์ unit และ building id
            self.data[i] = df #เก็บข้อมูลไว้ใน self.data โดยแยกตาม หมายเลข building _id

        #ส่วนในการ train model โดยใช้ข้อมูลของแต่ละอาคาร
        for i in tqdm(range(1, 91)): #ให้ i แทน หมายเลข building_id ในแต่ละรอบตั้งแต่ 1-90
            if i not in self.data: #เช็คว่าข้อมูลของตึกไหนหาย ถ้าเจอก็ข้ามได้เลย เพื่อไม่ให้การ train หยุดทำงาน
                print(f"Skipping building {i}: No data available")
                continue 
            train = self.data[i]['unit'][:].dropna() #กำหนดตัวแปร train แทนข้อมูลของอาคารในแต่ละรอบ โดยที่รอบที่ i แทน building_id
            if train.empty: #ถ้าเจอว่าdataว่าง ให้ข้ามข้อมูลชุดนั้นได้เลย
                print(f"Skipping building {i}: No valid training data")
                continue

            #train Arima(Seasonal) model 
            model = auto_arima(
                train,
                seasonal=True,
                m=12,
                d=None,
                max_d=1,
                D=1,
                max_p=4,
                max_q=4,
                information_criterion='aic',
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            self.models[i] = model #เก็บ model ไว้ใน self.model ในแต่ละรอบ โดยที่รอบที่ i แทน building_id
            self.trains[i] = train #เก็บข้อมูลชุดtrainแต่ละรอบไว้
        return self.models

    def forecast(self): #ฟังก์ชันในการพยากรณ์ 
        for i in range(1, 91): #ให้ i แทน หมายเลข building_id ในแต่ละรอบตั้งแต่ 1-90
            model = self.models.get(i) #นำโมเดลที่เก็บไว้ใน self.models มาใช้ โดยรอบที่ i แทน หมายเลข building_id
            if model is not None: #ถ้ามีโมเดลให้ทำการ forecast
                try:
                    forecast = model.predict(n_periods=6) #ใช้model ที่เก็บไว้ในตัวแปร self.model พยากรณ์ 6เดือนถัดไป แล้วเก็บค่าไว้ใน self.forecasts
                    self.forecasts[i] = forecast #เก็บค่าพยากรณ์ ไว้ใน forecasts ในแต่ละรอบ โดยที่รอบที่ i แทน building_id
                except Exception as e: #ถ้าไม่มีให้แจ้งmodelที่หายไป
                    print(f"Forecast failed for building {i}: {e}") 
        return self.forecasts
    
    def forecast_to_api(self): #ฟังก์ชันในการส่งค่าพยากรณ์ไปยัง API
        forecast_data = [] #สร้างlistไว้เก็บค่าพยากรณ์ของแต่ละอาคาร
        for i in range(1, 91): #ให้ i แทน หมายเลข building_id ในแต่ละรอบตั้งแต่ 1-90
            forecast = self.forecasts.get(i) #นำค่าพยากรณ์จากที่เก็บไว้ใน self.forecast
            train = self.trains.get(i) #นำค่าพยากรณ์จากที่เก็บไว้ใน self.forecast
            if forecast is not None and train is not None: #ตรวจสอบว่า forecast และ train ไม่ใช่ค่า None 
                last_date = train.index[-1] #ดึงวันที่สุดท้ายจากข้อมูล train
                # สร้างลิสต์ของวันที่เริ่มต้นตั้งแต่เดือนถัดไปของ last_date จำนวน 6 เดือน โดยใช้ความถี่เป็นต้นเดือน ('MS' = Month Start)
                forecast_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=6, freq='MS') 
                for j, d in enumerate(forecast_dates): 
                    #เพิ่มข้อมูลพยากรณ์ลงในลิสต์ forecast_data ในรูปแบบ dict
                    forecast_data.append({
                        'building_id': i,
                        'record_date': f"{d.day:02d}/{d.month:02d}/{d.year + 543}", # แปลงวันที่ให้เป็นรูปแบบ พ.ศ.
                        'unit': float(forecast.iloc[j])  # ใส่ค่าพยากรณ์ในรูป float สำหรับ JSON
                    })
        return send_forecast_to_api('http://localhost:3000/api/forecast/90building', forecast_data)

#----------------------------------------------------------------------------------------------------------

# การใช้งาน
# 1.) กำหนด API url
api_url = 'http://localhost:3000/api/Energy'

# 2.) แปลงข้อมูลจาก API ให้เหมาะต่อการใช้งาน โดยผ่านฟังก์ชัน fetch_data_from_api()
df = fetch_data_from_api(api_url) 

# 3.) นำข้อมูลที่ผ่านการแปลงในข้อที่2 มาใช้
# 3.1) พยากรณ์สำหรับข้อมูลรวมทั้งหมด
Full_data = SarimaModel_SumAll(df) #เรียกใช้ คลาสสำหรับการพยากรณ์ โดยใช้ ข้อมูลที่แปลงมาจากข้อที่2
Full_data.train_model() #เรียกใช้ method train_model เพื่อtrain model
Full_data.forecast() #เรียกใช้ method forecast เพื่อพยากรณ์ค่า
response_sum = Full_data.forecast_to_api() #เรียกใช้ method forecast_to_api เพื่อส่งค่าพยากรณ์ไปยัง API
print("BuildingSum forecast response:", response_sum) #ส่งค่าพยากรณ์ไปยัง API

# 3.2) พยากรณ์สำหรับแต่ละอาคาร
Building_data = SarimaModel_90_Building(df) #เรียกใช้ คลาสสำหรับการพยากรณ์ โดยใช้ ข้อมูลที่แปลงมาจากข้อที่2
Building_data.train_model() #เรียกใช้ method train_model เพื่อtrain model
Building_data.forecast() #เรียกใช้ method forecast เพื่อพยากรณ์ค่า
response_90 = Building_data.forecast_to_api() #เรียกใช้ method forecast_to_api เพื่อส่งค่าพยากรณ์ไปยัง API
print("90Building forecast response:", response_90) #ส่งค่าพยากรณ์ไปยัง API