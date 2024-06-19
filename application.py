from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application = Flask(__name__)
app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Delivery_person_Age=float(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings = float(request.form.get('Delivery_person_Ratings')),
            Weather_conditions = request.form.get('Weather_conditions'),
            Road_traffic_density = request.form.get('Road_traffic_density'),
            Vehicle_condition = int(request.form.get('Vehicle_condition')),
            Type_of_vehicle = request.form.get('Type_of_vehicle'),
            multiple_deliveries = float(request.form.get('multiple_deliveries')),
            Festival= request.form.get('Festival'),
            City = request.form.get('City'),
            Ordered_Date_Year= int(request.form.get('Ordered_Date_Year')),
            Ordered_Date_Month = int(request.form.get('Ordered_Date_Month')),
            Ordered_Date_Day= int(request.form.get('Ordered_Date_Day')),
            Time_OrderPicked_hours = int(request.form.get('Time_OrderPicked_hours')),
            Time_OrderPicked_mins= int(request.form.get('Time_OrderPicked_mins')),
            Time_Orderd_hours = int(request.form.get('Time_Orderd_hours')),
            Time_Orderd_mins= int(request.form.get('Time_Orderd_mins')),
            Distance_covered = float(request.form.get('Distance_covered'))
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)
    
if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)