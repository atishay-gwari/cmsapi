import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn import preprocessing
import joblib
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load the trained model and LabelEncoders
model = joblib.load(open("xgb_best_model.pkl", "rb"))
label_encoders = joblib.load("label_encoders.pkl")

# Define request and response models using Pydantic
class PredictionInput(BaseModel):
    DeductibleAmtPaid: float
    Gender: int
    Race: int
    RenalDiseaseIndicator: str
    State: int
    County: int
    NoOfMonths_PartACov: int
    NoOfMonths_PartBCov: int
    ChronicCond_Alzheimer: int
    ChronicCond_Heartfailure: int
    ChronicCond_KidneyDisease: int
    ChronicCond_Cancer: int
    ChronicCond_ObstrPulmonary: int
    ChronicCond_Depression: int
    ChronicCond_Diabetes: int
    ChronicCond_IschemicHeart: int
    ChronicCond_Osteoporasis: int
    ChronicCond_rheumatoidarthritis: int
    ChronicCond_stroke: int
    IPAnnualReimbursementAmt: int
    IPAnnualDeductibleAmt: int
    OPAnnualReimbursementAmt: int
    OPAnnualDeductibleAmt: int
    PatientType: str
    ClaimStartDt: str
    ClaimEndDt: str
    DOB: str

class PredictionOutput(BaseModel):
    predicted_claim_amount: float

# Create FastAPI instance
app = FastAPI()

# Preprocessing function
def preprocess_data(data):
    # Convert date columns to datetime objects
    data['ClaimStartDt'] = pd.to_datetime(data['ClaimStartDt'])
    data['ClaimEndDt'] = pd.to_datetime(data['ClaimEndDt'])
    data['DOB'] = pd.to_datetime(data['DOB'])

    data['ClaimSettlementDays'] = (data['ClaimEndDt'] - data['ClaimStartDt']).dt.days

    data['ClaimStartDtYear'] = data['ClaimStartDt'].dt.year
    data['ClaimStartDtMonth'] = data['ClaimStartDt'].dt.month
    data['ClaimStartDtDay'] = data['ClaimStartDt'].dt.day
    data['ClaimStartDtDayOfWeek'] = data['ClaimStartDt'].dt.dayofweek
    # Extract relevant features from date columns
    data['ClaimEndDtYear'] = data['ClaimEndDt'].dt.year
    data['ClaimEndDtMonth'] = data['ClaimEndDt'].dt.month
    data['ClaimEndDtDay'] = data['ClaimEndDt'].dt.day
    data['ClaimEndDtDayOfWeek'] = data['ClaimEndDt'].dt.dayofweek

    data['DOBYear'] = data['DOB'].dt.year
    data['DOBMonth'] = data['DOB'].dt.month
    data['DOBDay'] = data['DOB'].dt.day
    data['DOBDayOfWeek'] = data['DOB'].dt.dayofweek
    # Add other date-related features...

    # Drop unnecessary columns
    data.drop(columns=['ClaimStartDt', 'ClaimEndDt', 'DOB'], inplace=True)

    return data

# Prediction endpoint
@app.post("/predict/", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Preprocess incoming data
        input_df = pd.DataFrame([input_data.dict()])
        preprocessed_data = preprocess_data(input_df)

        # Apply label encoding
        for column, encoder in label_encoders.items():
            if column in preprocessed_data.columns:
                preprocessed_data[column] = encoder.transform(preprocessed_data[column])

        # Make predictions
        print("hi")
        print(preprocessed_data.columns)


        prediction = model.predict(preprocessed_data)

        # Prepare response
        response = PredictionOutput(predicted_claim_amount=prediction[0])
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)


# {
#   "DeductibleAmtPaid": 1068.0,
#   "Gender": 2,
#   "Race": 1,
#   "RenalDiseaseIndicator": "Y",
#   "State": 45,
#   "County": 780,
#   "NoOfMonths_PartACov": 12,
#   "NoOfMonths_PartBCov": 12,
#   "ChronicCond_Alzheimer": 2,
#   "ChronicCond_Heartfailure": 1,
#   "ChronicCond_KidneyDisease": 1,
#   "ChronicCond_Cancer": 2,
#   "ChronicCond_ObstrPulmonary": 1,
#   "ChronicCond_Depression": 1,
#   "ChronicCond_Diabetes": 2,
#   "ChronicCond_IschemicHeart": 1,
#   "ChronicCond_Osteoporasis": 2,
#   "ChronicCond_rheumatoidarthritis": 2,
#   "ChronicCond_stroke": 2,
#   "IPAnnualReimbursementAmt": 21260,
#   "IPAnnualDeductibleAmt": 2136,
#   "OPAnnualReimbursementAmt": 120,
#   "OPAnnualDeductibleAmt": 100,
#   "PatientType": "Inpatient",
#   "ClaimStartDt": "2009-09-09",
#   "ClaimEndDt": "2009-09-16",
#   "DOB": "1938-04-01"
# }