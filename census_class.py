
from pydantic import BaseModel, Field

class Census(BaseModel):
    workclass: str = Field(..., example = 'State-gov')
    education: str = Field(..., example= 'Bachelors')
    marital_status: str = Field(..., alias = 'marital-status', example='Never-married')
    occupation: str = Field(..., example = 'Adm-clerical')
    relationship: str = Field(..., example = 'Not-in-family')
    race: str = Field(..., example = 'White')
    sex: str = Field(..., example = 'Male')
    native_country: str = Field(..., alias = 'native-country', 
                                    example = 'United-States')
    age: int = Field(..., example = 35)
    hours_per_week: int = Field(..., alias = 'hours-per-week',
                                     example = 50)