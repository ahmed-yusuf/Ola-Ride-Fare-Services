import streamlit as st
import datetime

import pandas as pd
import numpy as np

df = pd.read_csv('train.csv', low_memory=False)
df = df.sample(frac=0.1)

df.head(2)

df.info()

df.rename(columns={'vendor+AF8-id': "vendor_id",
                   'pickup+AF8-loc': "pickup_loc",
                   'drop+AF8-loc': "drop_loc",
                   'driver+AF8-tip': "driver_tip",
                   'mta+AF8-tax': "mta_tax",
                   'pickup+AF8-time': "pickup_time",
                   'drop+AF8-time': "drop_time",
                   'num+AF8-passengers': "num_passengers",
                   'toll+AF8-amount': "toll_amount",
                   'payment+AF8-method': "payment_method",
                   'rate+AF8-code': "rate_code",
                   'stored+AF8-flag': "stored_flag",
                   'extra+AF8-charges': "extra_charges",
                   'improvement+AF8-charge': "improvement_charge",
                   'total+AF8-amount': "total_amount"
                   }, inplace=True)

df.head(2)

df.dropna(inplace=True)

df.isnull().sum()

for i in ['pickup_time', 'drop_time']:
    df[i] = pd.to_datetime(df[i])

df.isnull().sum()

df.info()

df['duration'] = df.drop_time - df.pickup_time

df['duration'] = df.duration.dt.total_seconds()

df.isnull().sum()


for i in ['driver_tip', 'mta_tax', 'toll_amount', 'extra_charges', 'improvement_charge', 'total_amount']:
    df[i] = pd.to_numeric(df[i], errors='coerce')

round(df.isnull().sum() / len(df) * 100, 2)

df.dropna(inplace=True)

df2 = df.copy()

df2.isnull().sum()

df2.drop(['pickup_time', "drop_time", 'ID', "vendor_id", "drop_loc", "pickup_loc", "stored_flag", "mta_tax",
          "improvement_charge", "payment_method"], axis=1, inplace=True)

from sklearn.ensemble import GradientBoostingRegressor

df2.nunique()


X = df2.drop('total_amount', axis=1)
y = df2['total_amount']

X.isnull().sum()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.metrics import mean_absolute_error, r2_score

# define the model
model = GradientBoostingRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, predictions)
print(f" Mean Absolute Error:", mae)

# Calculate R-squared (R²)
r_squared = r2_score(y_test, predictions)
print(f" R-squared:", r_squared)




with st.form('Enter basic details Below'):
    # Streamlit form
    st.title('Taxi Fare Prediction')

    # Input fields
    driver_tip = st.number_input('Driver Tip')
    distance = st.number_input('Distance')
    num_passengers = st.number_input('Number of Passengers', min_value=1, step=1)
    toll_amount = st.number_input('Toll Amount',min_value=1.0)
    rate_code = st.number_input('Rate Code', min_value=1.0)
    extra_charges = st.number_input('Extra Charges')
    pickup_time = st.time_input('Enter pickup time', step=datetime.timedelta(minutes=1))
    drop_time = st.time_input('enter your drop time', step=datetime.timedelta(minutes=1))

    # Check if the form is submitted
    submit = st.form_submit_button('Click Here to get your cab Fare')

    if submit:
        # Concatenate time inputs with some date to create valid datetime objects
        pickup_datetime = pd.to_datetime('2022-01-01 ' + pickup_time.strftime('%H:%M:%S'))
        drop_datetime = pd.to_datetime('2022-01-01 ' + drop_time.strftime('%H:%M:%S'))
        # Calculate duration
        duration = drop_datetime - pickup_datetime
        duration1 = duration.total_seconds()
        data = np.array([driver_tip, distance, num_passengers, toll_amount, rate_code, extra_charges, duration1])

        # Reshape the input array to have one row and multiple columns
        data_reshaped = data.reshape(1, -1)

        # Now, you can use the reshaped data for prediction
        fare = model.predict(data_reshaped)
        st.write('Total time spent on trip is', duration1//60, 'minutes')
        st.write('Your total Fare is: $',round(fare[0],2))
