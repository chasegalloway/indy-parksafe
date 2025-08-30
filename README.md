# Indy ParkSafe

Indy ParkSafe is a machine learning-powered web application developed by Chase Galloway for the Congressional App Challenge 2025 that helps users find available parking spots in Indianapolis. The application predicts the probability of finding free parking at different meter locations based on various factors including time, date, location, and weather conditions. 

## Features

- Real-time parking availability predictions
- Interactive map visualization of parking meters
- Search by address with customizable radius
- Takes into account:
  - Time of day
  - Day of week
  - Holidays
  - Temperature

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chasegalloway/indy-parksafe.git
cd indy-parksafe
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Extract the data files:

## Usage

1. Start the FastAPI server:
```bash
uvicorn app:app --reload --port 8000
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

3. Enter an address in Indianapolis and specify:
   - Search radius (in meters)
   - Date and time
   - Temperature
   - Whether it's a holiday or not

The application will display:
- A map showing nearby parking meters
- Color-coded indicators for parking availability
- Detailed probability estimates for each meter
- Average availability in the selected area

## Data

The application uses several data sources:
- `indy_meters.csv`: Contains information about parking meter locations
- `indy_holidays_2025.csv`: Holiday calendar data
- `indy_parksafe_synth_shard_*.csv`: Training data for the machine learning model

## Model

The machine learning model takes into account various features:
- Temporal factors 
- Location features
- Environmental conditions 
- Special events 

The model is trained using historical parking data and is saved as `parksafe_model.pkl`.

## Development

To retrain the model with updated data:
```bash
python train.py
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
