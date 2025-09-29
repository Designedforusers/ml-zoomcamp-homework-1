# Machine Learning Course - Car Fuel Efficiency Dataset

## Overview
This project is for a machine learning course focusing on analyzing car fuel efficiency data using Python, pandas, and numpy.

## Dataset
- **Source**: Car fuel efficiency dataset from GitHub (alexeygrigorev/datasets)
- **File**: `car_fuel_efficiency.csv`
- **Size**: 9,704 rows × 11 columns
- **Target variable**: `fuel_efficiency_mpg`

### Features
- `engine_displacement`: Engine size
- `num_cylinders`: Number of cylinders (some missing values)
- `horsepower`: Engine horsepower (some missing values)
- `vehicle_weight`: Vehicle weight
- `acceleration`: Acceleration time
- `model_year`: Model year
- `origin`: Country of origin
- `fuel_type`: Type of fuel
- `drivetrain`: Drive train type
- `num_doors`: Number of doors (some missing values)

### Data Quality Notes
- Some columns have missing values that will need to be handled:
  - num_cylinders: 482 missing
  - horsepower: 708 missing
  - acceleration: 930 missing
  - num_doors: 502 missing

## Project Structure
- `main.py`: Main analysis script for exploring the dataset
- `car_fuel_efficiency.csv`: The downloaded dataset
- Python environment configured with pandas and numpy

## Recent Changes
- 2025-09-29: Initial setup
  - Installed Python 3.11, pandas, and numpy
  - Downloaded car fuel efficiency dataset
  - Created basic data exploration script
  - Set up Python workflow for running analysis

## User Preferences
- Working on machine learning course materials
- Using Python with pandas and numpy for data analysis

## Next Steps
Ready for machine learning analysis including:
- Data cleaning and preprocessing
- Exploratory data analysis
- Feature engineering
- Model building and evaluation