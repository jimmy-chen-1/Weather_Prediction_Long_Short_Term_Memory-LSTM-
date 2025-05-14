# Weather Prediction Using Long Short-Term Memory (LSTM)

This repository contains code for predicting weather using an LSTM (Long Short-Term Memory) model. The project demonstrates how to preprocess weather data, train an LSTM model, and make predictions for future weather conditions.


<img width="1299" alt="Screenshot 2025-05-14 at 11 34 49 AM" src="https://github.com/user-attachments/assets/0f93c0cb-92fc-496b-8b8a-10c861180e37" />

## Table of Contents
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [How to Use](#how-to-use)
- [Configuration](#configuration)
- [License](#license)

---

## Features
- Preprocess weather data for model training.
- Train an LSTM model to predict weather metrics.
- Visualize weather predictions for better understanding and analysis.

---

## Setup Instructions

### Prerequisites
Make sure you have the following installed:
1. Python 3.7 or higher
2. Required Python libraries (listed in `requirements.txt`)

### Clone the Repository
Clone this repository to your local machine:

```bash
git clone https://github.com/jimmy-chen-1/Weather_Prediction_Long_Short_Term_Memory-LSTM-.git
cd Weather_Prediction_Long_Short_Term_Memory-LSTM-
```

### Install Dependencies
Install the required Python libraries using pip:

```bash
pip install -r requirements.txt
```

---

## Configuration

### Add an API Key
This project may require weather data from an external API, such as [OpenWeatherMap](https://openweathermap.org/). Follow these steps to configure your API key:

1. Sign up on [OpenWeatherMap](https://openweathermap.org/) and get your API key.
2. Create a `.env` file in the root of the project, and add your API key like this:

```env
API_KEY=your_api_key_here
```

3. The application will automatically load the API key when executed.

---

## How to Use

1. **Prepare Weather Data**  
   - If you're using API-based data:  
     Run the script to download and preprocess data.
   - If you're using local data:  
     Place your weather dataset in the `data/` folder and update the preprocessing scripts accordingly.

2. **Train the LSTM Model**  
   Run the following command to train the model:

   ```bash
   python train.py
   ```

   This will preprocess the data, train the LSTM model, and save the trained model in the `models/` directory.

3. **Make Predictions**  
   After the model is trained, use the following command to make predictions:

   ```bash
   python predict.py
   ```

   This will generate predictions and save the results in the `outputs/` directory.

4. **Visualize Predictions**  
   Use the provided visualization scripts to create plots for the predicted weather data. For example:

   ```bash
   python visualize.py
   ```

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to suggest additional features or report bugs by creating an issue in this repository.
