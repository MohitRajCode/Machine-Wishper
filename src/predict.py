from pathlib import Path
import joblib
import pandas as pd

# Locate the saved model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "machinewhisperer_model.joblib"

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]


def predict_failure(
    machine_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
):
    # Arrange sensor readings in the exact format expected by the model
    machine_data = pd.DataFrame(
        [
            {
                "Type": machine_type,
                "Air temperature [K]": air_temperature,
                "Process temperature [K]": process_temperature,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
            }
        ]
    )

    # Get the model's predicted failure probability
    probability = model.predict_proba(machine_data)[0, 1]

    # Apply the selected threshold
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "result": "Failure Risk" if prediction else "No Failure Predicted",
        "failure_probability": round(float(probability), 4),
        "threshold": threshold,
    }


if __name__ == "__main__":
    result = predict_failure(
        machine_type="M",
        air_temperature=300.0,
        process_temperature=310.0,
        rotational_speed=1500,
        torque=40.0,
        tool_wear=100,
    )

    print(result)
