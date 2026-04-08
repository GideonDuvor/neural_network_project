from src.load_data import load_data
from src.preprocessing import preprocess_data
from src.train import train_model, save_model

print("STARTING NEURAL NETWORK MODEL")

df = load_data('data/Admission.csv')  # CHANGE THIS

X, y = preprocess_data(df)

model = train_model(X, y)

save_model(model)

print("Neural network pipeline executed successfully!")