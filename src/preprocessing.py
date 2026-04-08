def preprocess_data(df):
    try:
        df = df.dropna()

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        print("Preprocessing complete")
        return X, y

    except Exception as e:
        print(f"Preprocessing error: {e}")