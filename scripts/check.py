import pandas as pd

test = pd.read_csv("data/heart_test.csv")
pred = pd.read_csv("predictions.csv")

same = (test["id"].astype(int).values == pred["id"].astype(int).values).all()
print("ID order matches:", same)

if not same:
    diff_idx = (test["id"].astype(int).values != pred["id"].astype(int).values).nonzero()[0][:10]
    print("First mismatches at rows:", diff_idx.tolist())
    for i in diff_idx.tolist():
        print(i, "test_id=", int(test["id"].iloc[i]), "pred_id=", int(pred["id"].iloc[i]))