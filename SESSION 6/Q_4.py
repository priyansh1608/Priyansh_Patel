model_a_rmse = 8
model_a_r2 = 0.65

model_b_rmse = 6
model_b_r2 = 0.55

print("Model A")
print("RMSE:", model_a_rmse)
print("R²:", model_a_r2)

print()

print("Model B")
print("RMSE:", model_b_rmse)
print("R²:", model_b_r2)

print()

print("Model B has lower RMSE, so its prediction errors are smaller.")
print("Model A has higher R², so it explains more variation in delivery time.")
print("The choice depends on whether lower prediction error or higher explained variance is more important.")