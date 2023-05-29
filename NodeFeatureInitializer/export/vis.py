import joblib

model_path = "./topic_pca_model.bin"
pca = joblib.load(model_path)
print(sum(pca.explained_variance_ratio_))