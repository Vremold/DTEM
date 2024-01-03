import joblib

model_path = "./topic_pca_model.bin"
pca = joblib.load(model_path) 

# 输出模型的方差. 这个值表示了PCA模型对于数据降维前后保留的信息的比率. 
# 1 最好; 0 最差. 
print(sum(pca.explained_variance_ratio_))  

