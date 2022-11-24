from sklearn import linear_model as lm
# import statsmodels.api as sm

x = "iris"[["petal_lenght"]]
y = "iris"["petal_width"]


# fit the linear model

model = lm.LinearRegression()
results = model.fit(x,y)

# print the coefficients

print (model.intercept_, model.coef_)


