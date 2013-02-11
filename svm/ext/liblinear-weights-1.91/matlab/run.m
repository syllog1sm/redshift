[y,xt] = libsvmread('../heart_scale');
w = load('../heart_scale.wgt');
model=train(w, y, xt);
[l,a]=predict(y, xt, model);

