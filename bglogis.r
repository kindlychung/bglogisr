require(Matrix)
logreg = function(y, x) {
	x = as.matrix(x)
	x = apply(x, 2, scale)
	x = cbind(1, x)
	m = nrow(x)
	n = ncol(x)
	alpha = 2/m

	# b = matrix(rnorm(n))
	# b = matrix(summary(lm(y~x))$coef[, 1])
	b = matrix(rep(0, n))
	v = exp(-x %*% b)
	h = 1 / (1 + v)

	J = -(t(y) %*% log(h) + t(1-y) %*% log(1 -h))
	derivJ = t(x) %*% (h-y)


	while(1) {
		newb = b - alpha * derivJ
		v = exp(-x %*% newb)
		h = 1 / (1 + v)
		newJ = -(t(y) %*% log(h) + t(0-y) %*% log(1 -h))
		while((newJ - J) >= 0) {
			print("inner while...")
			# step adjust
			alpha = alpha / 1.15
			newb = b - alpha * derivJ
			v = exp(-x %*% newb)
			h = 1 / (1 + v)
			newJ = -(t(y) %*% log(h) + t(1-y) %*% log(1 -h))
		}
		if(max(abs(b - newb)) < 0.001) {
			break
		}
		b = newb
		J = newJ
		derivJ = t(x) %*% (h-y)
	}
	b
	w = h^2 * v
	# # hessian matrix of cost function
	hess = t(x) %*% Diagonal(x = as.vector(w)) %*% x
	seMat = sqrt(diag(solve(hess)))
	zscore = b / seMat
	cbind(b, zscore)
}

nr = 5000
nc = 5
# set.seed(17)
x = matrix(rnorm(nr*nc, 0, 999), nr)
x = apply(x, 2, scale)
# y = matrix(sample(0:1, nr, repl=T), nr)
h = 1/(1 + exp(-x %*% rnorm(nc)))
y = round(h)
y[1:round(nr/2)] = sample(0:1, round(nr/2), repl=T)


ntests = 300
testglm = function() {
	for(i in 1:ntests) {
		res = summary(glm(y~x, family=binomial))$coef
	}
	print(res)
}

testlogreg = function() {
	for(i in 1:ntests) {
		res = logreg(y, x)
	}
	print(res)
}

print(system.time(testlogreg()))
print(system.time(testglm()))
