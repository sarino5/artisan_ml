import regression as r

test = r.Regression([2,5,6],[0], 1, 'linear')
test_cost = r.Compute_cost([2,5,6],[5,11,13],[2.0386], 0.758, 'linear')
test_gradient = r.Compute_gradient([2,5,6],[5,11,13],[2], 0.5, 'linear')

print(test.X)
print(test.w)
print(test.b)
print(test.f())
print(test.g())
print(test_cost.compute_cost())
print(test_gradient.min_w(), test_gradient.min_b())