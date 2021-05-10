
def use_logging(func):
	def wrapper(*args, **kwargs):
		print("%s is running" % func.__name__)
		return func()
	return wrapper

@use_logging
def foo():
	print('i am a foo')


@use_logging
def bar():
	print("I am a bar")


print(foo.__name__)
print(bar.__name__)
bar()