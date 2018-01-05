from mnist import MNIST

mndata = MNIST('data')

images, labels = mndata.load_training()
index = random.randrange(0, len(images))  # choose an index ;-)
print(mndata.display(images[index]))
