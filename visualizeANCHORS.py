import numpy as np
import matplotlib.pyplot as plt

import config


def main():
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	for anchors,depths in zip(config.ANCHORS, config.DEPTH_ANCHORS):
		for anchor, depth in zip(anchors,depths):
			ax.scatter(anchor[0],anchor[1], depth)
	plt.show()


if __name__ == '__main__':
	main()