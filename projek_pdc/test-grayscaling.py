import matplotlib.pyplot as plt
import cv2

sample = r'DSC_0367.JPG'
img_read_as_grayscale = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)
img_read_as_color = cv2.imread(sample, cv2.IMREAD_COLOR)
img_RGB_to_grayscale = cv2.cvtColor(img_read_as_color, cv2.COLOR_RGB2GRAY)
img_BGR_to_grayscale = cv2.cvtColor(img_read_as_color, cv2.COLOR_BGR2GRAY)
plt.imshow(img_read_as_grayscale)
plt.title('img_read_as_grayscale')
plt.show()
plt.imshow(img_read_as_color)
plt.title('img_read_as_color')
plt.show()
plt.imshow(img_RGB_to_grayscale)
plt.title('img_RGB_to_grayscale')
plt.show()
plt.imshow(img_BGR_to_grayscale)
plt.title('img_BGR_to_grayscale')
plt.show()

channel_avg_div_separately = img_read_as_color[:,:,0]/3+img_read_as_color[:,:,1]/3+img_read_as_color[:,:,2]/3
channel_avg_div_together = (img_read_as_color[:,:,0]+img_read_as_color[:,:,1]+img_read_as_color[:,:,2])/3
channel_sum = img_read_as_color[:,:,0]+img_read_as_color[:,:,1]+img_read_as_color[:,:,2]
plt.imshow(channel_avg_div_separately)
plt.title('channel_avg_div_separately')
plt.show()
plt.imshow(channel_avg_div_together)
plt.title('channel_avg_div_together')
plt.show()
plt.imshow(channel_sum)
plt.title('channel_sum')
plt.show()