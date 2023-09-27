import numpy as np
import cv2

def subtractive_clustering(img, density, c):
    # Initialize list of centers with the first pixel in the image
    centers = [(0, 0, img[0][0])]
    # Iterate over every pixel in the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Check if the distance between the current pixel and all the centers is greater than the density
            if np.min([np.sqrt((i-x)**2 + (j-y)**2) for x, y, _ in centers]) > density:
                print("distance",np.min([np.sqrt((i-x)**2 + (j-y)**2) for x, y, _ in centers]))
                # If the distance is greater than the density, add the pixel to the centers
                centers.append((i, j, img[i][j]))
            # If the number of centers is greater than c, break the loop
            if len(centers) >= c:
                break
        if len(centers) >= c:
            break
    # Return the centers
    return centers

def k_means_clustering(img, k, centers):
    # Create an array to store the cluster assignments for each pixel
    labels = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
    # Create an array to store the cluster means
    means = np.zeros((k, 3), dtype=np.float32)
    # Assign each center to a cluster
    for i, center in enumerate(centers):
        means[i] = center[2]
    # Run the k-means algorithm
    for iter in range(10):
        # Assign each pixel to the nearest cluster
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # Compute the distance between the pixel and all the cluster means
                distances = np.sqrt(np.sum((img[i][j] - means) ** 2, axis=1))
                # Assign the pixel to the nearest cluster
                labels[i][j] = np.argmin(distances)
        # Update the cluster means
        for i in range(k):
            cluster_pixels = img[labels == i]
            if len(cluster_pixels) > 0:
                means[i] = np.mean(cluster_pixels, axis=0)
    # Return the cluster assignments for each pixel
    return labels

# Load the image
img = cv2.imread('/Users/adarshjatti/Desktop/Datasets/Screenshot 2023-03-28 at 10.17.35 PM copy.png')
#cv2.imshow('original Image', img)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("gray")
print(gray)
cv2.imshow('original Image', gray)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)
print("blur")
print(len(blur))
# Apply subtractive clustering to find initial centers
centers = subtractive_clustering(blur, 50, 10)
print("centers")
print(centers)

# Apply k-means clustering to segment the image
labels = k_means_clustering(blur, len(centers), centers)
# Display the segmented image
segmented = np.zeros_like(img)
for i in range(len(centers)):
    segmented[labels == i] = centers[i][2]
cv2.imshow('Old Segmented Image', segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()
