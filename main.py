import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


# [==================== Find the equation ====================]

# Read image
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    print("The input file path argument is missing...")
    sys.exit()
print("Reading an image...")
input_file = os.path.basename(file_path)
input_file_name, ext = os.path.splitext(input_file)
image = cv2.imread(file_path)
output_image = cv2.imread(file_path)
height, width = image.shape[:2]

# Detect lines
print("Detecting lines in the image...")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 0, 50, apertureSize=3)
maxLineGap = ((height + width) / 2) * 0.05
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=20)

# Find top line and left line vertices
print("Finding the top line and left line vertices in the image...")
top_line_x = []
left_line_y = []
change_in_x = 10
change_in_y = 10
for line in lines:
    x1, y1, x2, y2 = line[0]
#     cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # Vertical lines
    if abs(x2 - x1) < change_in_x:
        top_line_x.append(x2)
    # Horizontal lines
    if abs(y2 - y1) < change_in_y:
        left_line_y.append(y1)

# Find the unit distance between vertices
print("Finding the unit distance between the vertices...")
units = []
top_line_x.sort()
left_line_y.sort()
for i in range(len(top_line_x) - 1):
    units.append(abs(top_line_x[i + 1] - top_line_x[i]))
for i in range(len(left_line_y) - 1):
    units.append(abs(left_line_y[i + 1] - left_line_y[i]))
mean = sum(units) / len(units)
for unit in list(units):
    if unit < mean:
        units.remove(unit)
unit_value = max(set(units), key=units.count)

# Remove duplicate x points in the top line vertices
print("Removing duplicate x points in the top line vertices...")
remove_index = []
for i in range(len(top_line_x) - 1):
    if abs(top_line_x[i + 1] - top_line_x[i]) < (unit_value / 2):
        prev_index = 0
        if i == 0:
            remove_index.append(i + 1)
        else:
            prev_index = i - 1
            dislocation_1 = abs(unit_value - (top_line_x[i + 1] - top_line_x[prev_index]))
            dislocation_2 = abs(unit_value - (top_line_x[i] - top_line_x[prev_index]))
            if dislocation_1 > dislocation_2:
                remove_index.append(i + 1)
            else:
                remove_index.append(i)
    elif abs(top_line_x[i + 1] - top_line_x[i]) > (unit_value * 1.5):
        remove_index.append(i + 1)
for i in range(len(remove_index) - 1, -1, -1):
    del top_line_x[remove_index[i]]

# Remove duplicate y points in the left line vertices
print("Removing duplicate y points in the left line vertices...")
remove_index = []
for i in range(len(left_line_y) - 1):
    if abs(left_line_y[i + 1] - left_line_y[i]) < (unit_value / 2):
        prev_index = 0
        if i == 0:
            remove_index.append(i + 1)
        else:
            prev_index = i - 1
            dislocation_1 = abs(unit_value - (left_line_y[i + 1] - left_line_y[prev_index]))
            dislocation_2 = abs(unit_value - (left_line_y[i] - left_line_y[prev_index]))
            if dislocation_1 > dislocation_2:
                remove_index.append(i + 1)
            else:
                remove_index.append(i)
    elif abs(left_line_y[i + 1] - left_line_y[i]) > (unit_value * 1.5):
        remove_index.append(i + 1)
for i in range(len(remove_index) - 1, -1, -1):
    del left_line_y[remove_index[i]]

# Save the grid graph's vertices
print("Saving the grid graph's vertices...")
grid_vertices = []
for i in range(len(top_line_x)):
    for j in range(len(left_line_y)):
        grid_vertices.append([top_line_x[i], left_line_y[j]])

# Find the RGB of the curve
print("Finding the RGB of the curve...")
rgb = np.array(image[0, 0], dtype=np.int64)
max_rgb_ssd = -1
curve_bgr = []
for i in range(width):
    for j in range(height):
        rgb = np.array(image[j, i], dtype=np.int64)
        rgb_sum = sum(rgb)
        possible_max_rgb_ssd = (abs(rgb[0] - rgb[1]) ** 2) + (abs(rgb[0] - rgb[2]) ** 2) + (abs(rgb[1] - rgb[2]) ** 2)
        if max_rgb_ssd < possible_max_rgb_ssd:
            max_rgb_ssd = possible_max_rgb_ssd
            curve_bgr = np.array(image[j, i], dtype=np.int64)

# Find the origin
print("Finding the origin of the graph...")
origin = [0, 0]
window_size = int(height * 0.007)
min_window_rgb = -1
for i in range(len(top_line_x)):
    possible_min_window_rgb = 0
    for j in range(int(abs(left_line_y[-1] - left_line_y[0]))):
        for k in range(-window_size, window_size + 1):
            if ((left_line_y[0] + j) >= 0) and ((left_line_y[0] + j) < height) and ((top_line_x[i] + k) >= 0) and ((top_line_x[i] + k) < width):
                rgb = np.array(image[left_line_y[0] + j, top_line_x[i] + k], dtype=np.int64)
                possible_min_window_rgb += sum(rgb)
#                 cv2.circle(output_image, (top_line_x[i] + k, left_line_y[0] + j), 1, (255, 200, 10), 1)
            else:
                possible_min_window_rgb += (255 * 3)
    if (min_window_rgb == -1) or (possible_min_window_rgb < min_window_rgb):
        min_window_rgb = possible_min_window_rgb
        origin[0] = top_line_x[i]
min_window_rgb = -1
for i in range(len(left_line_y)):
    possible_min_window_rgb = 0
    for j in range(int(abs(top_line_x[-1] - top_line_x[0]))):
        for k in range(-window_size, window_size + 1):
            if ((left_line_y[i] + k) >= 0) and ((left_line_y[i] + k) < height) and ((top_line_x[0] + j) >= 0) and ((top_line_x[0] + j) < width):
                rgb = np.array(image[left_line_y[i] + k, top_line_x[0] + j], dtype=np.int64)
                possible_min_window_rgb += sum(rgb)
#                 cv2.circle(output_image, (top_line_x[0] + j, left_line_y[i] + k), 1, (255, 200, 10), 1)
            else:
                possible_min_window_rgb += (255 * 3)
    if (min_window_rgb == -1) or (possible_min_window_rgb < min_window_rgb):
        min_window_rgb = possible_min_window_rgb
        origin[1] = left_line_y[i]

# Find the curve
print("Locating the curve...")
curve_points = []
curve_top = [0, height]
curve_bottom = [0, 0]
curve_left = [width, 0]
curve_right = [0, 0]
curve_positions = []
b, g, r = curve_bgr
epsilon = 15
for i in range(width):
    for j in range(height):
        bgr = np.array(image[j, i], dtype=np.int64)
        if (bgr[0] > (b - epsilon)) and (bgr[1] > (g - epsilon)) and (bgr[2] > (r - epsilon)):
            if (bgr[0] < (b + epsilon)) and (bgr[1] < (g + epsilon)) and (bgr[2] < (r + epsilon)):
                curve_points.append([i, j])
                if j <= curve_top[1]:
                    curve_top = [i, j]
                    curve_positions.append(curve_top)
                if j >= curve_bottom[1]:
                    curve_bottom = [i, j]
                    curve_positions.append(curve_bottom)
                if i <= curve_left[0]:
                    curve_left = [i, j]
                    curve_positions.append(curve_left)
                if i >= curve_right[0]:
                    curve_right = [i, j]
                    curve_positions.append(curve_right)
top_x_average = 0
bottom_x_average = 0
left_y_average = 0
right_y_average = 0
top_x_count = 0
bottom_x_count = 0
left_y_count = 0
right_y_count = 0
for i in range(len(curve_positions)):
    if curve_positions[i][1] == curve_top[1]:
        top_x_average += curve_positions[i][0]
        top_x_count += 1
    if curve_positions[i][1] == curve_bottom[1]:
        bottom_x_average += curve_positions[i][0]
        bottom_x_count += 1
    if curve_positions[i][0] == curve_left[0]:
        left_y_average += curve_positions[i][1]
        left_y_count += 1
    if curve_positions[i][0] == curve_right[0]:
        right_y_average += curve_positions[i][1]
        right_y_count += 1
    if top_x_count != 0:
        curve_top = [int(top_x_average / top_x_count), curve_top[1]]
    if bottom_x_count != 0:
        curve_bottom = [int(bottom_x_average / bottom_x_count), curve_bottom[1]]
    if left_y_count != 0:
        curve_left = [curve_left[0], int(left_y_average / left_y_count)]
    if right_y_count != 0:
        curve_right = [curve_right[0], int(right_y_average / right_y_count)]
curve_center = [int((curve_top[0] + curve_bottom[0]) / 2), int((curve_left[1] + curve_right[1]) / 2)]

# Map the grid's vertices to the unit distance graph
print("Mapping the grid's vertices to the unit distance graph...")
grid_hashmap = {}
origin_index = grid_vertices.index(origin)
for i in range(len(grid_vertices)):
    x = ((i // len(left_line_y)) - (origin_index // len(left_line_y)))
    y = -((i % len(left_line_y)) - (origin_index % len(left_line_y)))
    grid_hashmap[(grid_vertices[i][0], grid_vertices[i][1])] = [x, y]
#     cv2.putText(output_image, str(x), (grid_vertices[i][0], grid_vertices[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

# Map the curve points to the unit distance graph
print("Mapping the curve points to the unit distance graph...")
curve_hashmap = {}
for i in range(-int(unit_value / 2), int(unit_value / 2) + 1):
    for j in range(-int(unit_value / 2), int(unit_value / 2) + 1):
#         cv2.circle(output_image, (curve_top[0] + i, curve_top[1] + j), 1, (255, 200, 10), 1)
        if (curve_top[0] + i, curve_top[1] + j) in grid_hashmap:
            curve_top_map = grid_hashmap[(curve_top[0] + i, curve_top[1] + j)]
            curve_hashmap[(curve_top[0], curve_top[1])] = curve_top_map
        if (curve_left[0] + i, curve_left[1] + j) in grid_hashmap:
            curve_left_map = grid_hashmap[(curve_left[0] + i, curve_left[1] + j)]
            curve_hashmap[(curve_left[0], curve_left[1])] = curve_left_map
        if (curve_center[0] + i, curve_center[1] + j) in grid_hashmap:
            curve_center_map = grid_hashmap[(curve_center[0] + i, curve_center[1] + j)]
            curve_hashmap[(curve_center[0], curve_center[1])] = curve_center_map

# Find the equation
print("Deriving the equation...")
h = curve_hashmap[(curve_center[0], curve_center[1])][0]
k = curve_hashmap[(curve_center[0], curve_center[1])][1]
a = int((((curve_hashmap[(curve_left[0], curve_left[1])][0] - curve_hashmap[(curve_center[0], curve_center[1])][0]) ** 2) + ((curve_hashmap[(curve_left[0], curve_left[1])][1] - curve_hashmap[(curve_center[0], curve_center[1])][1]) ** 2)) ** 0.5)
b = int((((curve_hashmap[(curve_top[0], curve_top[1])][0] - curve_hashmap[(curve_top[0], curve_top[1])][0]) ** 2) + ((curve_hashmap[(curve_top[0], curve_top[1])][1] - curve_hashmap[(curve_center[0], curve_center[1])][1]) ** 2)) ** 0.5)
r = 1
first_term = ''
second_term = ''
right_equation = ''
if a == b:
    r = a
    a = 1
    b = 1
    right_equation = str(r) + '^2'
    if h == 0:
        first_term = 'x^2'
    elif h < 0:
        first_term = '(x+' + str(-h) + ')^2'
    else:
        first_term = '(x-' + str(h) + ')^2'
    if k == 0:
        second_term = 'y^2'
    elif k < 0:
        second_term = '(y+' + str(-k) + ')^2'
    else:
        second_term = '(y-' + str(k) + ')^2'
else:
    right_equation = '1^2'
    if h == 0:
        first_term = f'\\frac{{x^2}}{{{a}^2}}'
    elif h < 0:
        first_term = f'\\frac{{(x+{str(-h)})^2}}{{{a}^2}}'
    else:
        first_term = f'\\frac{{(x-{str(h)})^2}}{{{a}^2}}'
    if k == 0:
        second_term = f'\\frac{{y^2}}{{{b}^2}}'
    elif k < 0:
        second_term = f'\\frac{{(y+{str(-k)})^2}}{{{b}^2}}'
    else:
        second_term = f'\\frac{{(y-{str(k)})^2}}{{{b}^2}}'
latex_text = f'{first_term}+{second_term}={right_equation}'
print(f'Equation: (x-{h})^2/{a}^2+(y-{k})^2/{b}^2={r}^2')


# [==================== Plot for debugging ====================]

# Plot the grid vertices
for i in range(len(grid_vertices)):
    cv2.circle(output_image, (grid_vertices[i][0], grid_vertices[i][1]), 1, (0, 0, 255), int(height * 0.007))

# Plot the origin
cv2.circle(output_image, (origin[0], origin[1]), 1, (255, 0, 0), int(height * 0.015))

# Plot the curve points
for i in range(len(curve_points)):
    cv2.circle(output_image, (curve_points[i][0], curve_points[i][1]), 1, (0, 255, 0), int(height * 0.007))
cv2.circle(output_image, (curve_top[0], curve_top[1]), 1, (0, 0, 0), int(height * 0.015))
cv2.circle(output_image, (curve_bottom[0], curve_bottom[1]), 1, (0, 0, 0), int(height * 0.015))
cv2.circle(output_image, (curve_left[0], curve_left[1]), 1, (0, 0, 0), int(height * 0.015))
cv2.circle(output_image, (curve_right[0], curve_right[1]), 1, (0, 0, 0), int(height * 0.015))
cv2.circle(output_image, (curve_center[0], curve_center[1]), 1, (0, 0, 0), int(height * 0.015))

# Plot the curve points' locations
cv2.putText(output_image, f'({str(curve_top_map[0])}, {curve_top_map[1]})', (curve_top[0], curve_top[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(output_image, f'({str(curve_left_map[0])}, {curve_left_map[1]})', (curve_left[0], curve_left[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(output_image, f'({str(curve_center_map[0])}, {curve_center_map[1]})', (curve_center[0], curve_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)


# [==================== Write and show the images ====================]

# Write the output images
cv2.imwrite('output/output_image/' + input_file_name + '_output.jpg', output_image)

# Create and write a solution image
fontsize = 20
dpi = 100
fig, ax = plt.subplots(figsize=(4, 1), dpi=dpi)
ax.axis('off')
ax.text(0.5, 0.5, f'${latex_text}$', fontsize=fontsize, ha='center', va='center', color='black')
plt.tight_layout(pad=0)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('output/output_solution/' + input_file_name + '_solution.jpg', format='jpg', dpi=dpi)
plt.close(fig)
solution_image = cv2.imread('output/output_solution/' + input_file_name + '_solution.jpg')

# Show the images
cv2.namedWindow('Output Image', cv2.WINDOW_NORMAL)
cv2.moveWindow('Output Image', 50, 50)
cv2.imshow('Output Image', output_image)
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.moveWindow('Original Image', 25, 25)
cv2.imshow('Original Image', image)
cv2.imshow('Solution Image', solution_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
