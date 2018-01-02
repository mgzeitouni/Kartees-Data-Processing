import numpy as np

def get_slope_intercept(y2,y1,x2,x1):

	slope = (y2-y1)/(x2-x1)

	intercept = y2 - slope * x2

	return intercept, slope



def connect_points(matrix):

	# Make first prices equal
	value = 0.0
	counter = 0

	# Grab first value
	while value == 0.0:

		current_row = matrix[counter]

		value = current_row[1]

		counter+=1

	# Populate beginning of array
	first_val = value
	value = 0.0
	counter = 0

	while value == 0.0:

		current_row = matrix[counter]

		value = current_row[1]

		matrix[counter,1] = first_val

		counter+=1

	# Also populate end points
	value = 0.0
	counter = matrix.shape[0]-1

	# Grab last value
	while value == 0.0:

		current_row = matrix[counter]

		value = current_row[1]

		counter-=1
	
	# Populate end of array

	last_val = value
	value = 0.0
	counter = matrix.shape[0]-1

	while value == 0.0:

		current_row = matrix[counter]

		value = current_row[1]

		matrix[counter,1] = last_val

		counter-=1

	
	# Populate middle values by connecting nearby points
	#row = 0
	row_num = 0
	y1 = None
	y2=None

	for row in matrix:

		current_y = row[1]
		current_x = row[0]
		
		if current_y==0.0:

			x_in_question = current_x

			# if row_num==6479:
			# 	pdb.set_trace()
			# Continue in array for next positive val
			i = 1
			while current_y==0.0:
				x2 = matrix[row_num+i,0]
				current_y = matrix[row_num+i,1]
				y2 = current_y
				i+=1

			b, m = get_slope_intercept(y2,y1,x2,x1)

			new_y_val = m * x_in_question + b

			matrix[row_num,1] = new_y_val


		else:
			y1 = current_y
			x1 = current_x

		row_num+=1

	return matrix
