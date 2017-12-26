import csv

if __name__ == "__main__":


	input_path = "training_sets/all/inputs/"
	output_path =  "training_sets/all/outputs/"

	with open("%s/1503367642.csv" %input_path,'rU') as input_file:

		reader = csv.reader(input_file)

		rows = [row for row in reader]

		i=0
		for row in rows:
			k=0
			for val in row:
				rows[i][k] = float(rows[i][k])
				k+=1
			i+=1

	new_file= open("%s/1503367642.txt" %input_path, 'w')

	new_file.write(str(rows))

	new_file.close()


	with open("%s/1503367642.csv" %output_path,'rU') as output_file:

		reader = csv.reader(output_file)

		rows = [row for row in reader]

		i=0
		for row in rows:
			k=0
			for val in row:
				rows[i][k] = float(rows[i][k])
				k+=1
			i+=1


	new_file= open("%s/1503367642.txt" %output_path, 'w')

	new_file.write(str(rows))

	new_file.close()

