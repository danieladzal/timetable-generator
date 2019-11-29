from utils import load_data

def main():

	subjects_order = {}
	groups_empty_space = {}
	teachers_empty_space = {}
	file = 'ulaz3.txt'
	data = load_data('test_files/' + file, teachers_empty_space, groups_empty_space, subjects_order)
	

	# calculate how different types of classrooms are occupied
	r = 0
	n = 0
	a = 0
	k = 0
	s = 0
	rd = 0
	nd = 0
	ad=0
	kd=0
	sd=0
	for i in range(len(data.classes)):
		if data.classrooms[data.classes[i].classrooms[0]].type == "r":
			r+=1
			rd+=int(data.classes[i].duration)
		elif data.classrooms[data.classes[i].classrooms[0]].type == "n":
			n+=1
			nd+=int(data.classes[i].duration)
		elif data.classrooms[data.classes[i].classrooms[0]].type == "a":
			a+=1
			ad+=int(data.classes[i].duration)
		elif data.classrooms[data.classes[i].classrooms[0]].type == "k":
			k+=1
			kd+=int(data.classes[i].duration)
		elif data.classrooms[data.classes[i].classrooms[0]].type == "s":
			s+=1
			sd+=int(data.classes[i].duration)
	print("n = ", n, "r = ", r, "a =",a ,"k=",k,"s= ",s)
	print("nd = ", nd, " rd = ", rd, "ad = ", ad, "kd=",kd, "sd = ", sd)

	# how many teachers are there, what is their occupation level
	print("Ukupno nastavnika: ", len(data.teachers), ", ukupan broj predmeta: ", len(data.classes))

if __name__ == '__main__':
	main()
