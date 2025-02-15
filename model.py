class Class:

    def __init__(self, groups, teacher, subject, type, duration, classrooms):
        self.groups = groups
        self.teacher = teacher
        self.subject = subject
        self.type = type
        self.duration = duration
        self.classrooms = classrooms

    def __str__(self):
        return "Groups {} | Teacher '{}' | Subject '{}' | Type {} | {} hours | Classrooms {} \n"\
            .format(self.groups, self.teacher, self.subject, self.type, self.duration, self.classrooms)

    def __repr__(self):
        return str(self)


class Classroom:

    def __init__(self, name, type):
        self.name = name
        self.type = type

    def __str__(self):
        return "{} - {} \n".format(self.name, self.type)

    def __repr__(self):
        return str(self)


class Data:

    def __init__(self, groups, teachers, classes, classrooms):
        self.groups = groups
        self.teachers = teachers
        self.classes = classes
        self.classrooms = classrooms

class Schedule:

    def __init__(self, filled, num_classrooms):
        self.filled = filled
        self.cost_hard_constraints = 0
        self.matrix = [[None for x in range(num_classrooms)] for y in range(60)]
        self.elimination_prob = 0
        self.cum_sum = 0
        self.free = []

    def fill_matrix(self):
        for index, fields_list in self.filled.items():
                for field in fields_list:
                    self.matrix[field[0]][field[1]] = index
    
    def fill_free(self):
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                if self.matrix[i][j] is None:
                    self.free.append((i, j))








