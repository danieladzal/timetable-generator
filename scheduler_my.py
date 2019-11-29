import random
from operator import itemgetter
from utils import load_data, show_timetable, set_up, show_statistics, write_solution_to_file, write_statistics
from costs import check_hard_constraints, hard_constraints_cost, empty_space_groups_cost, empty_space_teachers_cost, \
    free_hour, is_schedule_found
from model import Schedule
import copy
import math


def initial_population(data, free, groups_empty_space, teachers_empty_space, subjects_order, population_size):
    """
    Sets up initial timetable for given classes by inserting in free fields such that every class is in its fitting
    classroom.
    """
    classes = data.classes
    population = []

    while len(population) < population_size:
        free_temp = copy.deepcopy(free)
        filled_temp = {}
        random.shuffle(free_temp)
        feasible = True
        for class_id, classs in classes.items():
            ind = 0
            found = False
            while not found:
                if ind >= len(free_temp):
                    feasible = False
                    break

                start_field = free_temp[ind]
                found = True

                # check if class won't start one day and end on the next
                start_time = start_field[0]
                end_time = start_time + int(classs.duration) - 1
                if start_time%12 > end_time%12:
                    found = False
                    ind += 1
                    continue

                
                # check if whole block for the class is free
                for i in range(1, int(classs.duration)):
                    field = (i + start_time, start_field[1])
                    if field not in free_temp:
                        found = False
                if found == False:
                    ind += 1
                    continue

                # secure that classroom fits
                if start_field[1] not in classs.classrooms:
                    ind += 1
                    found = False
                    continue

                if found:
                    # for group_index in classs.groups:
                        # add order of the subjects for group
                        # insert_order(subjects_order, classs.subject, group_index, classs.type, start_time)
                        # add times of the class for group
                        # for i in range(int(classs.duration)):
                            # groups_empty_space[group_index].append(i + start_time)
                    for i in range(int(classs.duration)):
                        filled_temp.setdefault(class_id, []).append((i + start_time, start_field[1]))        # add to filled
                        free_temp.remove((i + start_time, start_field[1]))                                # remove from free
                        # add times of the class for teachers
                        # teachers_empty_space[classs.teacher].append(i + start_time)
        if feasible:
            sch = Schedule(filled_temp, len(data.classrooms))
            population.append(sch)
            print("1 more done ")
        else:
            print("infeasible")
    return population

def exchange_two(matrix, filled, ind1, ind2):
    """
    Changes places of two classes with the same duration in timetable matrix.
    """
    fields1 = filled[ind1]
    filled.pop(ind1, None)
    fields2 = filled[ind2]
    filled.pop(ind2, None)

    for i in range(len(fields1)):
        t = matrix[fields1[i][0]][fields1[i][1]]
        matrix[fields1[i][0]][fields1[i][1]] = matrix[fields2[i][0]][fields2[i][1]]
        matrix[fields2[i][0]][fields2[i][1]] = t

    filled[ind1] = fields2
    filled[ind2] = fields1

    return matrix


##### Moje funkcije ######

# elitizam - vraca najbolji raspored u generaciji
def elite(population):
    best_schedule = population[0]
    for sch in population:
        if sch.cost_hard_constraints < best_schedule.cost_hard_constraints:
            best_schedule = sch
    return best_schedule

# selekcija 
"""
napravit cu eliminacijsku selekciju - umjesto odabira m najboljih, elminirat cu m "najlosijih" jedinki
i to na nacin da je vjerojatnost eliminacije proporcionalna hard_constraints_cost, a najbolja jedinka u generaciji je zasticena (vjerojatnost eliminacije je 0 - elitizam)
zapravo ne eliminiram nuzno najlosije jedinke ali najlosije jedinke imaju najmanju sansu za prezivljavanje
"""
def eliminate_selected(population, m):
    for i in range(m):
        min_cost = population[0].cost_hard_constraints
        sum_cost = 0
        cumulative_sum = 0
        for schedule in population:
            if schedule.cost_hard_constraints < min_cost:
                min_cost = schedule.cost_hard_constraints
            sum_cost += schedule.cost_hard_constraints
        sum_cost = sum_cost - len(population)*min_cost

        for schedule in population:
            # schedule.elimination_prob = (schedule.cost_hard_constraints - min_cost) / sum_cost
            cumulative_sum += schedule.cost_hard_constraints - min_cost
            schedule.cum_sum = cumulative_sum
        
        if sum_cost > 1:
            r = random.randrange(1, sum_cost)
            for i in range(1, len(population)):
                if r <= population[0].cum_sum:
                    population.remove(population[0])
                    break
                elif r > population[i-1].cum_sum and r <= population[i].cum_sum:
                    population.remove(population[i])
                    break
        else: 
            r = random.randrange(len(population))           ## if all chromosomes have equal cost remove any...
            population.remove(population[r])

# mutacija
"""
Take a schedule from population and randomly mutate one gene in it. 

"""
def mutate(population, m, data):
    num_classes = len(data.classes)
    classes = data.classes
    for mutation_id in range(m):
        sch = random.choice(population)
        class_id = random.randrange(0, num_classes)
        # find class duration, find sequence of terms in free of length = duration, replace old ones in sch.filled with them
        d = classes[class_id].duration
        old = sch.filled[class_id]
        sch.filled[class_id] = []
        sch.fill_matrix()
        sch.fill_free()
        ind = 0
        found = False
        while not found:
            if ind >= len(sch.free):
                sch.filled[class_id] = old
                break
            start_field = sch.free[ind]
            found = True

            # check if class won't start one day and end on the next
            start_time = start_field[0]
            end_time = start_time + int(d) - 1
            if start_time%12 > end_time%12:
                found = False
                ind += 1
                continue
            
            # check if whole block for the class is free
            for i in range(1, int(d)):
                field = (i + start_time, start_field[1])
                if field not in sch.free:
                    found = False
                if found == False:
                    ind += 1
                    continue

            # secure that classroom fits
            if start_field[1] not in classes[class_id].classrooms:
                ind += 1
                found = False
                continue

            if found:
                for i in range(int(d)):
                    sch.filled[class_id].append((i + start_time, start_field[1]))        # add to filled
                    sch.free.remove((i + start_time, start_field[1]))                                # remove from free
                    sch.fill_matrix()
                    sch.cost_hard_constraints, _, _, _, _ = hard_constraints_cost(sch.matrix, data)

# crossing - over
"""
Randomly choose two schedules from the population, radnomly choose place k to cut them and take first k classes from parent1 and the rest from parent2
"""

def reproduce(population, m, data):
    num_classrooms = len(data.classrooms)
    num_classes = len(data.classes)
    for i in range(m):
        parents = random.choices(population, k = 2)
        cut = random.randrange(0, num_classes)
        filled = {}

        for i in range(cut):
            filled[i] = parents[0].filled[i]
        for i in range(cut, num_classes):
            filled[i] = parents[1].filled[i]

        child = Schedule(filled, num_classrooms)
        child.fill_matrix()
        total, _, _, _, _ = hard_constraints_cost(child.matrix, data)
        child.cost_hard_constraints = total
        population.append(child)

# genetski algoritam
"""
def genetic_algorithm(initial_population, population_size):
    while(! postoji raspored s costom = 0):
        selektiraj bolje jedinke za reprodukciju()
        reprodukcijom stvori novu populaciju()
"""
def genetic_algorithm(population, mutation_count, death_birth_rate, data):
    generation = 0
    reached = True
    while is_schedule_found(population, data) == -1:
        generation += 1
        if generation > 2000: 
            reached = False
            break

        eliminate_selected(population, death_birth_rate)
        reproduce(population, death_birth_rate, data)
        mutate(population, mutation_count, data)

        print("Lowest cost in generation ",generation,":> ", elite(population).cost_hard_constraints)
        avg_cost = 0
        for sch in population:
            avg_cost += sch.cost_hard_constraints
        print("Average cost: ", avg_cost/len(population))
    if reached:
        show_timetable(elite(population).matrix)
        write_statistics(data, len(population), mutation_count, death_birth_rate, generation)
    else:
        print("to many generations.")
        write_statistics(data, len(population), mutation_count, death_birth_rate, -1)




def main():
    """
    free = [(row, column)...] - list of free fields (row, column) in matrix
    filled: dictionary where key = index of the class, value = list of fields in matrix

    subjects_order: dictionary where key = (name of the subject, index of the group), value = [int, int, int]
    where ints represent start times (row in matrix) for types of classes P, V and L respectively
    groups_empty_space: dictionary where key = group index, values = list of rows where it is in
    teachers_empty_space: dictionary where key = name of the teacher, values = list of rows where it is in

    matrix = columns are classrooms, rows are times, each field has index of the class or it is empty
    data = input data, contains classes, classrooms, teachers and groups
    """
    
    # filled = {}
    subjects_order = {}
    groups_empty_space = {}
    teachers_empty_space = {}
    file = 'ulaz2_optereceni.txt'

    data = load_data('test_files/' + file, teachers_empty_space, groups_empty_space, subjects_order)
    
    # ovdje cemo isprobavati razlicite velicine populacije i usporedivati rezultate
    population_size = 30
    death_birth_rate = 15
    mutation_count = 7

    
    _, free = set_up(len(data.classrooms))
    population = initial_population(data, free, groups_empty_space, teachers_empty_space, subjects_order, population_size)
    
    for schedule in population:
        schedule.fill_matrix()
        #show_timetable(schedule.matrix)
        total, _, _, _, _ = hard_constraints_cost(schedule.matrix, data)
        schedule.cost_hard_constraints = total
        #print(schedule.cost_hard_constraints)
    

    genetic_algorithm(population, mutation_count, death_birth_rate, data)

    
if __name__ == '__main__':
    main()
