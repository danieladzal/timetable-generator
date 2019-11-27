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
            print("initializing population...")
        """
        else:
            count failures??
        """
    return population


def insert_order(subjects_order, subject, group, type, start_time):
    """
    Inserts start time of the class for given subject, group and type of class.
    """
    times = subjects_order[(subject, group)]
    if type == 'P':
        times[0] = start_time
    elif type == 'V':
        times[1] = start_time
    else:
        times[2] = start_time
    subjects_order[(subject, group)] = times





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


def valid_teacher_group_row(matrix, data, index_class, row):
    """
    Returns if the class can be in that row because of possible teacher or groups overlaps.
    """
    c1 = data.classes[index_class]
    for j in range(len(matrix[row])):
        if matrix[row][j] is not None:
            c2 = data.classes[matrix[row][j]]
            # check teacher
            if c1.teacher == c2.teacher:
                return False
            # check groups
            for g in c2.groups:
                if g in c1.groups:
                    return False
    return True


def mutate_ideal_spot(matrix, data, ind_class, free, filled, groups_empty_space, teachers_empty_space, subjects_order):
    """
    Function that tries to find new fields in matrix for class index where the cost of the class is 0 (taken into
    account only hard constraints). If optimal spot is found, the fields in matrix are replaced.
    """

    # find rows and fields in which the class is currently in
    rows = []
    fields = filled[ind_class]
    for f in fields:
        rows.append(f[0])

    classs = data.classes[ind_class]
    ind = 0
    while True:
        # ideal spot is not found, return from function
        if ind >= len(free):
            return
        start_field = free[ind]

        # check if class won't start one day and end on the next
        start_time = start_field[0]
        end_time = start_time + int(classs.duration) - 1
        if start_time % 12 > end_time % 12:
            ind += 1
            continue

        # check if new classroom is suitable
        if start_field[1] not in classs.classrooms:
            ind += 1
            continue

        # check if whole block can be taken for new class and possible overlaps with teachers and groups
        found = True
        for i in range(int(classs.duration)):
            field = (i + start_time, start_field[1])
            if field not in free or not valid_teacher_group_row(matrix, data, ind_class, field[0]):
                found = False
                ind += 1
                break

        if found:
            # remove current class from filled dict and add it to free dict
            filled.pop(ind_class, None)
            for f in fields:
                free.append((f[0], f[1]))
                matrix[f[0]][f[1]] = None
                # remove empty space of the group from old place of the class
                for group_index in classs.groups:
                    groups_empty_space[group_index].remove(f[0])
                # remove teacher's empty space from old place of the class
                teachers_empty_space[classs.teacher].remove(f[0])

            # update order of the subjects and add empty space for each group
            for group_index in classs.groups:
                insert_order(subjects_order, classs.subject, group_index, classs.type, start_time)
                for i in range(int(classs.duration)):
                    groups_empty_space[group_index].append(i + start_time)

            # add new term of the class to filled, remove those fields from free dict and insert new block in matrix
            for i in range(int(classs.duration)):
                filled.setdefault(ind_class, []).append((i + start_time, start_field[1]))
                free.remove((i + start_time, start_field[1]))
                matrix[i + start_time][start_field[1]] = ind_class
                # add new empty space for teacher
                teachers_empty_space[classs.teacher].append(i+start_time)
            break


def evolutionary_algorithm(matrix, data, free, filled, groups_empty_space, teachers_empty_space, subjects_order):
    """
    Evolutionary algorithm that tires to find schedule such that hard constraints are satisfied.
    It uses (1+1) evolutionary strategy with Stifel's notation.
    """
    n = 3
    sigma = 2
    run_times = 5
    max_stagnation = 200

    for run in range(run_times):
        print('Run {} | sigma = {}'.format(run + 1, sigma))

        t = 0
        stagnation = 0
        cost_stats = 0
        while stagnation < max_stagnation:

            # check if optimal solution is found
            loss_before, cost_classes, cost_teachers, cost_classrooms, cost_groups = hard_constraints_cost(matrix, data)
            if loss_before == 0 and check_hard_constraints(matrix, data) == 0:
                print('Found optimal solution: \n')
                show_timetable(matrix)
                break

            # sort classes by their loss, [(loss, class index)]
            costs_list = sorted(cost_classes.items(), key=itemgetter(1), reverse=True)

            # 10*n
            for i in range(len(costs_list) // 4):
                # mutate one to its ideal spot
                if random.uniform(0, 1) < sigma and costs_list[i][1] != 0:
                    mutate_ideal_spot(matrix, data, costs_list[i][0], free, filled, groups_empty_space,
                                      teachers_empty_space, subjects_order)
                # else:
                #     # exchange two who have the same duration
                #     r = random.randrange(len(costs_list))
                #     c1 = data.classes[costs_list[i][0]]
                #     c2 = data.classes[costs_list[r][0]]
                #     if r != i and costs_list[r][1] != 0 and costs_list[i][1] != 0 and c1.duration == c2.duration:
                #         exchange_two(matrix, filled, costs_list[i][0], costs_list[r][0])

            loss_after, _, _, _, _ = hard_constraints_cost(matrix, data)
            if loss_after < loss_before:
                stagnation = 0
                cost_stats += 1
            else:
                stagnation += 1

            t += 1
            # Stifel for (1+1)-ES
            if t >= 10*n and t % n == 0:
                s = cost_stats
                if s < 2*n:
                    sigma *= 0.85
                else:
                    sigma /= 0.85
                cost_stats = 0

        print('Number of iterations: {} \nCost: {} \nTeachers cost: {} | Groups cost: {} | Classrooms cost:'
              ' {}'.format(t, loss_after, cost_teachers, cost_groups, cost_classrooms))
# check if optimal solution is found
        loss_before, cost_classes, cost_teachers, cost_classrooms, cost_groups = hard_constraints_cost(matrix, data)
        if loss_before == 0 and check_hard_constraints(matrix, data) == 0:
            break


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
        min_cost = population[1].cost_hard_constraints
        sum_cost = 0
        cumulative_sum = 0
        for schedule in population:
            if schedule.cost_hard_constraints < min_cost:
                min_cost = schedule.cost_hard_constraints
            sum_cost += schedule.cost_hard_constraints
        sum_cost = sum_cost - len(population)*min_cost

        for schedule in population:
            schedule.elimination_prob = (schedule.cost_hard_constraints - min_cost) / sum_cost
            cumulative_sum += (schedule.cost_hard_constraints - min_cost)
            schedule.cum_sum = cumulative_sum
        r = random.random()*sum_cost

        for schedule in population:
            if r > population[population.index(schedule) - 1].cum_sum and r <= schedule.cum_sum:
                population.remove(schedule) 





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
    file = 'ulaz3.txt'

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
    write_statistics(data, population_size, mutation_count, death_birth_rate, -1)

    """
    eliminate_selected(population, 3)

    print("After eliminating 3 schedules...")

    for schedule in population:
        schedule.fill_matrix()
        show_timetable(schedule.matrix)
        total, _, _, _, _ = hard_constraints_cost(schedule.matrix, data)
        schedule.cost_hard_constraints = total
        print(schedule.cost_hard_constraints)

    reproduce(population, 2, len(data.classrooms), len(data.classes))
    print("After adding 2 new schedules by reproduction...")

    for schedule in population:
        schedule.fill_matrix()
        show_timetable(schedule.matrix)
        total, _, _, _, _ = hard_constraints_cost(schedule.matrix, data)
        schedule.cost_hard_constraints = total
        print(schedule.cost_hard_constraints)
    
    mutate(population, 1, len(data.classrooms), data.classes)
    for schedule in population:
        schedule.fill_matrix()
        #show_timetable(schedule.matrix)
        print(schedule.filled)
        total, _, _, _, _ = hard_constraints_cost(schedule.matrix, data)
        schedule.cost_hard_constraints = total
        print(schedule.cost_hard_constraints)
    """
    # genetic_algorithm(population, mutation_count, death_birth_rate, data)

    # evolutionary_algorithm(matrix, data, free, filled, groups_empty_space, teachers_empty_space, subjects_order)
    # print('STATISTICS')
    # show_statistics(matrix, data, subjects_order, groups_empty_space, teachers_empty_space)
    # simulated_hardening(matrix, data, free, filled, groups_empty_space, teachers_empty_space, subjects_order, file)
    


if __name__ == '__main__':
    main()
