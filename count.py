count_larger_ten = 0
count_larger_five = 0
count = 0
with open('train_triplets.txt') as tt:
    for line in tt:
        count += 1
        played = int(line.split('\t')[2])
        if played >= 10:
            count_larger_ten += 1
        if played >= 5:
            count_larger_five += 1


print("count > 10 is {}".format(count_larger_ten))
print("count > 5 is {}".format(count_larger_five))
print("count  is {}".format(count))


# count > 10 is  2477559
# count > 5  is  7261443
# count      is 48373586