from my_proj.sum_func import add, add_positive


# add two numbers
def test_sum_func_add_4_5():
    assert add(4, 5) == 9
# Both numbers are positive or greater than 0
def test_sum_func_add_positive_3_5():
    assert add_positive(3, 5) == 8
# first number is negative, the other is positive
def test_sum_func_add_positive_2_4():
    assert add_positive(-2, 4) == None
# first number is positive, the other is negative
def test_sum_func_add_positive_3_6():
    assert add_positive(3,-6) == None
# Both Negative
def test_sum_func_add_positive_2_5():
    assert add_positive(-2,-5) == None