import os
os.system("clear")

# a = int(input("Enter an integer "))
# print(a)


# try:
#     a = int(input("Enter an integer "))
#     print(a)
# except ValueError:
#     print("Please enter an integer!")


# while True:
#         try:
#             x = int(input("Please enter a number: "))
#             break
#         except ValueError:
#             print("Oops!  That was no valid number.  Try again...")
# print("Your number is: " + str(x))

# a = int(input("Enter numerator "))
# b = int(input("Enter denominator "))
# c = a/b

# print(c)

# try:
#     a = int(input("Enter numerator "))
#     b = int(input("Enter denominator "))
#     c = a/b
#     print(c)
# except ZeroDivisionError:
#     print("You've tried to divide by zero!")

# def divide():
#     try:
#         a = int(input("Enter numerator "))
#         b = int(input("Enter denominator "))
#         c = a/b
#         return(c)
#     except ZeroDivisionError:
#         print("Cannot divide by zero -- Check denominator!")

# divide()

def divide(a, b):
    try:
        c = a/b
        return(c)
    except ZeroDivisionError:
        print("Cannot divide by zero -- Check denominator!")

divide(5,0)

