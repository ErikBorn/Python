#importing libraries
import turtle
import random
import time

#creating turtle screen
screen = turtle.Screen()
screen.title('DATAFLAIR-SNAKE GAME')
screen.setup(width = 1300, height = 700)
screen.tracer(0)
turtle.bgcolor('turquoise')

#Border variables
border_north = 290
border_west = -600
border_south = -300
border_east= 600

##creating a border for our game
turtle.speed(5)
turtle.pensize(4)
turtle.penup()
turtle.goto(-610,290)
turtle.pendown()
turtle.color('black')
turtle.forward(1220)
turtle.right(90)
turtle.forward(600)
turtle.right(90)
turtle.forward(1220)
turtle.right(90)
turtle.forward(600)
turtle.penup()
turtle.hideturtle()
colorized = random.randint(1,3)

#score
score = 0
delay = 0.05



#snake
snake = turtle.Turtle()
snake.speed(0)
snake.shape('square')
snake.color("black")
snake.penup()
snake.goto(200,0)
snake.direction = 'stop'

#snake number 2
snake_two = turtle.Turtle()
snake_two.speed(0)
snake_two.shape('square')
snake_two.color("white")
snake_two.penup()
snake_two.goto(-200,0)
snake_two.direction = 'stop'

#Random color changer
def change_color_number():
    global colorized
    colorized = random.randint(1,3)


def random_color():
    the_color = ''
    if colorized == 1:
        the_color = 'blue'
    elif colorized == 2:
        the_color = 'green'
    else:
        the_color = 'red'
    return the_color

def background_color_change():
    if score%40 == 10:
        screen.bgcolor('pink')
    elif score%40 == 20:
        screen.bgcolor('yellow')
    elif score%40 == 30:
        screen.bgcolor('orange')
    elif score%40 == 0 and score > 0:
        screen.bgcolor('turquoise')


def game_over():
    time.sleep(1)
    screen.clear()
    screen.bgcolor('turquoise')
    scoring.goto(0,0)
    scoring.write("    GAME OVER \n Your Score is {}".format(score),align="center",font=("Courier",30,"bold"))
    time.sleep(0)

#food
fruit = turtle.Turtle()
fruit.speed(1)
fruit.shape('circle')
fruit.color(random_color())
fruit.penup()
fruit.goto(30,30)

old_fruit=[]
old_fruit_two=[]

#scoring
scoring = turtle.Turtle()
scoring.speed(0)
scoring.color("black")
scoring.penup()
scoring.hideturtle()
scoring.goto(0,300)
scoring.write("Score :",align="center",font=("Courier",24,"bold"))



#######define how to move
def snake_go_up():
    if snake.direction != "down":
        snake.direction = "up"

def snake_go_down():
    if snake.direction != "up":
        snake.direction = "down"

def snake_go_left():
    if snake.direction != "right":
        snake.direction = "left"

def snake_go_right():
    if snake.direction != "left":
        snake.direction = "right"

def snake_move():
    if snake.direction == "up":
        y = snake.ycor()
        snake.sety(y + 20)

    if snake.direction == "down":
        y = snake.ycor()
        snake.sety(y - 20)

    if snake.direction == "left":
        x = snake.xcor()
        snake.setx(x - 20)

    if snake.direction == "right":
        x = snake.xcor()
        snake.setx(x + 20)

#######define how to move for Snake number 2
def snake_two_go_up():
    if snake_two.direction != "down":
        snake_two.direction = "up"

def snake_two_go_down():
    if snake_two.direction != "up":
        snake_two.direction = "down"

def snake_two_go_left():
    if snake_two.direction != "right":
        snake_two.direction = "left"

def snake_two_go_right():
    if snake_two.direction != "left":
        snake_two.direction = "right"

def snake_two_move():
    if snake_two.direction == "up":
        y = snake_two.ycor()
        snake_two.sety(y + 20)

    if snake_two.direction == "down":
        y = snake_two.ycor()
        snake_two.sety(y - 20)

    if snake_two.direction == "left":
        x = snake_two.xcor()
        snake_two.setx(x - 20)

    if snake_two.direction == "right":
        x = snake_two.xcor()
        snake_two.setx(x + 20)



# Keyboard bindings
screen.listen()
screen.onkeypress(snake_go_up, "Up")
screen.onkeypress(snake_go_down, "Down")
screen.onkeypress(snake_go_left, "Left")
screen.onkeypress(snake_go_right, "Right")

# Keyboard bindings for snake two
screen.onkeypress(snake_two_go_up, "w")
screen.onkeypress(snake_two_go_down, "s")
screen.onkeypress(snake_two_go_left, "a")
screen.onkeypress(snake_two_go_right, "d")

#main loop

while True:
        screen.update()
            #snake and fruit coliisions
        print(fruit.color())
        if (snake.distance(fruit) < 20) and (fruit.color() != ('blue','blue')):
                x = random.randint(border_west, border_east)
                y = random.randint(border_south, border_north)
                fruit.color(random_color())
                fruit.goto(x,y)
                scoring.clear()
                score+=1
                scoring.write("Score:{}".format(score),align="center",font=("Courier",24,"bold"))
                delay-=0.001

                ## creating new_ball
                new_fruit = turtle.Turtle()
                new_fruit.speed(0)
                new_fruit.shape('square')
                new_fruit.color(random_color())
                new_fruit.penup()
                old_fruit.append(new_fruit)

                change_color_number()
                fruit.color(random_color())

                #background color change
                background_color_change()

                
        if snake_two.distance(fruit)< 20:
                x = random.randint(border_west, border_east)
                y = random.randint(border_south, border_north)
                fruit.color(random_color())
                fruit.goto(x,y)
                scoring.clear()
                score+=1
                scoring.write("Score:{}".format(score),align="center",font=("Courier",24,"bold"))
                delay-=0.001

                ## creating new_ball
                new_fruit = turtle.Turtle()
                new_fruit.speed(0)
                new_fruit.shape('square')
                new_fruit.color(random_color())
                new_fruit.penup()
                old_fruit_two.append(new_fruit)

                change_color_number()
                fruit.color(random_color())
                

                #background color change
                background_color_change()

        #adding ball to snake
        for index in range(len(old_fruit)-1,0,-1):
                a = old_fruit[index-1].xcor()
                b = old_fruit[index-1].ycor()
                
                old_fruit[index].goto(a,b)
                                     
        if len(old_fruit)>0:
                a= snake.xcor()
                b = snake.ycor()
                old_fruit[0].goto(a,b)
        
        for index in range(len(old_fruit_two)-1,0,-1):
                a_two = old_fruit_two[index-1].xcor()
                b_two = old_fruit_two[index-1].ycor()
                
                old_fruit_two[index].goto(a_two,b_two)
                                     
        if len(old_fruit_two)>0:
                a_two = snake_two.xcor()
                b_two = snake_two.ycor()
                old_fruit_two[0].goto(a_two,b_two)

        snake_move()
        snake_two_move()


        
        ##snake and border collision    
        if snake.xcor()>border_east or snake.xcor()< border_west or snake.ycor()>border_north or snake.ycor()<border_south:
                game_over()
        
        if snake_two.xcor()>border_east or snake_two.xcor()< border_west or snake_two.ycor()>border_north or snake_two.ycor()<border_south:
                game_over()

        if snake.distance(snake_two)< 20:
                game_over()


        ## snake collision
        for food in old_fruit:
                if food.distance(snake) < 20 or food.distance(snake_two) < 20:
                        game_over()
        
        for food_two in old_fruit_two:
                if food_two.distance(snake_two) < 20 or food_two.distance(snake) < 20:
                    game_over()

        time.sleep(delay)

turtle.Terminator()