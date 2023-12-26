from keras.models import load_model
import numpy as np
# from itertools import permutations
import random

m = load_model(r'tic-tac-toe.h5')
m.summary()
scores = []

def legal_move(mat,x):
    # x is a variable which stores whose turn it is
    l = []
    for i in range(3):
        for j in range(3):
            # print(place(x,i,j,mat))
            if (place1(x,i,j,mat) != np.zeros((3,3))).any():
                l.append(place1(x,i,j,mat))
    return l
def move_selector(model,mat,x):
    t = []
    s = []  
    l4 = legal_move(mat,x)
    # print(len(l4))
    for i in l4:
        s = model.predict(i.reshape(1,9))
        # print('s',s)
        # print(type(s))
        t.append(s[0][0])
    return l4[t.index(max(t))],max(t)

def place1(p,i,j,t):
    t1=t.copy()
    if t1[i,j]==0:
            t1[i,j] = p
    else: 
        return np.zeros((3,3))
    return t1
mat = np.zeros((3,3))
l = []
def place(p,i,j,t):
    if (i,j) not in l1:
        t[i,j] = p
        l1.append((i,j))
    else: 
        print("illegal move!try again")
    return t
def other_diagonal(t):
    if t[0,2]==1 and t[1,1] == 1 and t[2,0] == 1:
        return 1
    elif t[0,2]==2 and t[1,1] == 2 and t[2,0] == 2:
        return 2
    return 0
def row_win(t):
    for i in range(3):
        if (t[i:i+1,:] == np.array([1,1,1])).all():
            return 1
        elif (t[i:i+1,:] == np.array([2,2,2])).all():
            return 2
def column_win(t):
    for i in range(3):
        if (t[:,i:i+1] == np.array([1,1,1])).all():
            return 1
        elif (t[:,i:i+1] == np.array([2,2,2])).all():
            return 2
def diagonal_win(t):
    if (t.diagonal()==np.array([1,1,1])).all() or other_diagonal(t)==1:
        return 1
    elif (t.diagonal()==np.array([2,2,2])).all() or other_diagonal(t)==2:
        return 2

def win(t):
    if row_win(t) == 1 or column_win(t) ==1 or diagonal_win(t) ==1:
        return 1
    elif row_win(t) == 2 or column_win(t) ==2 or diagonal_win(t) ==2:
        return 2
def draw(t):
    if (t[:,:] ==np.zeros([3,3])).any():
        return 0
    return 1
def print_game(board):
    rows = len(board)
    cols = len(board)
    print("-------------------")
    for r in range(rows):
        print(board[r][0], " |", board[r][1], "|", board[r][2])
        print("-------------------")
    return board

def computer_move(mat,loop=0):
    
    mat,s1 = move_selector(m,mat,i)
    print_game(mat)
    if win(mat) == 1:
        print("Player 1 won")
        chk =1
        loop = 1
    elif win(mat)==2:
        print("Player 2 won")
        chk = 1
        loop =1
    elif draw(mat):
        print("Draw")
        chk = 1
        loop = 1
    return loop,mat


def player_move(mat,loop=0):
	mat = place(i,int(input("Enter row index: \n")),int(input("Enter column index: \n")),mat)
	print_game(mat)
	if win(mat) ==1:
						print("Player 1 won")
						chk=1
						loop=1
	elif win(mat)==2:
						print("Player 2 won")
						chk=1
						loop=1
	elif draw(mat):
						print("Draw!")
						chk=1
						loop=1
	return loop,mat
status = ''
chk = 0
while chk==0:
	print_game(mat)
	l = [1,2]
	l1= []
	choice = random.randint(1,2)
	# choice=2
	if choice ==1:
		loop = 0
		while loop==0:
			for i in l:
				if i==1:
					print(f'Player {i} turn:')
					loop,mat = computer_move(mat)
				elif i==2:
					print(f'Player {i} turn: ')
					loop,mat = player_move(mat)
	
	elif choice==2:
		loop = 0
		while loop==0:
			for i in l:
			
				if i==1:
					print(f'Player {i} turn:')
					loop,mat = player_move(mat)
				elif i==2:
					print(f'Player {i} turn: ')
					loop,mat = computer_move(mat)
	mat = np.zeros((3,3))
			



    
		


		


