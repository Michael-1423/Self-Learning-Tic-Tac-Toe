import numpy as np

def place(p,i,j,t):
    t1=t.copy()
    if t1[i,j]==0:
            t1[i,j] = p
    else: 
        return np.zeros((3,3))
    return t1
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
    if (t.diagonal()==np.array([1,1,1])).all() or (np.fliplr(t).diagonal()==np.array([1,1,1])).all():
        return 1
    elif (t.diagonal()==np.array([2,2,2])).all() or (np.fliplr(t).diagonal()==np.array([2,2,2])).all():
        return 2

def win(t):
    if row_win(t) == 1 or column_win(t) ==1 or diagonal_win(t) ==1:
        return 1
    elif row_win(t) == 2 or column_win(t) ==2 or diagonal_win(t) ==2:
        return 2
    else:
        return 0
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