#Self learning tic-tac-toe
from itertools import permutations
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
from keras.models import load_model
from scipy.ndimage.interpolation import shift
from ttt import *
import random as rm
from time import time
from matplotlib import pyplot as plt


st = time()
win_counter=0
loose_counter=0
draw_counter=0

n = 1000 # number of training iterations/games to be played.

mat = np.zeros((3,3))
mat = np.array([[0,0,1],[2,0,0],[0,0,0]])
def legal_move(mat,x):
    # x is a variable which stores whose turn it is
    l = []
    for i in range(3):
        for j in range(3):
            # print(place(x,i,j,mat))
            if (place(x,i,j,mat) != np.zeros((3,3))).any():
                l.append(place(x,i,j,mat))
    return l
# print(legal_move(mat,1))
# model = load_model(r'tic-tac-toe.h5')
# model.summary()    

model = Sequential()
model.add(Dense(18, input_dim=9,kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(9, kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,kernel_initializer='normal'))

learning_rate = 0.001
momentum = 0.8

sgd = SGD(lr=learning_rate, momentum=momentum,nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd)
model.summary()

def move_selector(model,mat,x):
    t = []
    s = []
    l = legal_move(mat,x)
    # print(len(l))
    for i in l:
        s = model.predict(i.reshape(1,9))
        # print('s',s)
        # print(type(s))
        t.append(s[0][0])
    return l[t.index(max(t))],max(t) #max(t) is the score

def easy(mat,x):
    #chooses a random position from the legal_move_generator
    # l = legal_move(mat,x)
    return legal_move(mat,x)[rm.randint(0,len(legal_move(mat,x))-1)]
def any_2(mat,x):
    if (mat[:]==np.array([x,0,x])).all() or (mat[:]==np.array([x,x,0])).all() or (mat[:]==np.array([0,x,x])).all():
        return 1
    return 0
def check_for_two(mat,x):
    # l = legal_move(mat,x)
    # for k in l:
        for i in range(3):
                if any_2(mat[i:i+1,:],x):
                    return 1
                elif any_2(np.transpose(mat[:,i:i+1]),x):
                    return 1
                elif any_2(mat.diagonal(),x):
                    return 1
                elif any_2(np.fliplr(mat).diagonal(),x):
                    return 1
        return 0
def cut(l,i):
    l1 = []
    for j in range(len(l)):
        if not (l[j]==i).all():
            l1.append(l[j])
    return l1
def hard(mat,x):
    l = legal_move(mat,x)
    for i in l:
        if win(i)==x:
            # print('w')
            return i
    for i in l:
        if check_for_two(i,x):
            # print('chk')
            return i
    if x==1:        
        for i in l:
            l1 = legal_move(i,2)
            for j in l1:
                if win(j)==2:
                    # print("i",i)
                    # print("j",j)
                    # print('chkn1')
                    l = cut(l,i)
    elif x==2:
        for i in l:
            l1 = legal_move(i,1)
            for j in l1:
                if win(j)==1:
                    # print("i",i)
                    # print("j",j)
                    # print('chkn2')
                    l = cut(l,i)
    # print("length of l: ",len(l))
    # print(l)
    if len(l)==0:
        return easy(mat,x)
    return l[rm.randint(0,len(l)-1)]



        


#train the model
trend = []
def train(model,win_counter,loose_counter,draw_counter,mode):
    # mode= 'pc'
    trend.append(win_counter)
    status=False
    scores= []
    board_list =[]
    corrected_scores_list=[]
    mat = np.zeros((3,3))
    print(mat)
    print("Start the game")
    print("Mode: ",mode)
    q = rm.randint(1,2)
    # q=1
    if q ==1:
        print("Computer is 1\n trainer is 2")
        print("Comupter will start")
        while(1):
            mat,s1 = move_selector(model,mat,1)
            print(mat)
            # print("score " ,s1)
            scores.append(s1)
            board_list.append(mat)
            # print("scores length: ",len(scores))
            # print("board list length ",len(board_list))
            # print(f'row win: {row_win(mat)}')
            # print(f'column win: {column_win(mat)}')
            # print(f'diagonal win: {diagonal_win(mat)}')
            # print("win condition" , win(mat))
            # print('draw condition: ',(mat[:,:] ==np.zeros([3,3])) )
            # print("draw condition" , draw(mat))
            if draw(mat)==1:
                if win(mat)==1:
                    status='win'
                    print(f'Computer has won!')
                    win_counter+=1
                    break
                elif win(mat)==2:
                    status='lost'
                    print("Trainer has won")
                    loose_counter+=1
                    break    
                status='draw'
                print("Draw!")
                draw_counter+=1
                break
            if win(mat)==0:
                if mode=='easy':
                    mat = easy(mat,2)
                elif mode == 'hard':
                    mat=hard(mat,2)
                else:
                    mat,g = move_selector(model1,mat,2)

                print(mat)
                # print(f'row win: {row_win(mat)}')
                # print(f'column win: {column_win(mat)}')
                # print(f'diagonal win: {diagonal_win(mat)}')
                # print("win mat",win(mat))
                if draw(mat)==1:
                    if win(mat)==1:
                        status='win'
                        print(f'Computer has won!')
                        win_counter+=1
                        break
                    elif win(mat)==2:
                        status='lost'
                        print("Trainer has won")
                        loose_counter+=1
                        break    
                    status='draw'
                    print("Draw!")
                    draw_counter+=1
                    break
                if win(mat)==2:
                    status='lost'
                    print("Trainer has won")
                    loose_counter+=1
                    break
                # print('draw condition: ',(mat[:,:] ==np.zeros([3,3])) )
                # print(draw(mat))
                
            if win(mat)==1:
                status='win'
                print(f'Computer has won!')
                win_counter+=1
                break
            elif win(mat)==2:
                status='lost'
                print("Trainer has won")
                loose_counter+=1
                break

            
    elif q==2:
        
        print("Computer is 2\n trainer is 1")
        print("Trainer will start")
        while(1):
                if mode=='easy':
                    mat = easy(mat,1)
                elif mode == 'hard':
                    mat=hard(mat,1)
                else:
                    mat,g = move_selector(model1,mat,1)
                print(mat)
                # print(f'row win: {row_win(mat)}')
                # print(f'column win: {column_win(mat)}')
                # print(f'diagonal win: {diagonal_win(mat)}')
                # print("win mat",win(mat))
                if draw(mat)==1:
                    if win(mat)==1:
                        status='lost'
                        print(f'Trainer has won!')
                        loose_counter+=1
                        break
                    elif win(mat)==2:
                        status='win'
                        print("Computer has won")
                        win_counter+=1
                        break    
                    status='draw'
                    print("Draw!")
                    draw_counter+=1
                    break
                # if win(mat)==1:
                #     status='lost'
                #     print("Trainer has won")
                #     break
                # print('draw condition: ',(mat[:,:] ==np.zeros([3,3])) )
                # print(draw(mat))
                if win(mat)==0:
                    mat,s1 = move_selector(model,mat,2)
                    print(mat)
                    # print("score " ,s1)
                    scores.append(s1)
                    board_list.append(mat)
                    # print("scores length: ",len(scores))
                    # print("board list length ",len(board_list))
                    # print(f'row win: {row_win(mat)}')
                    # print(f'column win: {column_win(mat)}')
                    # print(f'diagonal win: {diagonal_win(mat)}')
                    # print("win condition" , win(mat))
                    # print('draw condition: ',(mat[:,:] ==np.zeros([3,3])) )
                    # print("draw condition" , draw(mat))
                    if draw(mat)==1:
                        if win(mat)==1:
                            status='lost'
                            print(f'Trainer has won!')
                            loose_counter+=1
                            break
                        elif win(mat)==2:
                            status='win'
                            print("Computer has won")
                            win_counter+=1
                            break    
                        status='draw'
                        print("Draw!")
                        draw_counter+=1
                        break
                if win(mat)==1:
                    status='lost'
                    print(f'Trainer has won!')
                    loose_counter+=1
                    break
                elif win(mat)==2:
                    status='win'
                    print("Computer has won")
                    win_counter+=1
                    break  
    # print("scores",scores)
    board_list = np.array(board_list)
    # print("new board list:",board_list)
    # print(board_list.shape)
    if status=='win':
        corrected_scores_list=shift(scores,-1,cval=1.0)
    elif status=='lost':
        corrected_scores_list=shift(scores,-1,cval=-1.0)
    elif status=='draw':
        corrected_scores_list=shift(scores,-1,cval=0)
    x = board_list
    y = corrected_scores_list
    # print(len(x))
    # print(len(y))
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
    
    # shuffle x and y in unison
    x,y=unison_shuffled_copies(x,y)
    x=x.reshape(-1,9) 
    
    # update the weights of the model, one record at a time
    model.fit(x,y,epochs=1,batch_size=1,verbose=0)
    return model,y,status,win_counter,loose_counter,draw_counter,trend

for i in range(1,n+1):
    choice = rm.randint(1,2)
    # choice=1
    if choice==1:
        model,y,result,win_counter,loose_counter,draw_counter,trend = train(model,win_counter,loose_counter,draw_counter,'easy')
    elif choice==2:
        model,y,result,win_counter,loose_counter,draw_counter,trend = train(model,win_counter,loose_counter,draw_counter,'hard')
    if i%1000==0:
        rr = [j for j in range(i+1)]
        # print(rr)
        # print(trend)
        plt.plot(rr,trend)
        plt.xlabel("Games") 
        plt.ylabel("Trend")
        plt.show()

model.save(r'tic-tac-toe.h5')
et = time()

print(f"Time take to run: {(et-st)/60} hours")
print("Win frequency:" , win_counter)
print("Win percentage:\n",(win_counter/n)*100)

print("Loose frequency:" , loose_counter)
print("Loose percentage:\n",(loose_counter/n)*100)

print("Draw frequency:" , draw_counter)
print("Draw percentage:\n",(draw_counter/n)*100)

rr = [i for i in range(n)]
plt.plot(rr,trend)
plt.xlabel("Games") 
plt.ylabel("Trend")
plt.show()
