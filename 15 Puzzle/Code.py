#Colloborated with Bharat Mallala
import Queue as q
import sys
#Reading the input from the specified file
file_name=sys.argv[1]
inputfile=open(file_name,"r")
initial_board=[]
#used this piece of code from stack overflow
types=[int,int,int,int]
for line in inputfile.readlines():
  x=[t(e) for t,e in zip(types, line.split())]
  initial_board.append(x)
goal_state=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]]      
l=len(initial_board)
#Implementing A*
def solve(initial_board):
  #g(s)
  gs=0
  closed=set()
  fringe = q.PriorityQueue()
  fringe.put((hcost(initial_board)/3,[initial_board,'']))
  
  while len(fringe.queue) > 0:
    f=fringe.get()[1]
    s=f[0]
    spath=f[1]
    closed.add(str(s))
    if s==goal_state:
      return spath
    gs+=1
    for s1 in successors(s):
      if s1[0]==goal_state:
        return spath+' '+s1[1]
      if str(s1[0]) not in closed:
          closed.add(str(s1[0]))
          s1path=s1[1]
          if s1[0] in fringe.queue:
            for k in range(len(fringe.queue)):
              if fringe.queue[k][0]>(hcost(s1[0])+gs)/3:
                fringe.get(k)
          fringe.put(((hcost(s1[0])+gs)/3,[s1[0],spath+ ' '+s1path]))
     
           
#Finding the position of a tile
def where(s,tile):
  for n in range(l):
    for m in range(l):
      if s[n][m]==tile:
        r=n
        c=m
  return r,c

#Calculating the heuristic h(s)
def hcost(s):
  h=0
  conflict=0
  for tile in range(1,16):
      r,c=where(goal_state,tile)
      r_g,c_g=where(s,tile)
      for i in range(1,l):
        if r_g-i>=0:
          if s[r_g-i][c_g] == goal_state[r_g][c_g] and goal_state[r_g-i][c_g] == s[r_g][c_g]:
            conflict+=1
        if r_g+i<=l-1:
          if s[r_g+i][c_g] == goal_state[r_g][c_g] and goal_state[r_g+i][c_g] == s[r_g][c_g]:
            conflict+=1
        if c_g-i>=0:
          if s[r_g][c_g-i] == goal_state[r_g][c_g] and goal_state[r_g][c_g-i] == s[r_g][c_g]:
            conflict+=1
        if c_g+i <= l-1:
          if s[r_g][c_g+i] == goal_state[r_g][c_g] and goal_state[r_g][c_g+i] == s[r_g][c_g]:
            conflict+=1
          
      h+=abs(r-r_g)+abs(c-c_g)
  h+=2*conflict
  return h
#copying a 2dlist list by list
def copy1(s):
  copy_list1=[]
  for i in range(len(s)):
    copy_list=[]
    for j in range(len(s)):
      copy_list.append(s[i][j])
    copy_list1.append(copy_list)
  return copy_list1

#defining the successor function
def successors(s):
  successor=[]
  r,c=where(s,0)
  successor1=copy1(s)
  successor2=copy1(s)
  successor3=copy1(s)
  successor4=copy1(s)
  if r+1<l:
    successor1[r][c],successor1[r+1][c]=successor1[r+1][c],successor1[r][c]
    successor.append([copy1(successor1),'U1'+str(c+1)])
  if r+2<l:
    successor1[r+1][c],successor1[r+2][c]=successor1[r+2][c],successor1[r+1][c]
    successor.append([copy1(successor1),'U2'+str(c+1)])
  if r+3<l:
    successor1[r+2][c],successor1[r+3][c]=successor1[r+3][c],successor1[r+2][c]
    successor.append([copy1(successor1),'U3'+str(c+1)])
  
  if c+1<l:
    successor2[r][c],successor2[r][c+1]=successor2[r][c+1],successor2[r][c]
    successor.append([copy1(successor2),'L1'+str(r+1)])
  if c+2<l:
    successor2[r][c+1],successor2[r][c+2]=successor2[r][c+2],successor2[r][c+1]
    successor.append([copy1(successor2),'L2'+str(r+1)])
  if c+3<l:
    successor2[r][c+2],successor2[r][c+3]=successor2[r][c+3],successor2[r][c+2]
    successor.append([copy1(successor2),'L3'+str(r+1)])
  if r-1>=0:
    successor3[r][c],successor3[r-1][c]=successor3[r-1][c],successor3[r][c]
    successor.append([copy1(successor3),'D1'+str(c+1)])
  if r-2-1>=0:
    successor3[r-1][c],successor3[r-2][c]=successor3[r-2][c],successor3[r-1][c]
    successor.append([copy1(successor3),'D2'+str(c+1)])
  if r-3>=0:
    successor3[r-2][c],successor3[r-3][c]=successor3[r-3][c],successor3[r-2][c]
    successor.append([copy1(successor3),'D3'+str(c+1)])
  if c-1>=0:
    successor4[r][c],successor4[r][c-1]=successor4[r][c-1],successor4[r][c]
    successor.append([copy1(successor4),'R1'+str(r+1)])
  if c-2>=0:
    successor4[r][c-1],successor4[r][c-2]=successor4[r][c-2],successor4[r][c-1]
    successor.append([copy1(successor4),'R2'+str(r+1)])
  if c-3>=0:
    successor4[r][c-2],successor4[r][c-3]=successor4[r][c-3],successor4[r][c-2]
    successor.append([copy1(successor4),'R3'+str(r+1)])
  return successor
  

# successors if empty tile is in the corners



#Checking if the initial state is the goal state
if initial_board==goal_state:
  print("solution found","The sequence of moves is","None","Given state is itself the goal state")
else:
  print("Looking for solution.......")
  goal_path=solve(initial_board)
  print("solution found","The sequence of moves to take to reach the goal state from initial state are:")
  print(goal_path)
