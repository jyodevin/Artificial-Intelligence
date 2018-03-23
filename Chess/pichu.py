#!/usr/bin/env python

#Bharat Mallala, Jyothi Pranavi Devineni, Harshit Krishnakumar
'''
This code uses alpha beta pruning with mini max algorithm to predict chess moves. The evaluation function is a weighted sum of number of pieces of each kind. 
'''
import sys
import math
from collections import Counter
import numpy as np
from random import shuffle

def print_board(board):
	print ()
	for row in board:
		col_str=''
		for col in row:
			col_str+=' '+col
		print(col_str)
	print ()
	return

def open_rows(board,r,c):
	rows = []
	new_r = r
	while new_r+1 in range(8):
		new_r += 1
		if board[new_r][c] in our_pieces:
			break
		rows.append(new_r)
		if board[new_r][c] in their_pieces:
			break
		
	new_r = r
	while new_r-1 in range(8):
		new_r -= 1
		if board[new_r][c] in our_pieces:
			break
		rows.append(new_r)
		if board[new_r][c] in their_pieces:
			break
	return rows


def open_cols(board,r,c):
	cols = []
	new_c = c
	while new_c+1 in range(8):
		new_c +=1
		if board[r][new_c] in our_pieces:
			break
		cols.append(new_c)
		if board[r][new_c] in their_pieces:
			break
	new_c = c
	while new_c-1 in range(8):
		new_c -=1
		if board[r][new_c] in our_pieces:
			break
		if board[r][new_c] in their_pieces:
			break
		cols.append(new_c)
	return cols

def open_b_diag(board, r,c):
	diags_1 = []
	row = r
	col = c
	while row < 7 and col < 7:
		row+=1
		col+=1
		if board[row][col] in our_pieces:
			break
		diags_1.append([row,col])
		if board[row][col] in their_pieces:
			diags_1.reverse()
			break
	diags_2 = []
	while row >0 and col >0:
		row-=1
		col-=1
		if board[row][col] in our_pieces:
			break
		diags_2.append([row,col])
		if board[row][col] in their_pieces:
			diags_2.reverse()
			break
	return [diags_1[0]] + [diags_2[0]] + diags_1[1:]+ diags_2[1:] if (diags_1!=[] and diags_2!=[]) else diags_1+diags_2

def open_f_diag(board, r,c):
	diags_1 = []
	row = r
	col = c
	while row < 7 and col > 0:
		row+=1
		col-=1
		if board[row][col] in our_pieces:
			break
		diags_1.append([row,col])
		if board[row][col] in their_pieces:
			diags_1.reverse()
			break
	diags_2=[]
	while row >0 and col <7:
		row-=1
		col+=1
		if board[row][col] in our_pieces:
			break
		diags_2.append([row,col])
		if board[row][col] in their_pieces:
			diags_2.reverse()
			break
	return [diags_1[0]] + [diags_2[0]] + diags_1[1:]+ diags_2[1:] if (diags_1!=[] and diags_2!=[]) else diags_1+diags_2


def next_p_move(board,r,c, curr_player):
	moves=[]
	if curr_player == 'w':
		if (r+1)<=7 and (c+1)<=7 and board[r+1][c+1] in their_pieces:
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r+1][c+1] = our_pieces[0]
			moves.append(move)
		if (r+1)<=7 and (c-1)>=0 and board[r+1][c-1] in their_pieces:
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r+1][c-1] = our_pieces[0]
			moves.append(move)
		if (r+1)<7 and board[r+1][c]=='.':
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r+1][c] = our_pieces[0]
			moves.append(move)
		elif (r+1)==7 and board[r+1][c]=='.': # pawn reaches the end of board and becomes queen
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r+1][c] = our_pieces[4]
			moves.append(move)
		if r==1 and board[r+1][c]=='.' and board[r+2][c]=='.': # two moves from initial position
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r+2][c] = our_pieces[0]
			moves.append(move)
	else:
		if (r-1)>=0 and (c+1)<=7 and board[r-1][c+1] in their_pieces:
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r-1][c+1] = our_pieces[0]
			moves.append(move)
		if (r-1)>=0 and (c-1)>=0 and board[r-1][c-1] in their_pieces:
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r-1][c-1] = our_pieces[0]
			moves.append(move)

		if (r-1)>0 and board[r-1][c]=='.':
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r-1][c] = our_pieces[0]
			moves.append(move)
		elif (r-1)==0 and board[r-1][c]=='.':# pawn reaches the end of board and becomes queen
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r-1][c] = our_pieces[4]
			moves.append(move)
		if r==6 and board[r-2][c]=='.'and board[r-1][c]=='.': # two moves from initial position
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r-2][c] = our_pieces[0]
			moves.append(move)
	return moves

def next_r_move(board, r,c):
	open_columns = open_cols(board,r,c)
	open_row = open_rows(board,r,c)
	moves = []
	for row in open_row:
		move = [[j for j in i]for i in board]
		move[r][c] = '.'
		move[row][c] = our_pieces[1]
		moves.append(move)
	for col in open_columns:
		move = [[j for j in i]for i in board]
		move[r][c] = '.'
		move[r][col] = our_pieces[1]
		moves.append(move)
	return moves

def next_n_move(board, r,c):
	moves = []
	if (r-2) in range(8) and (c+1) in range(8) and board[r-2][c+1] not in our_pieces:
		move = [[j for j in i]for i in board]
		move[r][c] = '.'
		move[r-2][c+1] = our_pieces[2]
		moves.append(move)
	if (r-1) in range(8) and (c+2) in range(8) and board[r-1][c+2] not in our_pieces:
		move = [[j for j in i]for i in board]
		move[r][c] = '.'
		move[r-1][c+2] = our_pieces[2]
		moves.append(move)
	if (r+1) in range(8) and (c+2) in range(8) and board[r+1][c+2] not in our_pieces:
		move = [[j for j in i]for i in board]
		move[r][c] = '.'
		move[r+1][c+2] = our_pieces[2]
		moves.append(move)
	if (r+2) in range(8) and (c+1) in range(8) and board[r+2][c+1] not in our_pieces:
		move = [[j for j in i]for i in board]
		move[r][c] = '.'
		move[r+2][c+1] = our_pieces[2]
		moves.append(move)
	if (r+2) in range(8) and (c-1) in range(8) and board[r+2][c-1] not in our_pieces:
		move = [[j for j in i]for i in board]
		move[r][c] = '.'
		move[r+2][c-1] = our_pieces[2]
		moves.append(move)
	if (r+1) in range(8) and (c-2) in range(8) and board[r+1][c-2] not in our_pieces:
		move = [[j for j in i]for i in board]
		move[r][c] = '.'
		move[r+1][c-2] = our_pieces[2]
		moves.append(move)
	if (r-1) in range(8) and (c-2) in range(8) and board[r-1][c-2] not in our_pieces:
		move = [[j for j in i]for i in board]
		move[r][c] = '.'
		move[r-1][c-2] = our_pieces[2]
		moves.append(move)
	if (r-2) in range(8) and (c-1) in range(8) and board[r-2][c-1] not in our_pieces:
		move = [[j for j in i]for i in board]
		move[r][c] = '.'
		move[r-2][c-1] = our_pieces[2]
		moves.append(move)
	return moves

def next_b_move(board,r,c):
	open_f_diags = open_f_diag(board, r,c)
	open_b_diags = open_b_diag(board,r,c)
	moves = []
	for position in open_f_diags:
		move = [[j for j in i]for i in board]
		move[r][c] = '.'
		move[position[0]][position[1]] = our_pieces[3]
		moves.append(move)
	for position in open_b_diags:
		move = [[j for j in i]for i in board]
		move[r][c] = '.'
		move[position[0]][position[1]] = our_pieces[3]
		moves.append(move)
	return moves

def next_q_move(board, r, c):
	open_columns = open_cols(board,r,c)
	open_row = open_rows(board,r,c)
	moves = []
	for row in open_row:
		move = [[j for j in i]for i in board]
		move[r][c] = '.'
		move[row][c] = our_pieces[4]
		moves.append(move)
	for col in open_columns:
		move = [[j for j in i]for i in board]
		move[r][c] = '.'
		move[r][col] = our_pieces[4]
		moves.append(move)
	open_f_diags = open_f_diag(board, r,c)
	open_b_diags = open_b_diag(board,r,c)
	for position in open_f_diags:
		move = [[j for j in i]for i in board]
		move[r][c] = '.'
		move[position[0]][position[1]] = our_pieces[4]
		moves.append(move)
	for position in open_b_diags:
		move = [[j for j in i]for i in board]
		move[r][c] = '.'
		move[position[0]][position[1]] = our_pieces[4]
		moves.append(move)
	return moves


def next_k_move(board,r,c):
	moves=[]
	# horizontal and vertical moves
	if r+1 <=7:
		if board[r+1][c] not in our_pieces:
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r+1][c] = our_pieces[5]
			moves.append(move)
	if r-1 >=0:
		if board[r-1][c] not in our_pieces:
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r-1][c] = our_pieces[5]
			moves.append(move)
	if c+1<=7:
		if board[r][c+1] not in our_pieces:
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r][c+1] = our_pieces[5]
			moves.append(move)
	if c-1>=0:
		if board[r][c-1] not in our_pieces:
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r][c-1] = our_pieces[5]
			moves.append(move)
	# diagonal moves
	if r+1<=7 and c+1<=7:
		if board[r+1][c+1] not in our_pieces:
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r+1][c+1] = our_pieces[5]
			moves.append(move)
	if r+1<=7 and c-1>=0:
		if board[r+1][c-1] not in our_pieces:
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r+1][c-1] = our_pieces[5]
			moves.append(move)
	if r-1>=0 and c+1<=7:
		if board[r-1][c+1] not in our_pieces:
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r-1][c+1] = our_pieces[5]
			moves.append(move)
	if r-1>=0 and c-1<=0:
		if board[r-1][c-1] not in our_pieces:
			move = [[j for j in i]for i in board]
			move[r][c] = '.'
			move[r-1][c-1] = our_pieces[5]
			moves.append(move)
	return moves


def successor(board, curr_player):
	successors = []
	if curr_player == 'w':
		globals()['our_pieces'] = ['P','R','N','B','Q','K']
		globals()['their_pieces'] = ['p','r','n','b','q','k']
		for r in range(8):
			for c in range(8):
				if board[r][c]!='.':
					if board[r][c]=='P':
						successors.extend(next_p_move(board,r,c, curr_player))
					elif board[r][c]=='R':
						successors.extend(next_r_move(board,r,c))
					elif board[r][c]=='N':
						successors.extend(next_n_move(board,r,c))
					elif board[r][c]=='B':
						successors.extend(next_b_move(board,r,c))
					elif board[r][c]=='Q':
						successors.extend(next_q_move(board,r,c))
					elif board[r][c]=='K':
						successors.extend(next_k_move(board,r,c))
	else:
		globals()['our_pieces'] = ['p','r','n','b','q','k']
		globals()['their_pieces'] = ['P','R','N','B','Q','K']
		for r in range(8):
			for c in range(8):
				if board[r][c]!='.':
					if board[r][c]=='p':
						successors.extend(next_p_move(board,r,c, curr_player))
					elif board[r][c]=='r':
						successors.extend(next_r_move(board,r,c))
					elif board[r][c]=='n':
						successors.extend(next_n_move(board,r,c))
					elif board[r][c]=='b':
						successors.extend(next_b_move(board,r,c))
					elif board[r][c]=='q':
						successors.extend(next_q_move(board,r,c))
					elif board[r][c]=='k':
						successors.extend(next_k_move(board,r,c))
	return successors

def evaluation(board):
	c=Counter([col for row in board for col in row])
	return 10*(c['P']-c['p']) + 70*(c['R']-c['r']) + 40*(c['N']-c['n']) + 100*(c['B']-c['b']) + 300*(c['Q']-c['q'])  if player == 'w' else 10*(c['p']-c['P']) + 70*(c['r']-c['R']) + 40*(c['n']-c['N']) + 100*(c['b']-c['B']) + 300*(c['q']-c['Q']) 

def terminal(board):
	white_king = sum([1 if 'K' in row else 0 for row in board]) 
	black_king = sum([1 if 'k' in row else 0 for row in board])
	if (white_king == 1 and black_king==0):
		return 'white' 
	elif (white_king==0 and black_king==1):
		return 'black' 
	else:
		return 'not_terminal'


def alphabeta(board):
	list_of_succ=[[successors, minval(successors,-infinity,infinity,0)] for successors in successor(board,player)]
	shuffle(list_of_succ)
	return max(list_of_succ, key = lambda x: x[1])


def minval(s,alpha,beta,depth):
	temp_terminal = terminal(s)
	if (temp_terminal =='white' and player =='w') or (temp_terminal =='black' and player =='b'):
		return infinity 
	elif (temp_terminal =='black' and player =='w') or (temp_terminal =='white' and player =='b'):
		return -infinity
	depth += 1
	# print('min val', alpha, beta, depth)
	# print_board(s)

	if depth <maxdepth:
		color = 'b' if player == 'w' else 'w'
		for s1 in successor(s,color):
			beta = min(beta,maxval(s1,alpha,beta,depth))
			if alpha>=beta:
				break
	elif depth == maxdepth:
		beta = evaluation(s)
	return beta

def maxval(s, alpha, beta,depth):
	temp_terminal = terminal(s)
	if (temp_terminal =='white' and player =='w') or (temp_terminal =='black' and player =='b'):
		return infinity 
	elif (temp_terminal =='black' and player =='w') or (temp_terminal =='white' and player =='b'):
		return -infinity

	depth +=1
	if depth < maxdepth:
		for s2 in successor(s,player):
			alpha = max(alpha,minval(s2,alpha,beta,depth))
			if alpha >= beta:
				break
	elif depth==maxdepth:
		alpha = evaluation(s)
	return alpha



player = sys.argv[1]
str_board= sys.argv[2]
time = sys.argv[3]


infinity = float('inf')

init_board=(np.array(list(str_board)).reshape(8,8)).tolist()
for i in range(3,6):
	maxdepth = i
	solve = alphabeta(init_board)
	print("".join([solve[0][i][j] for i in range(8) for j in range(8)]))



