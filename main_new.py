# tic tac toe game
import datetime
import random

import numpy as np
import pandas as pd

# Game variables
board = [' ' for x in range(10)]  # board
pBoard = ''                       # Board output
movesHist = []                    # Intialise data for game history.
allMovesHist = []                 # Intialise data for game history all moves
allWinningMovesHist = []          # Intialise data for game history all moves
nrOfDraws = 0
nrOfXWins = 0
nrOfOWins = 0
nrOfHWins = 0
nrOfUWins = 0
nrOfSWins = 0
nrOfLWins = 0


# Global varables Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(18,)))
model.add(Dense(12, activation='relu'))          # Add one hidden layer
model.add(Dense(12, activation='relu'))          # Add one hidden layer
model.add(Dense(9, activation='softmax'))        # Add an output layer     //
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


class Game:

    menuSel = ''
    player1 = ''
    player2 = ''

    def __init__(self):
        print('Game init')

    def Menu(self):
        print('---------------------------------------------------------------------------------')
        print('                                  TIC TAC TOE')
        print('Training a Computer to play Tic Tac Toe.  Teaching a human about Neural Networks.')
        print('---------------------------------------------------------------------------------')
        # print('')
        print('Options:')
        print('1. Human vs Human')
        print('2. Human vs untrained agent')
        print('3. Untrained agent vs itself')
        print('4. Train Agent Smit')
        print('5. Human vs Agent Smit')
        print('6. Agent Smit vs untrained agent')
        print('7. Agent Smit vs Agent Smit')
        print('8. Train Linki')
        print('9. Human vs Agent Linki')
        print('10. Agent Linki vs untrained agent')
        print('11. Help')

        print('')

        # get selection
        self.menuSel = input('Enter selection: ')

        # set players
        self.player1, self.player2 = self.getPlayers()

        if self.menuSel == '11':
            print('Please refer to README.md')

    def getPlayers(self):
        if self.menuSel == '1':
            return 'H', 'H'
        if self.menuSel == '2':
            return 'H', 'U'
        if self.menuSel == '3':
            return 'U', 'U'
        if self.menuSel == '4':
            return 'U', 'U'
        if self.menuSel == '5':
            return 'H', 'S'
        if self.menuSel == '6':
            return 'U', 'S'
        if self.menuSel == '7':
            return 'S', 'S'
        if self.menuSel == '8':             # just training
            return 'H', 'L'
        if self.menuSel == '9':
            return 'H', 'L'
        if self.menuSel == '10':
            return 'U', 'L'
        if self.gmenuSel == '11':
            return '', ''

# board to X
def boardToX(iboard):
    iX = np.zeros(18, dtype=int)
    iX[0] = iconv(iboard[1], 'X')
    iX[1] = iconv(iboard[2], 'X')
    iX[2] = iconv(iboard[3], 'X')
    iX[3] = iconv(iboard[4], 'X')
    iX[4] = iconv(iboard[5], 'X')
    iX[5] = iconv(iboard[6], 'X')
    iX[6] = iconv(iboard[7], 'X')
    iX[7] = iconv(iboard[8], 'X')
    iX[8] = iconv(iboard[9], 'X')
    iX[9] = iconv(iboard[1], 'O')
    iX[10] = iconv(iboard[2], 'O')
    iX[11] = iconv(iboard[3], 'O')
    iX[12] = iconv(iboard[4], 'O')
    iX[13] = iconv(iboard[5], 'O')
    iX[14] = iconv(iboard[6], 'O')
    iX[15] = iconv(iboard[7], 'O')
    iX[16] = iconv(iboard[8], 'O')
    iX[17] = iconv(iboard[9], 'O')

    return iX

# Insert a letter in the board
def insertLetter(letter, pos, gameNr):
    iboard = board[:]

    iX = boardToX(iboard)

    iY = np.zeros(9, dtype=int)
    if pos == 1:
        iY[0] = '1'
    if pos == 2:
        iY[1] = '1'
    if pos == 3:
        iY[2] = '1'
    if pos == 4:
        iY[3] = '1'
    if pos == 5:
        iY[4] = '1'
    if pos == 6:
        iY[5] = '1'
    if pos == 7:
        iY[6] = '1'
    if pos == 8:
        iY[7] = '1'
    if pos == 9:
        iY[8] = '1'

    movesHist.append({'GameNr': gameNr,
                      'Winner': '',

                      # Save the state can be used by other training models
                      'S1': iboard[1],
                      'S2': iboard[2],
                      'S3': iboard[3],
                      'S4': iboard[4],
                      'S5': iboard[5],
                      'S6': iboard[6],
                      'S7': iboard[7],
                      'S8': iboard[8],
                      'S9': iboard[9],

                      # X and Y of neural network prepared
                      'iX': iX,
                      'iY': iY,
                      'NextMove': pos,
                      'Player': letter})

    board[pos] = letter


# Check if the space is available
def spaceIsFree(pos):
    return board[pos] == ' '


# Print board
def printBoard(i_board):
    if pBoard == 'y' or pBoard == 'Y':
        print('')
        print(i_board[1] + '|' + i_board[2] + '|' + i_board[3])
        print('-----')
        print(i_board[4] + '|' + i_board[5] + '|' + i_board[6])
        print('-----')
        print(i_board[7] + '|' + i_board[8] + '|' + i_board[9])
        print('')


# Check if input is winner
def isWinner(bo, le):
    return ((bo[7] == le and bo[8] == le and bo[9] == le) or
            (bo[4] == le and bo[5] == le and bo[6] == le) or
            (bo[1] == le and bo[2] == le and bo[3] == le) or
            (bo[1] == le and bo[4] == le and bo[7] == le) or
            (bo[2] == le and bo[5] == le and bo[8] == le) or
            (bo[3] == le and bo[6] == le and bo[9] == le) or
            (bo[1] == le and bo[5] == le and bo[9] == le) or
            (bo[3] == le and bo[5] == le and bo[7] == le))


def saveWinner(let, GameNr, xPlayer, oPlayer):
    global nrOfXWins, nrOfOWins, nrOfDraws, nrOfHWins, nrOfUWins, nrOfSWins, nrOfLWins

    if pBoard == 'Y' or pBoard == 'y':
        if let == '-':
            print('Draw')
        else:
            print(let + ' won!')

    i = 0

    if (let == 'O' and oPlayer == 'U') or (let == 'X' and xPlayer == 'U'):
        # save the winner in the list
        while i < len(movesHist):
            if GameNr == movesHist[i]['GameNr']:
                movesHist[i]['Winner'] = let
                allMovesHist.append(movesHist[i])
                if movesHist[i]['Player'] == let:
                    allWinningMovesHist.append(movesHist[i])

            i = i + 1

        # 1. take last win
        # 2. save the winning move
        # 3. go to last loosing move
        i = i - 1            # move index back to last i
        # print(let)
        # print(movesHist[i]['Player'])
        if movesHist[i]['Player'] == let:

            improveMove = movesHist[i-1]
            improveMove['NextMove'] = movesHist[i]['NextMove']
            improveMove['iY'] = movesHist[i]['iY']

            # print(improveMove)

            # append this 10times.                  # this is a loosing move and should be penalised
            allWinningMovesHist.append(improveMove)
            allWinningMovesHist.append(improveMove)

    # save the number of wins
    if let == 'X':
        if pBoard == 'Y' or pBoard == 'y':
            print(xPlayer)
        nrOfXWins = nrOfXWins + 1
        if xPlayer == 'H':                  # Human
            nrOfHWins = nrOfHWins + 1

        if xPlayer == 'U':                  # Untrained
            nrOfUWins = nrOfUWins + 1

        if xPlayer == 'S':                  # Agent Smit
            nrOfSWins = nrOfSWins + 1

        if xPlayer == 'L':                  # Agent Smit
            nrOfLWins = nrOfLWins + 1

    if let == 'O':
        if pBoard == 'Y' or pBoard == 'y':
            print(oPlayer)
        nrOfOWins = nrOfOWins + 1
        if oPlayer == 'H':                  # Human
            nrOfHWins = nrOfHWins + 1
        if oPlayer == 'U':                  # Untrained
            nrOfUWins = nrOfUWins + 1
        if oPlayer == 'S':                  # Agent Smit
            nrOfSWins = nrOfSWins + 1
        if oPlayer == 'L':                  # Agent Linki
            nrOfLWins = nrOfLWins + 1

    if let == '-':
        nrOfDraws = nrOfDraws + 1

    movesHist.clear()


def makePrediction(iboard, mode):

    prednr = 0
    iX = np.zeros((1, 18), dtype=int)
    iX[0] = np.array(boardToX(iboard))

    if pBoard == 'y' or pBoard == 'Y':
        print(iX)    # print(iX.shape)

    #ynew = model.predict_proba(iX)
    if mode == '9' or mode == '10':
        ynew = loaded_modelL.predict_proba(iX)
        if np.sum(iX[0]) == 0:
            ynew = np.random.multinomial(1, ynew[0])
    else:
        ynew = loaded_modelS.predict_proba(iX)
        if np.sum(iX[0]) == 0:
            ynew = np.random.multinomial(1, ynew[0])
    if pBoard == 'y' or pBoard == 'Y':
        print(np.around(ynew, decimals=3))

    prednr = np.argmax(ynew) + 1

    if pBoard == 'y' or pBoard == 'Y':
        # print(ynew)
        print(prednr)

    return prednr


# make move using mode
def makeMove(let, mode, GameNr):
    global pBoard
    run = True
    while run:
        if mode == '1':  # Human
            move = input('Player ' + let + ' select a position: ')
        elif mode == '2':  # Human
            move = input('Player ' + let + ' select a position: ')
        elif mode == '5' or mode == '6' or mode == '7' or mode == '9' or mode == '10':  # Agent Smit/Linki
            move = makePrediction(board, mode)

        else:  # Random
            move = random.randint(1, 9)
            # print('Random move is:' + str(move))
        try:
            move = int(move)
            if 0 < move < 10:
                if spaceIsFree(move):
                    run = False
                    insertLetter(let, move, GameNr)
                elif mode == '1':
                    print('Space is occupied!')
        except:
            print('error')

    if isBoardFull(board):
        run = False


# check if the board is full
def isBoardFull(board):
    if board.count(' ') > 1:
        return False
    else:
        return True


# ---------------------------------------------------------------------------------------------------------------------#
# Enter Game
# ---------------------------------------------------------------------------------------------------------------------#
def enterGame(gameMode, xPlayer, oPlayer, GameNr=1, intMode = ''):

    printBoard(board)

    game_end = False
    let = 'O'


    while not game_end:

        if let == 'X':
            let = 'O'
        else:
            let = 'X'

        # make move based on Mode
        if gameMode == '1':  # Human vs Human
            makeMove(let, gameMode, GameNr)

        if gameMode == '2':  # Human vs Random
            if intMode == '' or intMode == '3':
                intMode = '1'
            else:
                intMode = '3'
            makeMove(let, intMode, GameNr)

        if gameMode == '3':  # Random vs Random
            makeMove(let, gameMode, GameNr)

        if gameMode == '5':  # Human vs Agent Smit
            if intMode == '' or intMode == '5':
                intMode = '1'
            else:
                intMode = '5'
            makeMove(let, intMode, GameNr)

        if gameMode == '6':  # Random vs Agent Smit
            if intMode == '' or intMode == '6':
                intMode = '3'
            else:
                intMode = '6'
            makeMove(let, intMode, GameNr)

        if gameMode == '7':  # Agent Smit vs Agent Smit
            makeMove(let, gameMode, GameNr)

        if gameMode == '9':  # Human vs Agent Linki
            if intMode == '' or intMode == '9':
                intMode = '1'
            else:
                intMode = '9'
            makeMove(let, intMode, GameNr)

        if gameMode == '10':  # Agent Linki vs Untrained
            if intMode == '' or intMode == '10':
                intMode = '3'
            else:
                intMode = '10'
            makeMove(let, intMode, GameNr)

        printBoard(board)


        # determine winner/tie or continue game
        if isWinner(board, let):  # O Winner
            saveWinner(let, GameNr, xPlayer, oPlayer)
            game_end = True


        elif isBoardFull(board):
            saveWinner('-', GameNr, xPlayer, oPlayer)
            game_end = True



def outputGameStats(iGameNr):

    iper = int(round(nrOfDraws / int(iGameNr) * 100))
    print('Drawn: ' + str(nrOfDraws) + ' (' + str(iper) + '%)')
    iper = int(round(nrOfXWins / int(iGameNr) * 100))
    print('X Won: ' + str(nrOfXWins) + ' (' + str(iper) + '%)')
    iper = int(round(nrOfOWins / int(iGameNr) * 100))
    print('O Won: ' + str(nrOfOWins) + ' (' + str(iper) + '%)')

    print('Nr of Human wins: ' + str(nrOfHWins))
    print('Nr of Untrained Agent wins: ' + str(nrOfUWins))
    print('Nr of Agent Smit wins: ' + str(nrOfSWins))
    print('Nr of Agent Linki wins: ' + str(nrOfLWins))


def GameStats(gameNr, gameMode):

    print('')
    print('Saving moves...')
    histDf = pd.DataFrame(allMovesHist, columns=['GameNr',
                                                 'Winner',
                                                 'S1',
                                                 'S2',
                                                 'S3',
                                                 'S4',
                                                 'S5',
                                                 'S6',
                                                 'S7',
                                                 'S8',
                                                 'S9',
                                                 'iX',
                                                 'iY',
                                                 'NextMove',
                                                 'Player'])
    #
    # histDf.to_csv('moveHistory.csv')

    # justResultDf = histDf[['GameNr', 'Winner']]
    # justResultDf = justResultDf.drop_duplicates()

    print('Output game stats...')
    outputGameStats(gameNr)

#   print out X and y training data to file
    print('Prep X and y...')
    iX = np.zeros((len(allWinningMovesHist), 18), dtype=int)
    iY = np.zeros((len(allWinningMovesHist), 9), dtype=int)
    i = 0
    while i < len(allWinningMovesHist):
        iX[i] = allWinningMovesHist[i]['iX']
        iY[i] = allWinningMovesHist[i]['iY']
        i = i + 1

    print('Save X...')
    filename = 'TrainingData/iX_' + gameMode + '_' + str(gameNr) + '_.csv'
    np.savetxt(filename, iX, delimiter=' ', fmt='%d')

    print('Save y...')
    filename = 'TrainingData/iY_' + gameMode + '_' + str(gameNr) + '_.csv'
    np.savetxt(filename, iY, delimiter=' ', fmt='%d')



def iconv(idata, input):
    ret = 0
    if idata == input:
        ret = 1

    return int(ret)


def enterTraining(gameMode):

    print(datetime.datetime.now())

    if gameMode == '4':             # train agent Smit

        print('Load data to train Agent Smit...')

        # X1 = np.loadtxt("TrainingData/iX_3000000_.csv", dtype='int')
        X = np.loadtxt(inputFileX, dtype='int')

        print(X.shape)

        # y1 = np.loadtxt("TrainingData/iY_3000000_.csv", dtype='int')
        y = np.loadtxt(inputFileY, dtype='int')
        print(y.shape)

    elif gameMode == '8':           # Train Agent Linki

        # load X
        X1 = np.loadtxt(inputFileX, dtype='int')
        print(X1.shape)

        X2 = np.loadtxt(inputFileXR, dtype='int')
        print(X2.shape)

        if inputFileXR2 != '':
            X3 = np.loadtxt(inputFileXR2, dtype='int')
            print(X3.shape)
            X = np.vstack((X1, X2, X3))
            X_test = X3
        elif inputFileXR != '':
            X = np.vstack((X1, X2))
            X_test = X2
        else:
            X = X1

        # load Y
        y1 = np.loadtxt(inputFileY, dtype='int')
        print(y1.shape)

        y2 = np.loadtxt(inputFileYR, dtype='int')
        print(y2.shape)
        if inputFileYR2 != '':
            y3 = np.loadtxt(inputFileYR2, dtype='int')
            print(y3.shape)
            y = np.vstack((y1, y2, y2, y3, y3, y3))
            y_test = y3
        elif inputFileYR != '':
            y = np.vstack((y1, y2, y2))
            y_test = y2
        else:
            y = y1


    X_train = X
    y_train = y

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


    print(X_train.shape)
    print(y_train.shape)

    # 9. Fit model on training data
    model.fit(X_train, y_train, batch_size=1000, nb_epoch=10, verbose=1)

    # 10. Evaluate model on test data
    if inputFileYR != '':
        print('Evaluate Model')
        score = model.evaluate(X_test, y_test, verbose=0)

    print(score)

    # save the model
    # serialize model to JSON
    if gameMode == '4':             # train agent Smit
        imodeljson = 'modelAgentSmit.json'
        imodelh5 = 'modelAgentSmit.h5'

    elif gameMode == '8':
        imodeljson = 'modelAgentLinki.json'
        imodelh5 = 'modelAgentLinki.h5'


    model_json = model.to_json()
    with open(imodeljson, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(imodelh5)
    print('Saved model to disk (' + imodelh5 + ')')


def load_model(agent):
    if agent == 'S':
        # load json and create model
        json_file = open('modelAgentSmit.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("modelAgentSmit.h5")
        print("Loaded Agent Smit from disk")
        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return loaded_model

    if agent == 'L':
        # load json and create model
        json_file = open('modelAgentLinki.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_modelL = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_modelL.load_weights("modelAgentLinki.h5")
        print("Loaded Agent Linki from disk")
        # evaluate loaded model on test data
        loaded_modelL.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return loaded_modelL

# ---------------------------------------------------------------------------------------------------------------------#
# Main program
# ---------------------------------------------------------------------------------------------------------------------#
# create instance of class myGame
myGame = Game()

# show menu and get user feedback on action to take
myGame.Menu()

# set game mode to Menu selection
gameMode = myGame.menuSel

# get players (is player1 X or O, is player2 X or O
player1, player2 = myGame.getPlayers()

# load Agent Smit
if myGame.player1 == 'S' or myGame.player2 == 'S':
    loaded_modelS = load_model('S')

# load Agent Linki
if myGame.player1 == 'L' or myGame.player2 == 'L':
    loaded_modelL = load_model('L')

# game modes
if gameMode != '4' and gameMode != '8':
    gameNr = input('How many games: ')
    pBoard = input('Print board (y/n): ')

    i = 0
    while i < int(gameNr):
        movesHist = []
        board[0:10] = [' ' for x in range(10)]
        i = i + 1
        enterGame(gameMode, player1, player2, i)

        if gameMode == '6' or gameMode == '10':
            board[0:10] = [' ' for x in range(10)]
            i = i + 1
            enterGame(gameMode, player2, player1, i, '3')      # swap x an o players

        if ((i/100000).is_integer() and i != 0) or i == int(gameNr):
            print(str(i) + ' games played...')

    GameStats(gameNr, gameMode)


# training mode
if gameMode == '4' or gameMode == '8':
    inputFileX = input('Training data file name (for example "TrainingData/iX_3_8000000_.csv": ')
    inputFileY = input('Training data file name (for example "TrainingData/iY_3_8000000_.csv": ')

    if gameMode == '8':
        inputFileXR = input('Additional training data file name (for example "TrainingData/iX_6_1000000_.csv": ')
        inputFileYR = input('Additional training data file name (for example "TrainingData/iY_6_1000000_.csv": ')
        if inputFileXR != '':
            inputFileXR2 = input('Additional training data file name (for example "TrainingData/iX_10_1000000_.csv": ')
            inputFileYR2 = input('Additional training data file name (for example "TrainingData/iY_10_1000000_.csv": ')
    else:
        inputFileXR = ''
        inputFileYR = ''
        inputFileXR2 = ''
        inputFileYR2 = ''
    print('Start training')
    enterTraining(gameMode)


print('The End')

# ---------------------------------------------------------------------------------------------------------------------#


