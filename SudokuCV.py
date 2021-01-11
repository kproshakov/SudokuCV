
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from sudoku import SudokuSolver

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 60, kernel_size=5)
        self.conv2 = nn.Conv2d(60, 60, kernel_size=5)
        self.conv3 = nn.Conv2d(60, 30, kernel_size=3)
        self.conv4 = nn.Conv2d(30, 30, kernel_size=3)
        self.lin1 = nn.Linear(4*4*30, 500)
        self.lin2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p=0.5, training = self.training)
        x = x.view(-1, 4*4*30)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training = self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

class SudokuCV():
    def __init__(self):
        self.model = Model()
        self.model.load_state_dict(torch.load("./Classifier/model.pt"))

    def solve_sudoku_pic(self, img_path):
        num_impossible = 0

        img = cv2.imread(img_path)
        frame = cv2.resize(img, (450, 800))
        width = 450


        blur = cv2.GaussianBlur(frame, (7, 7), 0)

        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)


        imgCanny = cv2.Canny(gray, 40, 69)

        kernel = np.ones((5,5))
        imgDilated = cv2.dilate(imgCanny, kernel, iterations=1)

        contours, _ = cv2.findContours(imgDilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        tgray = gray
        thres = gray
        cell_size = 0
        while True:

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 15000:
                    epsilon = 0.1*cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    if len(approx) == 4:
                        approx = np.array(approx)
                        x_bar = (approx[0][0][0] + approx[1][0][0] + approx[2][0][0] + approx[3][0][0])//4
                        y_bar = (approx[0][0][1] + approx[1][0][1] + approx[2][0][1] + approx[3][0][1])//4
                        a1,a2,a3,a4 = 0,0,0,0
                        for i in approx:
                            if (i[0][0] < x_bar and i[0][1] < y_bar):
                                a1 += i[0] - np.array([1, -1])
                            elif (i[0][0] < x_bar and i[0][1] > y_bar):
                                a4 += i[0] - np.array([1, 1])
                            elif (i[0][0] > x_bar and i[0][1] < y_bar):
                                a2 += i[0] - np.array([-1, -1])
                            else:
                                a3 += i[0] - np.array([-1, 1])


                        pts1 = np.float32([a1, a2, a3, a4])
                        pts2 = np.float32([[0, 0], [width, 0], [width, width], [0, width]])

                        M = cv2.getPerspectiveTransform(pts1, pts2)
                        tgray = cv2.warpPerspective(gray, M, (int(width), int(width)))
                        warped_sudoku = cv2.warpPerspective(frame, M, (int(width), int(width)))
                        thres = cv2.adaptiveThreshold(tgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 6)
                        cell_size = width//9
                        grid= []
                        added = []
                        

                        for row in range(9):
                            a = []
                            r = []
                            for col in range(9):
                                a1 = [col*cell_size + 1, row*cell_size + 1]
                                a2 = [(col+1)*cell_size - 1, row*cell_size + 1]
                                a3 = [(col+1)*cell_size - 1, (row+1)*cell_size - 1]
                                a4 = [col*cell_size + 1, (row+1)*cell_size - 1]

                                pts1 = np.float32([a1, a2, a3, a4])
                                pts2 = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])

                                M = cv2.getPerspectiveTransform(pts1, pts2)
                                cell = cv2.warpPerspective(thres, M, (100, 100))
                                
                                # Draw a border
                                cell = cv2.line(cell, (0, 0), (100, 0), (255), 25)
                                cell = cv2.line(cell, (100, 0), (100, 100), (255), 25)
                                cell = cv2.line(cell, (100, 100), (0, 100), (255), 25)
                                cell = cv2.line(cell, (0, 0), (0, 100), (255), 25)

                                digit = np.array(np.float32(np.reshape(cv2.resize(cell, (32, 32)), (1, 32, 32))))
                                digit = np.array([digit])


                                var = Variable(torch.from_numpy(digit).type(torch.LongTensor)).float()
                                predictions = self.model.forward(var)
                                predicted_num = torch.max(predictions, 1)[1]

                                if np.mean(digit) >= 250:
                                    a.append(True)
                                    r.append(0)
                                else:
                                    a.append(False)
                                    r.append(int(predicted_num))

                            grid.append(r)
                            added.append(a)
                        # print(np.matrix(grid))
                        break

            s = SudokuSolver(grid=grid)
            if (s.is_valid()):
                s.solve()
                for row in range(9):
                    for col in range(9):
                        if added[row][col]:
                            cv2.putText(warped_sudoku, str(grid[row][col]), (col*cell_size + 10, (row+1)*cell_size - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 200), 2, cv2.LINE_AA)
                warped_sudoku = cv2.cvtColor(warped_sudoku, cv2.COLOR_BGR2RGB)
                return warped_sudoku
            else:
                num_impossible += 1
                if (num_impossible >= 10):
                    return img
                continue
