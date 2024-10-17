# импортирование модулей, которые в дальнейшем будут использоваться в написании кода, из имеющихся библиотек
from collections import deque import numpy as np
import cv2 import argparse import math
from PIL import Image
# считывание изображения/видеоизображения/видеопотока с камеры
img = cv2.imread("123.jpg")
# значения начала и конца цветового диапазона фильтра
h_max = np.array((163, 20, 255), np.uint8)
h_min = np.array((120, 56, 50), np.uint8)
# создание окна для отображения информации с возможностью изменения параметров HSV( H - цвет, S - насыщенность, V - яркость).
cv2.namedWindow("setHSV")
# создание ползунков для параметров
cv2.createTrackbar('h1', 'setHSV', 0, 255, nothing)
cv2.createTrackbar('s1', 'setHSV', 0, 255, nothing)
cv2.createTrackbar('v1', 'setHSV', 0, 255, nothing)
cv2.createTrackbar('h2', 'setHSV', 0, 255, nothing)
cv2.createTrackbar('s2', 'setHSV', 0, 255, nothing)
cv2.createTrackbar('v2', 'setHSV', 0, 255, nothing)
cv2.createTrackbar('Area', 'setHSV', 0, 10000, nothing)
# создание переключателя
switch = '0 : OFF \n1 : ON'
# создание возможности менять положение ползунков для параметров
cv2.createTrackbar(switch, 'setHSV', 0, 1 ,nothing) cv2.setTrackbarPos('h1', 'setHSV',0) cv2.setTrackbarPos('s1', 'setHSV',0) cv2.setTrackbarPos('v1', 'setHSV',0) cv2.setTrackbarPos('h2', 'setHSV',171) cv2.setTrackbarPos('s2', 'setHSV',255) cv2.setTrackbarPos('v2', 'setHSV',250) cv2.setTrackbarPos('Area', 'setHSV',0)
# создание прямоугольных матриц разных размеров
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)) #матрица
1x1
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)) #матрица 10x10
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)) #матрица 21x21
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) #матрица 5x5
# значения необходимых параметров
maxRadius = 1
x = 0
y = 0
lastx = 0
lasty = 0

color_yellow = (0,255,255)


PixelSize = 0.0000014 #метров #dF = 1.35
F = 0.000000028 #фокусное расстояние линзы
# параметры маяков
L = 60 #расстояние между световыми маяками на посадочной прямой M2 = 4 #расстояние между световыми маяками на матрице
# задаем анализатор для работы с видео
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file") ap.add_argument("-b", "--buffer", type=int, default=64,help="max buffer
size")
args = vars(ap.parse_args())
pts = deque(maxlen=args["buffer"]) #двухсторонняя очередь, максимальная длина которой будет равна буферу = 64.
# создание списков
pts=[] pts1=[] pts2=[] pts3=[] pts4=[] cnts2=[] cnts3=[] i=0

X=[]
Y=[]
rad=[]

Center=[]


aM01=[] aM10=[]
adArea=[] M={}
# фильтрация изображения с камеры
while(True):
ret, img = cap.read() #получение кадра с камеры dimensions = img.shape #получение размера изображения
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #конвертация изображения из BRG в HSV пространство
h_max = np.array((1, 205, 255), np.uint8)
h_min = np.array((0, 0, 243), np.uint8)
thresh = cv2.inRange(hsv, h_min, h_max) #наложение фильтра на изображение в HSV
thresh2 = cv2.GaussianBlur(thresh, (9, 9), 6) #применение размытия по Гауссу
ret,thresh3 = cv2.threshold(thresh2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #определение порога
dilation = cv2.dilate(thresh3,kernel3,iterations = 5) #расширение изображения путем увеличения пикселей
# создание контуров и совмещение с изображением
edges = cv2.Canny(img,80,250) edges2 = cv2.bitwise_not(edges)
# выделение точек являющимися концами отрезков контуров
cnts = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
# cv2.RETR_EXTERNAL – извлечение только внешних контуров # cv2.CHAIN_APPROX_SIMPLE – удаление всех лишних точек # аппроксимация SIMPLE – контур хранится в виде отрезка
# создание списков
cnts1=[[0],[0],[0],[0]]
# цикл получения элементов списка по индексу
for i in range (len(cnts)):
cnts1[i]=cnts[i]
# создание минимальной окружности вокруг объекта
for i in range(len(cnts1)):
if len(cnts1[i]) > 6:
((x, y), radius) = cv2.minEnclosingCircle(cnts1[i]) X.append(int(x))
Y.append(int(y)) rad.append(int(radius)) M1=cv2.moments(cnts1[i]) M[i]=(M1)
# вычисление центра окружности маяка
for i in range (len(M)):
center = (int(M[i]["m10"] / M[i]["m00"]), int(M[i]["m01"] / M[i]["m00"])) Center.append(center)
pts.append(center)
# вырисовка окружности маяка и его центра на изображении
for i in range (len(Center)):
cv2.circle(img, Center[i], 60, (0, 255, 0), 5) #(изображение, координаты центра, радиус, цвет, толщина линии границы круга в пикселях)
for i in range (len(pts)): cv2.circle(img, pts[i], 1, (0, 255, 0), 4)

# удаление всех элементов из списка если их больше 1
if len(pts) > 1:
pts.clear()
if len(Center) > 1:
Center.clear()
# определение моментов
for i in range (len(M)):
M01 = M[i]['m01'] #момент первого порядка M10 = M[i]['m10'] #момент первого порядка dArea = M[i]['m00'] #момент нулевого порядка if dArea>10:
x1 = int(M10 / dArea) X.append(x1)
y1 = int(M01 /dArea) Y.append(y1)
# подписываются номера маяков на экране
cv2.putText(img, "ID-%d" % (i+1), (x1+10,y1- 10),cv2.FONT_HERSHEY_SIMPLEX, (0.7), color_yellow, 2)
#cv2.putText(img, "ID%d " % (i+1) + " %d-%d" % (x1,y1), (20,f), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,44,0),2)
# контур посадочной площадки на изображении
cv2.line(img, (X[0],Y[0]),(X[1],Y[1]), (0, 0, 255), 2) #(изображение, начало и конец отрезка, цвет, толщина)
cv2.line(img, (X[1],Y[1]),(X[3],Y[3]), (0, 0, 255), 2)
cv2.line(img, (X[3],Y[3]),(X[2],Y[2]), (0, 0, 255), 2)
cv2.line(img, (X[2],Y[2]),(X[0],Y[0]), (0, 0, 255), 2)
# нахождение центра изображение
ch = dimensions[0]//2 cw = dimensions[1]//2
dx = x1 - cw dy = y1 – ch
# круг в центре маяка, круг в центре квадрата и линия между ними на изображении
cv2.circle(img, (X[1],Y[1]), 10, (0,255,0), 2) #(изображение, центр, радиус, цвет, толщина)
cv2.circle(img, (cw,ch), 5, (0,255,0), 2)
cv2.line(img, (X[1],Y[1]), (cw,ch), (0,255,0), 2)
D2 = (math.sqrt(math.pow(X[4]-X[5],2)+math.pow(Y[4]-Y[5],2))) #корень из суммы квадратов
F = 0.278 Фокусное расстояние линзы
L = 2.9 #Расстояние между световыми маяками на посадке Q2 = 100 #Расстояние между световыми маяками на матрицу

R2 = (((F*L*Q2)/D2)*10000) #расчет R2
R31 = math.ceil(R2) #округление числа R2 до наименьшего целого числа, которое ≥ R2
val = ((D2/R31)/100)
fRoll =abs( math.ceil( math.degrees ( math.tan(val)))) #расчет крена
БПЛА
cv2.putText(img, "LA", (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
(0,0,255),2)
cv2.putText(img, "Kren: "+ str(fRoll), (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
fX1 = math.ceil (D2/(F*L*Q2))
Y2_sht = x1+(((x1-y1)*(1+Q2))/(2*Q2)) pl = ((fX1*(fX1+Q3))/(Q3*L*F))
fY1 = (y1*(Y2_sht - dy)) pv1 = dx / F
pv2 = D2 *(5)/(2*F*Q2)
pv3 = (fX1 + Q3*L) / (Q3*L*F)


pv4 = Y2_sht -x1
fPitch = math.ceil ((pv1 + pv2 - pv3 * pv4)/ 100) #расчет тангажа БПЛА pz1 = (((fX1*(fX1 + Q3*L))*(dx))/(F * Q3 * L))
pz2 = ((((fX1 + Q3*L)/(Q2*L)) + (fX1*L)) * (2*Q3*L))
fZl = pz1 + pz2
fYaw = math.ceil (((dx / F) + (L/2 + fZl) / (fX1 - Q3*L))/100) #расчет курса БПЛА
cv2.putText(img, "Kyrs: "+ str(fYaw), (10,200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
cv2.putText(img, "Tangahg: "+ str(fPitch), (10,220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
S = math.ceil((D2*D2*D2*D2)/10000000000) #Расчет площади посадочной площадки
Tlof1 = math.ceil(D2) #Расчет радиуса посадочной зоны Tlof2 = Tlof1/10
# вывод текста/информации на экран
cv2.putText(img, "Tlof : "+ str(Tlof2), (10,320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255),2) #(изображение, текст, координаты, тип шрифта, размер шрифта, цвет, толщина пера)
cv2.putText(img, "Sk : "+ str(S), (10,300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255),2)
cv2.putText(img, "dY: "+ str(dy), (10,400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
cv2.putText(img, "dX: "+ str(dx), (10,380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
cv2.putText(img, "lights", (10,360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
cv2.putText(img, "dZ: "+ str(R31), (10,420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.putText(img, "before touching: "+ str(R31), (10,260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),2)
M={}
# удаление всех элементов из списка
X.clear()
Y.clear() Center.clear()
# отображение/вывод изображения на экран
cv2.imshow('Clear2',edges) #(заголовок окна, в котором будет показано изображение, какое изображение выводим)
cv2.imshow('Clear9',edges2) cv2.imshow('Clear4',img) cv2.imshow('Clear1',hsv) cv2.imshow('Clear3',thresh) cv2.imshow('Clear8',dilation) cv2.imshow('Clear5',thresh2) cv2.imshow('Clear6',thresh3) #cv2.imshow('Clear7',erosion)
# закрытие окна при нажатии кнопки esc
ch = cv2.waitKey(5) #ожидание 5 миллисекунд if ch == 27:
break
# обозначение конца выполнения кода, освобождение оперативной памяти и закрытие всех окон в скрипте
cap.release() cv2.destroyAllWindows()
