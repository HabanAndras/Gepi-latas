import cv2
import numpy as np
from scipy.stats import itemfreq
import glob


def get_dominant_color(image, n_colors):
    pixels = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    flags, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    return palette[np.argmax(itemfreq(labels)[:, -1])]

bh=0
e=0
j=0
b=0
ej=0
eb=0

cv_img = []
for i in glob.glob("Behajtanitilos/*.jpg"):
    n= cv2.imread(i)
    cv_img.append(n)

kepek=len(cv_img)
for i in range(kepek):
    img=cv_img[i]
     


###
    meret = 500
    height, width,channels = img.shape 

    arany = height/width

    meretarany = int(round(meret*arany))
    meretk = cv2.resize(img, (meret,meretarany), interpolation = cv2.INTER_AREA)
 
    frame = meretk
    
#      #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
#      #lower_red = np.array([50,50,50]) 
#      #upper_red = np.array([130,255,255])
#      #mask = cv2.inRange(hsv, lower_red, upper_red)
#      #res = cv2.bitwise_and(frame,frame, mask= mask)
    

    
    szurke = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    homaly= cv2.medianBlur(szurke,11)
    korok = cv2.HoughCircles(homaly, cv2.HOUGH_GRADIENT,
                              1, 50, param1=190, param2=100)
    
    
    if not korok is None:
        korok = np.uint16(np.around(korok))
        max_r, max_i = 0, 0
        for i in range(len(korok[:, :, 2][0])):
            if korok[:, :, 2][0][i] > 50 and korok[:, :, 2][0][i] > max_r:
                max_i = i
                max_r = korok[:, :, 2][0][i]
        x, y, r = korok[:, :, :][0][max_i]
        if y > r and x > r:
            negyzet = frame[y-r:y+r, x-r:x+r]

            szin = get_dominant_color(negyzet, 2)
            if szin[2] > 100:
                szoveg="Behajtani tilos!"
                bh=bh+1
            elif szin[0] > 80:
                balnyil= negyzet[negyzet.shape[0]*3//8:negyzet.shape[0]
                                * 5//8, negyzet.shape[1]*1//8:negyzet.shape[1]*3//8]
                balnyil_szin = get_dominant_color(balnyil, 1)
                
                felnyil = negyzet[(negyzet.shape[0]*1//8)-30:(negyzet.shape[0]
                                * 3//8)-30, (negyzet.shape[1]*3//8):(negyzet.shape[1]*5//8)]
                felnyil_szin = get_dominant_color(felnyil, 1)
                
                jobbnyil= negyzet[negyzet.shape[0]*3//8:negyzet.shape[0]
                                * 5//8, negyzet.shape[1]*5//8:negyzet.shape[1]*7//8]
                jobbnyil_szin = get_dominant_color(jobbnyil, 1)

                if felnyil_szin[2] < 60:
                    if sum(balnyil_szin) > sum(jobbnyil_szin):
                        szoveg = "Balra!"
                        b=b+1
                        cv2.imshow('Bal',balnyil)
                    else:
                        szoveg = "Jobbra!"
                        j=j+1
                        cv2.imshow('Jobb',jobbnyil)
                else:
                    if sum(felnyil_szin) > sum(balnyil_szin) and sum(felnyil_szin) > sum(jobbnyil_szin):
                        szoveg = "Egyenesen!"
                        e=e+1
                        cv2.imshow('Egyenes',felnyil)
                    elif sum(balnyil_szin) > sum(jobbnyil_szin):
                        szoveg = "Egyenesen es balra!"
                        eb=eb+1
                        cv2.imshow('Bal',balnyil)
                        cv2.imshow('Egyenes',felnyil)
                    else:
                        szoveg = "Egyenesen es jobbra!"
                        ej=ej+1
                        cv2.imshow('Jobb',jobbnyil)
                        cv2.imshow('Egyenes',felnyil)
            else:
                szoveg = "Nem tabla!"

        for i in korok[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2) 
            cv2.putText(frame,szoveg,(i[0]-150,i[1]),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255),2, lineType=cv2.LINE_AA) 
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    #cv2.imshow('mask',zone_2) 
    #cv2.imshow('res',res)
    cv2.imshow('Aktualis kep', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print("A behajtani tilos tablakat a program ekkora százalékban ismeri fel:",round(bh/(kepek-2)*100,2),"%")
# print("A kötelező haladási irány jobbra tablakat a program ekkora százalékban ismeri fel:",round(j/kepek*100,2),"%")
# print("A kötelező haladási irány balra tablakat a program ekkora százalékban ismeri fel:",round(b/(kepek)*100,2),"%")
# print("A kötelező haladási irány egyenesen tablakat a program ekkora százalékban ismeri fel:",round(e/kepek*100,2),"%")
# print("A kötelező haladási irány jobbra és egyenesen tablakat a program ekkora százalékban ismeri fel:",round(ej/kepek*100,2),"%")
# print("A kötelező haladási irány balra és egyenesen tablakat a program ekkora százalékban ismeri fel:",round(eb/kepek*100,2),"%")
