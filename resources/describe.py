{\rtf1\ansi\ansicpg1252\cocoartf1561\cocoasubrtf600
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 class describing:\
    \
    def __init__(self):\
        self = self\
        \
    def __counter(self, X, col):\
        Y = pd.DataFrame(X.filter([str(col)]))\
        Y.columns = ['col']\
        Y = Y.dropna().reset_index(drop=True)\
        Y['One'] = 1\
        return sum(Y['One'])\
    \
    def __meaner(self,X,col):\
        return sum(X[col].dropna())/self.__counter(X, col)\
    \
    def count(self, X, col):\
        return self.__counter(X,col)\
    \
    def mean(self,X,col):\
        return self.__meaner(X,col)\
    \
    def std(self, X, col):\
        New = pd.DataFrame(X[col].dropna().reset_index(drop=True))\
        std = pd.DataFrame((New[col] - self.__meaner(New,col))**2)\
        return np.sqrt(1/(self.__counter(std,col))*sum(std[col]))\
    \
    def minimum(self, X, col):\
        Y = pd.DataFrame(X[col])\
        m = Y[col].iloc[0]\
        for i in np.arange(1, len(Y[col])):\
            if (m > Y[col].iloc[i]): m = Y[col].iloc[i]\
        return m\
    \
    def maximum(self, X, col):\
        Y = pd.DataFrame(X[col])\
        ma = Y[col].iloc[0]\
        for i in np.arange(1, len(Y[col])):\
            if (ma < Y[col].iloc[i]): ma = Y[col].iloc[i]\
        return ma\
    \
    def quartile(self,X,col,percentage):\
        quart = pd.DataFrame(X[col].dropna().reset_index(drop=True))\
        quart = quart.sort_values(by=[col], ascending=True, axis=0).reset_index(drop=True)\
        return quart[col].iloc[int((len(quart[col])*percentage/100))]}