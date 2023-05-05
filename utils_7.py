import pandas as pd
import numpy as np

def get_minor_matrix(A: pd.DataFrame, i, j):
    "Zwraca macierz powstałą przez wykreślenie wiersza i i kolumny j z macierzy A."
    B = A.to_numpy()
    result =  np.delete(np.delete(B, [j], axis = 1), [i], axis = 0)
    return result

def get_cofactor(A: pd.DataFrame, i, j):
    """Zwraca dopełnienie algebraiczne elementu NAZWANEGO i, j macierzy A (pd.DataFrame)"""
    # znajdujemy numery indeksów i,j w uciętej macierzy
    i_ = np.where(A.index == i)[0][0]
    j_ = np.where(A.index == j)[0][0]
    A_ = get_minor_matrix(A, i_, j_)
    return np.linalg.det(A_)*(-1)**(i_+j_)

def get_partial_ij_ks(corr: np.array, i: int, j: int, ks: list) -> float:
    """Zwraca wartość korelacji zmiennych i,j pod warunkiem zmiennych ks."""
    # dla przekątnej zwracamy 1
    if i == j:
        return 1.00
    # przesuwamy indeksy o 1 dla zgodności z zadaniem i tworzymy uciętą macierz
    named_corr = pd.DataFrame(corr)
    named_corr.columns = range(1,len(corr)+1) 
    named_corr.index = range(1,len(corr)+1)
    reduced_corr = named_corr.loc[[i,j] + ks,[i,j] + ks]
    
    # obliczamy dopełnienia algebraiczne
    C_ii = get_cofactor(reduced_corr, i, i)
    C_jj = get_cofactor(reduced_corr, j, j)
    C_ij = get_cofactor(reduced_corr, i, j)
    return - C_ij / (np.sqrt(C_ii*C_jj))