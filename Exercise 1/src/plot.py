#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def plot_and_save(csv,png):
    arr = np.loadtxt(csv, delimiter=" ", dtype=float)
    fig, ax = plt.subplots()
    ax.set(xlabel='x', ylabel='y',title=f'{csv}')
    plt.imshow(arr, cmap='RdBu', extent=[0, 1, 0, 1], origin="lower")
    plt.colorbar()
    fig.savefig(png)
    plt.show()

if __name__ == '__main__':
    p = argparse.ArgumentParser(prog='plot')
    p.add_argument('-i', '--csv', type=Path, required=True)
    p.add_argument('-o', '--png', type=Path, required=True)    
    args = p.parse_args()    
    plot_and_save(args.csv,args.png)


